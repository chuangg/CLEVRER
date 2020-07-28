import os
import cv2
import sys
import random
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import copy
import gzip
import pickle
import h5py
import json

import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from models import PropagationNetwork

from utils import count_parameters
from utils import prepare_relations, convert_mask_to_bbox, crop, encode_attr
from utils import normalize, check_attr, get_identifier, get_identifiers
from utils import check_same_identifier, check_same_identifiers, check_contain_id
from utils import get_masks, check_valid_masks, check_duplicate_identifier
from utils import rand_float, init_stat, combine_stat, load_data, store_data
from utils import decode, make_video, Tee


parser = argparse.ArgumentParser()
parser.add_argument('--pn', type=int, default=1)
parser.add_argument('--pstep', type=int, default=2)
parser.add_argument('--n_rollout', type=int, default=0)
parser.add_argument('--epoch', type=int, default=0)
parser.add_argument('--iter', type=int, default=0)
parser.add_argument('--env', default='CLEVR')
parser.add_argument('--dt', type=float, default=1./5.)
parser.add_argument('--train_valid_ratio', type=float, default=0.90909)
parser.add_argument('--time_step', type=int, default=0)
parser.add_argument('--nf_relation', type=int, default=128)
parser.add_argument('--nf_particle', type=int, default=128)
parser.add_argument('--nf_effect', type=int, default=128*4)
parser.add_argument('--filter_cnter', type=int, default=10)

parser.add_argument('--st_idx', type=int, default=0)
parser.add_argument('--ed_idx', type=int, default=0)
parser.add_argument('--video', type=int, default=0)
parser.add_argument('--store_img', type=int, default=0)
parser.add_argument('--back_ground', default='background.png')

parser.add_argument('--H', type=int, default=100)
parser.add_argument('--W', type=int, default=150)
parser.add_argument('--bbox_size', type=int, default=24)
parser.add_argument('--data_dir', default='frames')
parser.add_argument('--label_dir', default='derender_proposals')
parser.add_argument('--des_dir', default='propnet_predictions')
parser.add_argument('--outf', default='files')
parser.add_argument('--dataf', default='data')
parser.add_argument('--evalf', default='eval')
parser.add_argument('--eval_type', default='valid', help='rollout|valid')
parser.add_argument('--eval', type=int, default=1)
parser.add_argument('--edge_superv', type=int, default=1, help='whether to include edge supervision')
parser.add_argument('--use_attr', type=int, default=1, help='whether using attributes or not')

parser.add_argument('--eval_part', default='valid')

parser.add_argument('--n_his', type=int, default=2)
parser.add_argument('--frame_offset', type=int, default=5)

parser.add_argument('--what_if', type=int, default=-1)

# object attributes:
# [rubber, metal, cube, cylinder, sphere]
parser.add_argument('--attr_dim', type=int, default=5)

# object state:
# [mask, x, y, r, g, b]
parser.add_argument('--state_dim', type=int, default=6)

# relation:
# [collision, dx, dy]
parser.add_argument('--relation_dim', type=int, default=3)

args = parser.parse_args()


if args.env == 'CLEVR':
    args.n_rollout = 11000
    args.time_step = 100
else:
    raise AssertionError("Unsupported env")

args.outf = args.outf + '_' + args.env
if args.use_attr == 0:
    args.outf += '_noAttr'
    args.des_dir += '_noAttr'
if args.edge_superv == 0:
    args.outf += '_noEdgeSuperv'
    args.des_dir += '_noEdgeSuperv'
if args.pn:
    args.outf += '_pn'
args.outf += '_pstep_' + str(args.pstep)
# args.dataf = args.dataf + '_' + args.env
args.evalf = args.evalf + '_' + args.env
if args.use_attr == 0:
    args.evalf += '_noAttr'
if args.edge_superv == 0:
    args.evalf += '_noEdgeSuperv'

os.system('mkdir -p ' + args.evalf)
os.system('mkdir -p ' + args.des_dir)

# setup recorder
tee = Tee(os.path.join(args.des_dir, 'eval.log'), 'w')
print(args)

use_gpu = torch.cuda.is_available()

# define interaction network
model = PropagationNetwork(args, residual=True, use_gpu=use_gpu)
print("model #params: %d" % count_parameters(model))

if args.epoch == 0 and args.iter == 0:
    model_path = os.path.join(args.outf, 'net_best.pth')
else:
    model_path = os.path.join(args.outf, 'net_epoch_%d_iter_%d.pth' % (args.epoch, args.iter))

print("Loading saved ckp from %s" % model_path)
model.load_state_dict(torch.load(model_path))
model.eval()

criterionMSE = nn.MSELoss()
criterionSL1 = nn.SmoothL1Loss()

if use_gpu:
    model.cuda()


def forward_step(frames, model, objs_gt=None):

    n_frames = len(frames)

    if n_frames < args.n_his + 1:
        return [], [], []

    ##### filter frames to predict
    # st_time = time.time()
    ids_predict = []
    objs_first = frames[0][0]
    for i in range(len(objs_first)):
        id = objs_first[i][2]

        id_to_predict = True
        for j in range(1, n_frames):
            objs = frames[j][0]

            contain_id = False
            for k in range(len(objs)):
                if check_same_identifier(objs[k][2], id):
                    contain_id = True
                    break
            if not contain_id:
                id_to_predict = False
                break

        if id_to_predict:
            ids_predict.append(id)

    n_objects = len(ids_predict)

    if n_objects == 0:
        return [], [], []

    # print("Time - filter frame", time.time() - st_time)

    ##### prepare inputs
    # st_time = time.time()
    feats_rec = []
    attrs = []
    for i in range(n_frames):
        objs = frames[i][0]

        feat_cur_frame = []
        for j in range(len(ids_predict)):
            for k in range(len(objs)):
                attr, feat, id = objs[k]
                if check_same_identifier(ids_predict[j], id):
                    feat_cur_frame.append(feat.clone())
                    if i == 0:
                        attrs.append(attr.unsqueeze(0).clone())
                    break

        feats_rec.append(torch.cat(feat_cur_frame, 0))

    attrs = torch.cat(attrs, 0)
    feats = torch.cat(feats_rec.copy(), 1)

    n_relations = n_objects * n_objects
    Ra = torch.FloatTensor(
        np.ones((
            n_relations,
            args.relation_dim * (args.n_his + 1),
            args.bbox_size,
            args.bbox_size)) * -0.5)

    relation_dim = args.relation_dim
    state_dim = args.state_dim
    for i in range(n_objects):
        for j in range(n_objects):
            idx = i * n_objects + j
            Ra[idx, 1::relation_dim] = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # x
            Ra[idx, 2::relation_dim] = feats[i, 2::state_dim] - feats[j, 2::state_dim]  # y

    if args.edge_superv:
        for i in range(n_frames):
            rels = frames[i][1]
            for j in range(len(rels)):
                id_0, id_1 = rels[j][0], rels[j][1]
                x, y = -1, -1

                for k in range(n_objects):
                    if check_same_identifier(id_0, ids_predict[k]):
                        x = k
                    if check_same_identifier(id_1, ids_predict[k]):
                        y = k

                # if x == -1 or y == -1:
                # continue

                idx_rel_xy = x * n_objects + y
                idx_rel_yx = y * n_objects + x
                Ra[idx_rel_xy, i * relation_dim] = 0.5
                Ra[idx_rel_yx, i * relation_dim] = 0.5

    '''
    # change absolute pos to relative pos
    feats[:, state_dim+1::state_dim] = \
            feats[:, state_dim+1::state_dim] - feats[:, 1:-state_dim:state_dim]   # x
    feats[:, state_dim+2::state_dim] = \
            feats[:, state_dim+2::state_dim] - feats[:, 2:-state_dim:state_dim]   # y
    feats[:, 1] = 0
    feats[:, 2] = 0
    '''
    rel = prepare_relations(n_objects)
    rel.append(Ra)

    # print("Time - prepare inputs", time.time() - st_time)

    ##### predict
    # st_time = time.time()
    node_r_idx, node_s_idx, Ra = rel[3], rel[4], rel[5]
    Rr_idx, Rs_idx, value = rel[0], rel[1], rel[2]

    Rr = torch.sparse.FloatTensor(
        Rr_idx, value, torch.Size([node_r_idx.shape[0], value.size(0)]))
    Rs = torch.sparse.FloatTensor(
        Rs_idx, value, torch.Size([node_s_idx.shape[0], value.size(0)]))

    data = [attrs, feats, Rr, Rs, Ra]

    with torch.set_grad_enabled(False):
        for d in range(len(data)):
            if use_gpu:
                data[d] = Variable(data[d].cuda())
            else:
                data[d] = Variable(data[d])

        attr, feats, Rr, Rs, Ra = data
        # print('attr size', attr.size())
        # print('feats size', feats.size())
        # print('Rr size', Rr.size())
        # print('Rs size', Rs.size())
        # print('Ra size', Ra.size())

        # st_time = time.time()
        pred_obj, pred_rel, pred_feat = model(
            attr, feats, Rr, Rs, Ra, node_r_idx, node_s_idx, args.pstep, ret_feat=True)
        # print(time.time() - st_time)

    # print("Time - predict", time.time() - st_time)

    #### transform format
    # st_time = time.time()
    objs_pred = []
    rels_pred = []
    feats_pred = []

    pred_obj = pred_obj.data.cpu()
    pred_rel = pred_rel.data.cpu()
    pred_feat = pred_feat.data.cpu()

    assert pred_obj.shape[0] == pred_feat.shape[0]
    assert pred_feat.shape[1] == args.nf_effect

    '''
    mask = pred_obj[:, 0]
    position = pred_obj[:, 1:3]
    image = pred_obj[:, 3:]
    collision = pred_rel

    print('mask\n', mask)
    print('x\n', position[0])
    print('y\n', position[1])
    print('img\n', image[0])

    time.sleep(10)
    '''

    # print('pred_obj shape', pred_obj.shape)
    # print('pred_rel shape', pred_rel.shape)

    if objs_gt is not None:
        for i in range(n_objects):
            # cnt_id_in_gt = 0
            # id_gt = -1
            for j in range(len(objs_gt)):
                if check_same_identifier(ids_predict[i], objs_gt[j][2]):
                    objs_pred.append(objs_gt[j])
                    feats_pred.append(pred_feat[i])
                    break
                    # id_gt = j
                    # cnt_id_in_gt += 1

            '''
            if cnt_id_in_gt == 0 or cnt_id_in_gt > 1:
                feat = pred_obj[i:i+1]
                feat[0, 1] += feats_rec[-1][i, 1]   # x
                feat[0, 2] += feats_rec[-1][i, 2]   # y
                feat[0, 0, feat[0, 0] >= 0] = 0.5   # mask
                feat[0, 0, feat[0, 0] < 0] = -0.5

                obj = [attrs[i], feat, ids_predict[i]]
                objs_pred.append(obj)
            else:
                objs_pred.append(objs_gt[id_gt])
            '''

    else:
        for i in range(n_objects):
            feat = pred_obj[i:i+1]

            # print(ids_predict[i])
            # print(feat[0, 1])
            # print(feats_rec[-1][i, 1])

            feat[0, 1] += feats_rec[-1][i, 1]   # x

            # print(feat[0, 1])
            # time.sleep(1)
            feat[0, 2] += feats_rec[-1][i, 2]   # y
            feat[0, 0, feat[0, 0] >= 0] = 0.5   # mask
            feat[0, 0, feat[0, 0] < 0] = -0.5

            obj = [attrs[i], feat, ids_predict[i]]
            objs_pred.append(obj)
            feats_pred.append(pred_feat[i])

    '''
    print(objs_pred[0][1][0, 0])
    print(objs_pred[0][1][0, 1])
    print(objs_pred[0][1][0, 2])
    print(objs_pred[0][1][0, 3])
    print(objs_pred[0][1][0, 4])
    '''

    for i in range(n_relations):
        x = i // n_objects
        y = i % n_objects
        if x >= y:
            continue

        idx_0, idx_1 = i, y * n_objects + x
        if pred_rel[idx_0] + pred_rel[idx_1] > 0 and args.edge_superv:
            rels_pred.append([ids_predict[x], ids_predict[y]])

    # print("Time - transform format", time.time() - st_time)

    return objs_pred, rels_pred, feats_pred



test_list = np.arange(args.st_idx, args.ed_idx).tolist()
n_his = args.n_his
frame_offset = args.frame_offset
bbox_size = args.bbox_size
H = args.H
W = args.W

for test_idx in range(len(test_list)):

    print("[%d/%d]" % (test_idx, len(test_list)))

    des_pred = dict()
    des_path = os.path.join(args.des_dir, 'sim_%05d.json' % test_list[test_idx])

    with open(os.path.join(args.label_dir, 'sim_%05d.json' % test_list[test_idx])) as f:
        data = json.load(f)

    gt = data['ground_truth']
    gt_ids = gt['objects']
    gt_collisions = gt['collisions']

    ids_cnter = []

    for i in range(len(data['frames'])):
        objects = data['frames'][i]['objects']
        ids = get_identifiers(objects)

        for x in range(len(ids)):
            found = False
            for y in range(len(ids_cnter)):
                if check_same_identifier(ids_cnter[y][0], ids[x]):
                    ids_cnter[y][1] += 1
                    found = True

            if not found:
                ids_cnter.append([ids[x], 1])

    ids_filter = []
    for i in range(len(ids_cnter)):
        if ids_cnter[i][1] >= args.filter_cnter:
            ids_filter.append(ids_cnter[i][0])

    frames_gt = []
    for i in range(0, len(data['frames']), frame_offset):
        objects = data['frames'][i]['objects']
        frame_filename = data['frames'][i]['frame_filename']
        n_objects = len(objects)

        img = cv2.imread(os.path.join(args.data_dir, frame_filename))
        img = cv2.resize(img, (args.W, args.H), interpolation=cv2.INTER_AREA) / 255.

        frame_objs = []
        frame_rels = []

        ids_cur_frame = []

        for j in range(n_objects):
            id = get_identifier(objects[j])
            if check_contain_id(id, ids_filter) and not check_contain_id(id, ids_cur_frame):
                ids_cur_frame.append(id)

                material = objects[j]['material']
                shape = objects[j]['shape']
                attr = encode_attr(material, shape, bbox_size, args.attr_dim)
                mask_raw = decode(objects[j]['mask'])
                mask = cv2.resize(mask_raw, (W, H), interpolation=cv2.INTER_NEAREST)
                # cv2.imshow('mask', mask * 255)
                # cv2.waitKey(0)

                bbox, pos = convert_mask_to_bbox(mask_raw, H, W, bbox_size)

                pos_mean = torch.FloatTensor(np.array([H / 2., W / 2.]))
                pos_mean = pos_mean.unsqueeze(1).unsqueeze(1)
                pos_std = pos_mean

                pos = normalize(pos, pos_mean, pos_std)
                mask_crop = normalize(crop(mask, bbox, H, W), 0.5, 1).unsqueeze(0)
                img_crop = normalize(crop(img, bbox, H, W), 0.5, 0.5).permute(2, 0, 1)

                s = [attr, torch.cat([mask_crop, pos, img_crop], 0).unsqueeze(0), id]
                frame_objs.append(s)

        if args.edge_superv:
            for j in range(len(gt_collisions)):
                frame_id = gt_collisions[j]['frame']
                if 0 <= frame_id - i < frame_offset:
                    # print(gt_collisions[j]['object'])
                    # print(gt_ids)
                    id_0 = gt_collisions[j]['object'][0]
                    id_1 = gt_collisions[j]['object'][1]
                    for k in range(len(gt_ids)):
                        if id_0 == gt_ids[k]['id']:
                            id_x = get_identifier(gt_ids[k])
                        if id_1 == gt_ids[k]['id']:
                            id_y = get_identifier(gt_ids[k])

                    # id_0 = get_identifier(gt_ids[gt_collisions[j]['object'][0]])
                    # id_1 = get_identifier(gt_ids[gt_collisions[j]['object'][1]])
                    frame_rels.append([id_x, id_y])

        frames_gt.append([frame_objs, frame_rels, None])

    if args.video:
        path = os.path.join(args.evalf, '%d_gt' % test_list[test_idx])
        make_video(path, frames_gt, H, W, bbox_size, args.back_ground, args.store_img)

    if args.use_attr:
        des_pred['objects'] = []
        for i in range(len(ids_filter)):
            obj = dict()
            obj['color'] = ids_filter[i][0]
            obj['material'] = ids_filter[i][1]
            obj['shape'] = ids_filter[i][2]
            obj['id'] = i
            des_pred['objects'].append(obj)


    ##### prediction from the learned model

    des_pred['predictions'] = []

    if args.use_attr == 1:
        what_if_ed_idx = len(ids_filter)
    else:
        what_if_ed_idx = 0

    for what_if in range(-1, what_if_ed_idx):
        frames_pred = []

        what_if_shown_up = False
        for i in range(0, len(data['frames']), frame_offset):

            idx = i // frame_offset

            frame_objs = []
            frame_rels = []
            frame_feats = []

            objs_gt = frames_gt[idx][0]
            rels_gt = frames_gt[idx][1]

            for j in range(len(objs_gt)):
                id = objs_gt[j][2]

                valid_idx = False
                for k in range(len(ids_filter)):
                    if check_same_identifier(id, ids_filter[k]):

                        if k == what_if:
                            what_if_shown_up = True
                            continue

                        valid_idx = True
                        break

                if not valid_idx:
                    continue

                already_considered = True
                if idx < n_his + 1:
                    already_considered = False
                else:
                    for k in range(n_his + 1):
                        objs_considering = frames_pred[idx - k - 1][0]

                        contain_id = False
                        for t in range(len(objs_considering)):
                            if check_same_identifier(id, objs_considering[t][2]):
                                contain_id = True
                                break
                        if not contain_id:
                            already_considered = False
                            break

                if not already_considered:
                    frame_objs.append(objs_gt[j])

            # st_time = time.time()
            if what_if == -1 or not what_if_shown_up:
                objs_gt = frames_gt[idx][0]
            else:
                objs_gt = None

            objs_pred, rels_pred, feats_pred = forward_step(frames_pred[idx-n_his-1:idx], model, objs_gt)
            # print(time.time() - st_time)

            frame_objs += objs_pred
            frame_rels += rels_pred
            frame_feats += feats_pred

            frames_pred.append([frame_objs, frame_rels, frame_feats])

        # rollout for extra 12 frames
        # if what_if == -1:
        st_idx = len(frames_pred)
        for idx in range(st_idx, st_idx + 12):
            objs_pred, rels_pred, feats_pred = forward_step(frames_pred[idx-n_his-1:idx], model)
            frames_pred.append([objs_pred, rels_pred, feats_pred])

        traj_predict = dict()

        if what_if == -1:
            video_name = '%d_pred' % test_list[test_idx]
            traj_predict['what_if'] = -1
        else:
            video_name = '%d_pred_%d' % (test_list[test_idx], what_if)
            traj_predict['what_if'] = what_if

        traj_predict['trajectory'] = []
        traj_predict['collisions'] = []

        for i in range(len(frames_pred)):
            objs, rels, feats = frames_pred[i]
            frame = dict()
            frame['frame_index'] = i * args.frame_offset

            if args.use_attr:
                frame['objects'] = []

                for j in range(len(objs)):
                    obj = objs[j][1][0]
                    id = objs[j][2]
                    x = obj[1, 0, 0] * H/2. + H/2.
                    y = obj[2, 0, 0] * W/2. + W/2.
                    obj_pred = dict()
                    obj_pred['x'] = float(x)
                    obj_pred['y'] = float(y)
                    obj_pred['color'] = id[0]
                    obj_pred['material'] = id[1]
                    obj_pred['shape'] = id[2]

                    frame['objects'].append(obj_pred)
            else:
                frame['feats'] = []
                for j in range(len(feats)):
                    frame['feats'].append(feats[j].numpy().tolist())

            traj_predict['trajectory'].append(frame)

            if args.use_attr:
                for j in range(len(rels)):
                    id_0, id_1 = rels[j][0], rels[j][1]
                    collide = dict()
                    collide['frame'] = i * args.frame_offset
                    collide['objects'] = []

                    obj_collide = dict()
                    obj_collide['color'] = id_0[0]
                    obj_collide['material'] = id_0[1]
                    obj_collide['shape'] = id_0[2]
                    collide['objects'].append(obj_collide)

                    obj_collide = dict()
                    obj_collide['color'] = id_1[0]
                    obj_collide['material'] = id_1[1]
                    obj_collide['shape'] = id_1[2]
                    collide['objects'].append(obj_collide)

                    traj_predict['collisions'].append(collide)

        des_pred['predictions'].append(traj_predict)

        if args.video:
            path = os.path.join(args.evalf, video_name)
            make_video(path, frames_pred, H, W, bbox_size, args.back_ground, args.store_img)

    with open(des_path, 'w') as f:
        json.dump(des_pred, f)

