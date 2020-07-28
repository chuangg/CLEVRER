import os
import torch
import time
import random
import numpy as np
import gzip
import pickle
import h5py
import json

import pycocotools._mask as _mask
import cv2
from skimage import io, transform
from PIL import Image

import multiprocessing as mp
import scipy.spatial as spatial
from sklearn.cluster import MiniBatchKMeans

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from utils import prepare_relations, convert_mask_to_bbox, crop, encode_attr
from utils import normalize, check_attr, get_identifier, get_identifiers
from utils import check_same_identifier, check_same_identifiers, check_contain_id
from utils import get_masks, check_valid_masks, check_duplicate_identifier
from utils import rand_float, init_stat, combine_stat, load_data, store_data
from utils import decode, make_video


def collate_fn(data):
    return data[0]


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


class PhysicsCLEVRDataset(Dataset):

    def __init__(self, args, phase):
        self.args = args
        self.phase = phase
        self.loader = default_loader
        self.data_dir = args.data_dir
        self.label_dir = args.label_dir
        self.valid_idx_lst = 'valid_idx_' + self.phase + '.txt'
        self.H = 100
        self.W = 150
        self.bbox_size = 24

        ratio = self.args.train_valid_ratio
        n_train = round(self.args.n_rollout * ratio)
        if phase == 'train':
            self.st_idx = 0
            self.n_rollout = n_train
        elif phase == 'valid':
            self.st_idx = n_train
            self.n_rollout = self.args.n_rollout - n_train
        else:
            raise AssertionError("Unknown phase")

        if self.args.gen_valid_idx:
            self.gen_valid_idx()
        else:
            self.read_valid_idx()

    def read_valid_idx(self):
        # if self.phase == 'train':
        # return
        print("Reading valid idx ...")
        self.n_valid_idx = 0
        self.valid_idx = []
        self.metadata = []
        fin = open(self.valid_idx_lst, 'r').readlines()

        self.n_valid_idx = len(fin)
        for i in range(self.n_valid_idx):
            a = int(fin[i].strip().split(' ')[0])
            b = int(fin[i].strip().split(' ')[1])
            self.valid_idx.append((a, b))

        for i in range(self.st_idx, self.st_idx + self.n_rollout):
            if i % 500 == 0:
                print("Reading valid idx %d/%d" % (i, self.st_idx + self.n_rollout))

            with open(os.path.join(self.label_dir, 'sim_%05d.json' % i)) as f:
                data = json.load(f)
                self.metadata.append(data)

    def gen_valid_idx(self):
        print("Preprocessing valid idx ...")
        self.n_valid_idx = 0
        self.valid_idx = []
        self.metadata = []
        fout = open(self.valid_idx_lst, 'w')

        n_his = self.args.n_his
        frame_offset = self.args.frame_offset

        for i in range(self.st_idx, self.st_idx + self.n_rollout):
            if i % 500 == 0:
                print("Preprocessing valid idx %d/%d" % (i, self.st_idx + self.n_rollout))

            with open(os.path.join(self.label_dir, 'sim_%05d.json' % i)) as f:
                data = json.load(f)
                self.metadata.append(data)

            gt = data['ground_truth']
            gt_ids = gt['objects']
            gt_collisions = gt['collisions']

            for j in range(
                n_his * frame_offset,
                len(data['frames']) - frame_offset):

                objects = data['frames'][j]['objects']
                n_object_cur = len(objects)
                identifiers_cur = get_identifiers(objects)
                valid = True

                # check whether the current frame is valid:
                if check_duplicate_identifier(objects):
                    valid = False

                '''
                masks = get_masks(objects)
                if not check_valid_masks(masks):
                    valid = False
                '''

                # check whether history window is valid
                for k in range(n_his):
                    idx = j - (k + 1) * frame_offset
                    objects = data['frames'][idx]['objects']
                    n_object = len(objects)
                    identifiers = get_identifiers(objects)
                    # masks = get_masks(objects)

                    if (not valid) or n_object != n_object_cur:
                        valid = False
                        break
                    if not check_same_identifiers(identifiers, identifiers_cur):
                        valid = False
                        break
                    if check_duplicate_identifier(objects):
                        valid = False
                        break

                    '''
                    if not check_valid_masks(masks):
                        valid = False
                        break
                    '''

                # check whether the target is valid
                idx = j + frame_offset
                objects_nxt = data['frames'][idx]['objects']
                n_object_nxt = len(objects_nxt)
                identifiers_nxt = get_identifiers(objects_nxt)
                if n_object_nxt != n_object_cur:
                    valid = False
                elif not check_same_identifiers(identifiers_nxt, identifiers_cur):
                    valid = False
                elif check_duplicate_identifier(objects_nxt):
                    valid = False

                # check if detected the right objects for collision
                for k in range(len(gt_collisions)):
                    if 0 <= gt_collisions[k]['frame'] - j < frame_offset:
                        gt_obj = gt_collisions[k]['object']

                        id_0 = gt_obj[0]
                        id_1 = gt_obj[1]
                        for t in range(len(gt_ids)):
                            if id_0 == gt_ids[t]['id']:
                                id_x = get_identifier(gt_ids[t])
                            if id_1 == gt_ids[t]['id']:
                                id_y = get_identifier(gt_ids[t])

                        # id_0 = get_identifier(gt_ids[gt_obj[0]])
                        # id_1 = get_identifier(gt_ids[gt_obj[1]])
                        if not check_contain_id(id_x, identifiers_cur):
                            valid = False
                        if not check_contain_id(id_y, identifiers_cur):
                            valid = False

                '''
                masks_nxt = get_masks(objects_nxt)
                if not check_valid_masks(masks_nxt):
                    valid = False
                '''

                if valid:
                    self.valid_idx.append((i - self.st_idx, j))
                    fout.write('%d %d\n' % (i - self.st_idx, j))
                    self.n_valid_idx += 1

        fout.close()

    '''
    def read_valid_idx(self):
        fin = open(self.valid_idx_lst, 'r').readlines()
        self.n_valid_idx = len(fin)
        self.valid_idx = []
        for i in range(len(fin)):
            idx = [int(x) for x in fin[i].strip().split(' ')]
            self.valid_idx.append((idx[0], idx[1]))
    '''

    def __len__(self):
        return self.n_valid_idx

    def __getitem__(self, idx):
        n_his = self.args.n_his
        frame_offset = self.args.frame_offset
        idx_video, idx_frame = self.valid_idx[idx][0], self.valid_idx[idx][1]

        objs = []
        attrs = []
        for i in range(
            idx_frame - n_his * frame_offset,
            idx_frame + frame_offset + 1, frame_offset):

            frame = self.metadata[idx_video]['frames'][i]
            frame_filename = frame['frame_filename']
            objects = frame['objects']
            n_objects = len(objects)

            img = self.loader(os.path.join(self.data_dir, frame_filename))
            img = np.array(img)[:, :, ::-1].copy()
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA).astype(np.float) / 255.

            ### prepare object inputs
            object_inputs = []
            for j in range(n_objects):
                material = objects[j]['material']
                shape = objects[j]['shape']

                if i == idx_frame - n_his * frame_offset:
                    attrs.append(encode_attr(
                        material, shape, self.bbox_size, self.args.attr_dim))

                mask_raw = decode(objects[j]['mask'])
                mask = cv2.resize(mask_raw, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
                # cv2.imshow("mask", mask * 255)
                # cv2.waitKey(0)

                bbox, pos = convert_mask_to_bbox(mask_raw, self.H, self.W, self.bbox_size)
                # print(pos)

                pos_mean = torch.FloatTensor(np.array([self.H / 2., self.W / 2.]))
                pos_mean = pos_mean.unsqueeze(1).unsqueeze(1)
                pos_std = pos_mean

                pos = normalize(pos, pos_mean, pos_std)
                # print(pos)
                mask_crop = normalize(crop(mask, bbox, self.H, self.W), 0.5, 1).unsqueeze(0)
                img_crop = normalize(crop(img, bbox, self.H, self.W), 0.5, 0.5).permute(2, 0, 1)

                identifier = get_identifier(objects[j])

                # print(torch.max(pos), torch.min(pos))
                # print('mask_crop size', mask_crop.size())
                # print('pos size', pos.size())
                # print('img_crop size', img_crop.size())

                s = torch.cat([mask_crop, pos, img_crop], 0).unsqueeze(0), identifier
                object_inputs.append(s)

            objs.append(object_inputs)

        attr = torch.cat(attrs, 0).view(
            n_objects, self.args.attr_dim, self.bbox_size, self.bbox_size)

        feats = []
        for x in range(n_objects):
            feats.append(objs[0][x][0])

        for i in range(1, len(objs)):
            for x in range(n_objects):
                for y in range(n_objects):
                    id_x = objs[0][x][1]
                    id_y = objs[i][y][1]
                    if check_same_identifier(id_x, id_y):
                        feats[x] = torch.cat([feats[x], objs[i][y][0]], 1)

        # for i in range(1, self.args.state_dim * (n_his + 2), self.args.state_dim):
        # print(feats[0][0, i, 0, 0], feats[0][0, i, 1, 1])
        # print()

        try:
            feats = torch.cat(feats, 0)
        except:
            print(idx_video, idx_frame)
        # print("feats shape", feats.size())

        ### prepare relation attributes
        n_relations = n_objects * n_objects
        Ra = torch.FloatTensor(
            np.ones((
                n_relations,
                self.args.relation_dim * (self.args.n_his + 2),
                self.bbox_size,
                self.bbox_size)) * -0.5)

        # change to relative position
        relation_dim = self.args.relation_dim
        state_dim = self.args.state_dim
        for i in range(n_objects):
            for j in range(n_objects):
                idx = i * n_objects + j
                Ra[idx, 1::relation_dim] = feats[i, 1::state_dim] - feats[j, 1::state_dim]  # x
                Ra[idx, 2::relation_dim] = feats[i, 2::state_dim] - feats[j, 2::state_dim]  # y

        # add collision attr
        gt = self.metadata[idx_video]['ground_truth']
        gt_ids = gt['objects']
        gt_collisions = gt['collisions']

        label_rel = torch.FloatTensor(np.ones((n_objects * n_objects, 1)) * -0.5)

        if self.args.edge_superv:
            for i in range(
                idx_frame - n_his * frame_offset,
                idx_frame + frame_offset + 1, frame_offset):

                for j in range(len(gt_collisions)):
                    frame_id = gt_collisions[j]['frame']
                    if 0 <= frame_id - i < self.args.frame_offset:
                        id_0 = gt_collisions[j]['object'][0]
                        id_1 = gt_collisions[j]['object'][1]
                        for k in range(len(gt_ids)):
                            if id_0 == gt_ids[k]['id']:
                                id_x = get_identifier(gt_ids[k])
                            if id_1 == gt_ids[k]['id']:
                                id_y = get_identifier(gt_ids[k])

                        # id_0 = get_identifier(gt_ids[gt_collisions[j]['object'][0]])
                        # id_1 = get_identifier(gt_ids[gt_collisions[j]['object'][1]])

                        for k in range(n_objects):
                            if check_same_identifier(objs[0][k][1], id_x):
                                x = k
                            if check_same_identifier(objs[0][k][1], id_y):
                                y = k

                        idx_rel_xy = x * n_objects + y
                        idx_rel_yx = y * n_objects + x

                        # print(x, y, n_objects)

                        idx = i - (idx_frame - n_his * frame_offset)
                        idx /= frame_offset
                        Ra[idx_rel_xy, int(idx) * relation_dim] = 0.5
                        Ra[idx_rel_yx, int(idx) * relation_dim] = 0.5

                        if i == idx_frame + frame_offset:
                            label_rel[idx_rel_xy] = 1
                            label_rel[idx_rel_yx] = 1

        '''
        print(feats[0, -state_dim])
        print(feats[0, -state_dim+1])
        print(feats[0, -state_dim+2])
        print(feats[0, -state_dim+3])
        print(feats[0, -state_dim+4])
        '''

        '''
        ### change absolute pos to relative pos
        feats[:, state_dim+1::state_dim] = \
                feats[:, state_dim+1::state_dim] - feats[:, 1:-state_dim:state_dim]   # x
        feats[:, state_dim+2::state_dim] = \
                feats[:, state_dim+2::state_dim] - feats[:, 2:-state_dim:state_dim]   # y
        feats[:, 1] = 0
        feats[:, 2] = 0
        '''

        x = feats[:, :-state_dim]
        label_obj = feats[:, -state_dim:]
        label_obj[:, 1] -= feats[:, -2*state_dim+1]
        label_obj[:, 2] -= feats[:, -2*state_dim+2]
        rel = prepare_relations(n_objects)
        rel.append(Ra[:, :-relation_dim])

        '''
        print(rel[-1][0, 0])
        print(rel[-1][0, 1])
        print(rel[-1][0, 2])
        print(rel[-1][2, 3])
        print(rel[-1][2, 4])
        print(rel[-1][2, 5])
        '''

        # print("attr shape", attr.size())
        # print("x shape", x.size())
        # print("label_obj shape", label_obj.size())
        # print("label_rel shape", label_rel.size())

        '''
        for i in range(n_objects):
            print(objs[0][i][1])
            print(label_obj[i, 1])

        time.sleep(10)
        '''

        return attr, x, rel, label_obj, label_rel

