import sys
import argparse
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle
import h5py
import os
import pycocotools._mask as _mask

import torch
from torch.autograd import Variable


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()

    def close(self):
        self.__del__()


def decode(rleObjs):
    if type(rleObjs) == list:
        return _mask.decode(rleObjs)
    else:
        return _mask.decode([rleObjs])[:,:,0]


def prepare_relations(n):
    node_r_idx = np.arange(n)
    node_s_idx = np.arange(n)

    rel = np.zeros((n**2, 2))
    rel[:, 0] = np.repeat(np.arange(n), n)
    rel[:, 1] = np.tile(np.arange(n), n)

    # print(rel)

    n_rel = rel.shape[0]
    Rr_idx = torch.LongTensor([rel[:, 0], np.arange(n_rel)])
    Rs_idx = torch.LongTensor([rel[:, 1], np.arange(n_rel)])
    value = torch.FloatTensor([1] * n_rel)

    rel = [Rr_idx, Rs_idx, value, node_r_idx, node_s_idx]

    return rel


def convert_mask_to_bbox(mask, H, W, bbox_size):
    h, w = mask.shape[0], mask.shape[1]
    x = np.repeat(np.arange(h), w).reshape(h, w)
    y = np.tile(np.arange(w), h).reshape(h, w)
    x = np.sum(mask * x) / np.sum(mask) * (float(H) / h)
    y = np.sum(mask * y) / np.sum(mask) * (float(W) / w)
    bbox = int(x - bbox_size / 2), int(y - bbox_size / 2), bbox_size, bbox_size
    ret = np.ones((2, bbox_size, bbox_size))
    ret[0, :, :] *= x
    ret[1, :, :] *= y
    return bbox, torch.FloatTensor(ret)


def crop(src, bbox, H, W):
    x, y, h, w = bbox
    # print(bbox)
    shape = list(src.shape)
    shape[0], shape[1] = h, w
    ret = np.zeros(shape)

    x_ = max(-x, 0)
    y_ = max(-y, 0)
    x = max(x, 0)
    y = max(y, 0)
    h_ = min(h - x_, H - x)
    w_ = min(w - y_, W - y)

    # print(x, y, x_, y_, h_, w_)

    ret[x_:x_+h_, y_:y_+w_] = src[x:x+h_, y:y+w_]

    # print(src[x:x+h, y:y+w])
    # cv2.imshow('img', np.array(ret * 255, dtype=np.uint8))
    # cv2.waitKey(0)
    return torch.FloatTensor(ret)


def encode_attr(material, shape, bbox_size, attr_dim):
    attr = np.zeros(attr_dim)
    if material == 'rubber':
        attr[0] = 1
    elif material == 'metal':
        attr[1] = 1
    else:
        raise AssertionError("unknown material: " + material)

    if shape == 'cube':
        attr[2] = 1
    elif shape == 'cylinder':
        attr[3] = 1
    elif shape == 'sphere':
        attr[4] = 1
    else:
        raise AssertionError("unknown shape: " + shape)

    ret = np.ones((bbox_size, bbox_size, attr_dim)) * attr
    ret = np.swapaxes(ret, 0, 2)

    return torch.FloatTensor(ret)


def normalize(x, mean, std):
    return (x - mean) / std


def check_attr(id):
    color, material, shape = id
    if material == 'metal' or material == 'rubber':
        pass
    else:
        raise AssertionError("unknown material: " + material)

    if shape == 'cube' or shape == 'sphere' or shape == 'cylinder':
        pass
    else:
        raise AssertionError("unknown shape: " + shape)


def get_identifier(obj):
    color = obj['color']
    material = obj['material']
    shape = obj['shape']
    return color, material, shape


def get_identifiers(objects):
    ids = []
    for i in range(len(objects)):
        id = get_identifier(objects[i])
        check_attr(id)
        ids.append(id)
    return ids


def check_same_identifier(id_0, id_1):
    len_id = len(id_0)
    for i in range(len_id):
        if id_0[i] != id_1[i]:
            return False
    return True


def check_contain_id(id, ids):
    for i in range(len(ids)):
        if check_same_identifier(id, ids[i]):
            return True
    return False


def check_same_identifiers(ids_0, ids_1):
    len_ids = len(ids_0)
    for i in range(len_ids):
        find_same_id = False
        for j in range(len_ids):
            if check_same_identifier(ids_0[i], ids_1[j]):
                find_same_id = True
                break
        if not find_same_id:
            return False

    return True


def get_masks(objects):
    masks = []
    for i in range(len(objects)):
        mask = decode(objects[i]['mask'])
        masks.append(mask)
    return masks


def check_valid_masks(masks):
    for i in range(len(masks)):
        if np.sum(masks[i]) == 0:
            return False
    return True


def check_duplicate_identifier(objects):
    n_objects = len(objects)
    for xx in range(n_objects):
        id_xx = get_identifier(objects[xx])
        for yy in range(xx + 1, n_objects):
            id_yy = get_identifier(objects[yy])
            if check_same_identifier(id_xx, id_yy):
                return True
    return False


def store_data(data_names, data, path):
    hf = h5py.File(path, 'w')
    for i in range(len(data_names)):
        hf.create_dataset(data_names[i], data=data[i])
    hf.close()


def load_data(data_names, path):
    hf = h5py.File(path, 'r')
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()
    return data


def combine_stat(stat_0, stat_1):
    mean_0, std_0, n_0 = stat_0[:, 0], stat_0[:, 1], stat_0[:, 2]
    mean_1, std_1, n_1 = stat_1[:, 0], stat_1[:, 1], stat_1[:, 2]

    mean = (mean_0 * n_0 + mean_1 * n_1) / (n_0 + n_1)
    std = np.sqrt((std_0**2 * n_0 + std_1**2 * n_1 + \
                   (mean_0 - mean)**2 * n_0 + (mean_1 - mean)**2 * n_1) / (n_0 + n_1))
    n = n_0 + n_1

    return np.stack([mean, std, n], axis=-1)


def init_stat(dim):
    # mean, std, count
    return np.zeros((dim, 3))


def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


def norm(x):
    return np.sqrt(np.sum(x**2))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_variable(tensor, use_gpu, requires_grad=False):
    if use_gpu:
        return Variable(torch.FloatTensor(tensor).cuda(),
                        requires_grad=requires_grad)
    else:
        return Variable(torch.FloatTensor(tensor),
                        requires_grad=requires_grad)


def sort_by_x(obj):
    return obj[1][0, 1, 0, 0]


def merge_img_patch(img_0, img_1):
    # cv2.imshow('img_0', img_0.astype(np.uint8))
    # cv2.imshow('img_1', img_1.astype(np.uint8))

    ret = img_0.copy()
    idx = img_1[:, :, 0] > 0
    idx = np.logical_or(idx, img_1[:, :, 1] > 0)
    idx = np.logical_or(idx, img_1[:, :, 2] > 0)
    ret[idx] = img_1[idx]

    # cv2.imshow('ret', ret.astype(np.uint8))
    # cv2.waitKey(0)

    return ret


def make_video(filename, frames, H, W, bbox_size, back_ground=None, store_img=False):

    n_frame = len(frames)

    # print('states', states.shape)
    # print('actions', actions.shape)
    # print(filename)

    # print(actions[:, 0, :])
    # print(states[:20, 0, :])

    videoname = filename + '.avi'
    os.system('mkdir -p ' + filename)

    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 16}

    colors = [np.array([255,160,122]),
              np.array([224,255,255]),
              np.array([216,191,216]),
              np.array([255,255,224]),
              np.array([245,245,245]),
              np.array([144,238,144])]

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(videoname, fourcc, 3, (W, H))

    if back_ground is not None:
        bg = cv2.imread(back_ground)
        bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_AREA)

    for i in range(n_frame):
        objs, rels, feats = frames[i]
        n_objs = len(objs)

        if back_ground is not None:
            frame = bg.copy()
        else:
            frame = np.ones((H, W, 3), dtype=np.uint8) * 255

        objs = objs.copy()

        # obj: attr, [mask_crop, pos, img_crop], id
        objs.sort(key=sort_by_x)

        n_object = len(objs)
        for j in range(n_object):
            obj = objs[j][1][0]

            mask = obj[:1].permute(1, 2, 0).data.numpy()
            img = obj[3:].permute(1, 2, 0).data.numpy()
            mask = np.clip((mask + 0.5) * 255, 0, 255)
            img = np.clip((img * 0.5 + 0.5) * mask, 0, 255)
            # img *= mask

            n_rels = len(rels)
            collide = False
            for k in range(n_rels):
                id_0, id_1 = rels[k][0], rels[k][1]
                if check_same_identifier(id_0, objs[j][2]) or check_same_identifier(id_1, objs[j][2]):
                    collide = True

            if collide:
                _, cont, _ = cv2.findContours(
                    mask.astype(np.uint8)[:, :, 0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cont, -1, (0, 255, 0), 1)

                '''
                print(i, j)
                cv2.imshow('mask', mask.astype(np.uint8))
                cv2.imshow('img', img.astype(np.uint8))
                cv2.waitKey(0)
                '''

            if np.isnan(obj[1, 0, 0]) or np.isnan(obj[2, 0, 0]):
                # check if the position is NaN
                continue
            if np.isinf(obj[1, 0, 0]) or np.isinf(obj[2, 0, 0]):
                # check if the position is inf
                continue

            x = int(obj[1, 0, 0] * H/2. + H/2. - bbox_size/2)
            y = int(obj[2, 0, 0] * W/2. + W/2. - bbox_size/2)

            # print(x, y, H, W)
            h, w = int(bbox_size), int(bbox_size)
            x_ = max(-x, 0)
            y_ = max(-y, 0)
            x = max(x, 0)
            y = max(y, 0)
            h_ = min(h - x_, H - x)
            w_ = min(w - y_, W - y)

            # print(x, y, x_, y_, h_, w_)

            if x + h_ < 0 or x >= H or y + w_ < 0 or y >= W:
                continue

            frame[x:x+h_, y:y+w_] = merge_img_patch(
                frame[x:x+h_, y:y+w_], img[x_:x_+h_, y_:y_+w_])

        if store_img:
            cv2.imwrite(os.path.join(filename, 'img_%d.png' % i), frame.astype(np.uint8))
        # cv2.imshow('img', frame.astype(np.uint8))
        # cv2.waitKey(0)

        out.write(frame)

    out.release()


class ChamferLoss(torch.nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def chamfer_distance(self, x, y):
        # x: [N, D]
        # y: [M, D]
        x = x.repeat(y.size(0), 1, 1)   # x: [M, N, D]
        x = x.transpose(0, 1)           # x: [N, M, D]
        y = y.repeat(x.size(0), 1, 1)   # y: [N, M, D]
        dis = torch.norm(torch.add(x, -y), 2, dim=2)    # dis: [N, M]
        dis_xy = torch.mean(torch.min(dis, dim=1)[0])   # dis_xy: mean over N
        dis_yx = torch.mean(torch.min(dis, dim=0)[0])   # dis_yx: mean over M

        return dis_xy + dis_yx

    def __call__(self, pred, label):
        return self.chamfer_distance(pred, label)
