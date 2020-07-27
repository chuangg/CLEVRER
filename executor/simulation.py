import os
import json
import numpy as np


MOVING_V_TH = 0.1  # threshold above which an object is moving
DIR_ANGLE_TH = 20  # threshold for allowed angle deviation wrt each directions
FRAME_DIFF = 5

DV_TH = 0.1
DIST_TH = 20


class Simulation():

    def __init__(self, ann_path, n_vis_frames=128, use_event_ann=True):
        with open(ann_path) as f:
            ann = json.load(f)
            self.objs = ann['objects']
            self.preds = ann['predictions']
        self.num_objs = len(self.objs)
        self.frame_diff = FRAME_DIFF
        self.n_vis_frames = n_vis_frames
        self.moving_v_th = MOVING_V_TH
        if use_event_ann:
            self._init_sim()
        else:
            self._init_sim_no_event()

    def get_visible_objs(self):
        return [o['id'] for o in self.objs]

    def get_static_attrs(self, obj_idx):
        for o in self.objs:
            if o['id'] == obj_idx:
                attrs = {
                    'color': o['color'],
                    'material': o['material'],
                    'shape': o['shape'],
                }
                return attrs
        raise ValueError('Invalid object index')

    def is_visible(self, obj_idx, frame_idx=None, ann_idx=None, what_if=-1):
        if frame_idx is not None or ann_idx is not None:
            frame_ann = self._get_frame_ann(frame_idx, ann_idx, what_if)
            for o in frame_ann['objects']:
                oid = self._get_obj_idx(o)
                if oid == obj_idx:
                    return True
            return False
        else:
            for i, t in enumerate(self.preds[0]['trajectory']):
                if t['frame_index'] < self.n_vis_frames and \
                   self.is_visible(obj_idx, ann_idx=i, what_if=what_if):
                    return True
            return False

    def is_moving(self, obj_idx, frame_idx=None, ann_idx=None):
        if frame_idx is not None or ann_idx is not None:
            frame_ann = self._get_frame_ann(frame_idx, ann_idx)
            for o in frame_ann['objects']:
                oid = self._get_obj_idx(o)
                if oid == obj_idx:
                    speed = np.linalg.norm([o['vx'], o['vy']])
                    return speed > self.moving_v_th
            print(frame_idx, ann_idx)
            raise ValueError('Invalid object index')
        else:
            for i, t in enumerate(self.preds[0]['trajectory']):
                if self.is_visible(obj_idx, ann_idx=i) and \
                   self.is_moving(obj_idx, ann_idx=i):
                    return True
            return False

    def is_moving_left(self, obj_idx, frame_idx=None, ann_idx=None,
                       angle_half_range=DIR_ANGLE_TH):
        frame_ann = self._get_frame_ann(frame_idx, ann_idx)
        for o in frame_ann['objects']:
            oid = self._get_obj_idx(o)
            if oid == obj_idx:
                theta = np.arctan(o['vy'] / o['vx']) * 180 / np.pi
                if o['vx'] < 0:
                    theta += 180
                return theta > 270 - angle_half_range or \
                       theta < -90 + angle_half_range
        raise ValueError('Invalid object index')

    def is_moving_right(self, obj_idx, frame_idx=None, ann_idx=None,
                       angle_half_range=DIR_ANGLE_TH):
        frame_ann = self._get_frame_ann(frame_idx, ann_idx)
        for o in frame_ann['objects']:
            oid = self._get_obj_idx(o)
            if oid == obj_idx:
                theta = np.arctan(o['vy'] / o['vx']) * 180 / np.pi
                if o['vx'] < 0:
                    theta += 180
                return theta > 90 - angle_half_range and \
                       theta < 90 + angle_half_range
        raise ValueError('Invalid object index')

    def is_moving_up(self, obj_idx, frame_idx=None, ann_idx=None,
                       angle_half_range=DIR_ANGLE_TH):
        frame_ann = self._get_frame_ann(frame_idx, ann_idx)
        for o in frame_ann['objects']:
            oid = self._get_obj_idx(o)
            if oid == obj_idx:
                theta = np.arctan(o['vy'] / o['vx']) * 180 / np.pi
                if o['vx'] < 0:
                    theta += 180
                return theta > 180 - angle_half_range and \
                       theta < 180 + angle_half_range
        raise ValueError('Invalid object index')

    def is_moving_down(self, obj_idx, frame_idx=None, ann_idx=None,
                       angle_half_range=DIR_ANGLE_TH):
        frame_ann = self._get_frame_ann(frame_idx, ann_idx)
        for o in frame_ann['objects']:
            oid = self._get_obj_idx(o)
            if oid == obj_idx:
                theta = np.arctan(o['vy'] / o['vx']) * 180 / np.pi
                if o['vx'] < 0:
                    theta += 180
                return theta < 0 + angle_half_range and \
                       theta > 0 - angle_half_range
        raise ValueError('Invalid object index')

    def _init_sim_no_event(self):
        self.in_out = []
        self.collisions = []
        self.cf_events = {}
        for k, p in enumerate(self.preds):
            for i, t in enumerate(p['trajectory']):
                for o in t['objects']:
                    o['id'] = self._get_obj_idx(o)
                    vxs, vys = [], []
                    if i != 0 and not self.is_visible(o['id'], ann_idx=i-1, what_if=p['what_if']):
                        if k == 0:
                            self.in_out.append({'frame': t['frame_index'], 'type': 'in', 'object': [o['id']]})
                    elif i != 0:
                        x_prev, y_prev = self._get_obj_location(o['id'], ann_idx=i-1, what_if=p['what_if'])
                        vxs.append((o['x'] - x_prev) / self.frame_diff)
                        vys.append((o['y'] - y_prev) / self.frame_diff)
                    if i != len(p['trajectory']) - 1 and not self.is_visible(o['id'], ann_idx=i+1, what_if=p['what_if']):
                        if k == 0:
                            self.in_out.append({'frame': p['trajectory'][i+1]['frame_index'], 'type': 'out', 'object': [o['id']]})
                    elif i != len(p['trajectory']) - 1:
                        x_next, y_next = self._get_obj_location(o['id'], ann_idx=i+1, what_if=p['what_if'])
                        vxs.append((x_next - o['x']) / self.frame_diff)
                        vys.append((y_next - o['y']) / self.frame_diff)
                    if len(vxs) != 0:
                        o['vx'] = np.average(vxs)
                        o['vy'] = np.average(vys)
                    else:
                        o['vx'], o['vy'] = 0, 0

            if p['what_if'] == -1:
                self.collisions = self._get_col_proposals()
            else:
                self.cf_events[p['what_if']] = self._get_col_proposals(p['what_if'])


    def _init_sim(self):
        self.in_out = []
        self.collisions = []
        p = self.preds[0]
        for i, t in enumerate(p['trajectory']):
            for o in t['objects']:
                o['id'] = self._get_obj_idx(o)
                vxs, vys = [], []
                if i != 0 and not self.is_visible(o['id'], ann_idx=i-1):
                    self.in_out.append({'frame': t['frame_index'], 'type': 'in', 'object': [o['id']]})
                elif i != 0:
                    x_prev, y_prev = self._get_obj_location(o['id'], ann_idx=i-1)
                    vxs.append((o['x'] - x_prev) / self.frame_diff)
                    vys.append((o['y'] - y_prev) / self.frame_diff)
                if i != len(p['trajectory']) - 1 and not self.is_visible(o['id'], ann_idx=i+1):
                    self.in_out.append({'frame': p['trajectory'][i+1]['frame_index'], 'type': 'out', 'object': [o['id']]})
                elif i != len(p['trajectory']) - 1:
                    x_next, y_next = self._get_obj_location(o['id'], ann_idx=i+1)
                    vxs.append((x_next - o['x']) / self.frame_diff)
                    vys.append((y_next - o['y']) / self.frame_diff)
                if len(vxs) != 0:
                    o['vx'] = np.average(vxs)
                    o['vy'] = np.average(vys)
                else:
                    o['vx'], o['vy'] = 0, 0
        for c in p['collisions']:
            obj1_idx = self._get_obj_idx(c['objects'][0])
            obj2_idx = self._get_obj_idx(c['objects'][1])
            self.collisions.append({
                    'type': 'collision',
                    'object': [obj1_idx, obj2_idx],
                    'frame': c['frame'],
                })

        self.cf_events = {}
        for j in range(1, len(self.preds)):
            assert self.preds[j]['what_if'] != -1
            self.cf_events[self.preds[j]['what_if']] = []
            for c in self.preds[j]['collisions']:
                obj1_idx = self._get_obj_idx(c['objects'][0])
                obj2_idx = self._get_obj_idx(c['objects'][1])
                self.cf_events[self.preds[j]['what_if']].append({
                        'type': 'collision',
                        'object': [obj1_idx, obj2_idx],
                        'frame': c['frame'],
                    })    

    def _get_obj_idx(self, obj):
        for o in self.objs:
            if o['color'] == obj['color'] and \
               o['material'] == obj['material'] and \
               o['shape'] == obj['shape']:
                return o['id']
        return -1

    def _get_frame_ann(self, frame_idx=None, ann_idx=None, what_if=-1):
        assert ann_idx is not None or frame_idx is not None
        target = None
        if frame_idx is not None:
            for t in self.preds[what_if+1]['trajectory']:
                if t['frame_index'] == frame_idx:
                    target = t
                    break
        else:
            target = self.preds[what_if+1]['trajectory'][ann_idx]
        if target is None:
            raise ValueError('Invalid input frame')
        return target

    def _get_obj_location(self, obj_idx, frame_idx=None, ann_idx=None, what_if=-1):
        assert self.is_visible(obj_idx, frame_idx, ann_idx, what_if=what_if)
        frame_ann = self._get_frame_ann(frame_idx, ann_idx, what_if)
        for o in frame_ann['objects']:
            if self._get_obj_idx(o) == obj_idx:
                return o['x'], o['y']

    def get_trace(self, obj, what_if=-1):
        output = []
        pred = self.preds[what_if+1]
        for t in pred['trajectory']:
            for o in t['objects']:
                if o['id'] == obj:
                    o['frame'] = t['frame_index']
                    output.append(o)
        return output

    def _get_col_frame_proposals(self, obj, what_if=-1):
        proposed_frames = []
        trace = self.get_trace(obj, what_if)
        dvs = []
        for i, o in enumerate(trace):
            if i > 0:
                dvx = o['vx'] - trace[i-1]['vx']
                dvy = o['vy'] - trace[i-1]['vy']
                dv = np.linalg.norm([dvx, dvy])
            else:
                dv = 0 
            dvs.append(dv)
        for j, dv in enumerate(dvs):
            if j != 0 and j != len(dvs)-1:
                if dv > dvs[j-1] and dv > dvs[j+1] and dv > DV_TH and dv < 5 and self.is_visible(obj, frame_idx=trace[j]['frame'], what_if=what_if):
                    proposed_frames.append(trace[j]['frame'])
        return proposed_frames

    def _get_closest_obj(self, obj, frame_idx, what_if=-1):
        assert self.is_visible(obj, frame_idx=frame_idx, what_if=what_if)
        xo, yo = self._get_obj_location(obj, frame_idx=frame_idx, what_if=what_if)
        obj_idxs = [o['id'] for o in self.objs]
        min_dist = 99999
        closest_obj = -1
        for io in obj_idxs:
            if io != obj and self.is_visible(io, frame_idx=frame_idx, what_if=what_if):
                x, y = self._get_obj_location(io, frame_idx=frame_idx, what_if=what_if)
                dist = np.linalg.norm([x-xo, y-yo])
                if dist < min_dist:
                    min_dist = dist
                    closest_obj = io
        return closest_obj, min_dist

    def _get_col_proposals(self, what_if=-1):
        cols = []
        col_pairs = []
        obj_idxs = [o['id'] for o in self.objs]
        for io in obj_idxs:
            col_frames = self._get_col_frame_proposals(io, what_if)
            for f in col_frames:
                partner, dist = self._get_closest_obj(io, f, what_if)
                if dist < DIST_TH and {io, partner} not in col_pairs:
                    col_event = {
                        'type': 'collision',
                        'object': [io, partner],
                        'frame': f,
                    }
                    cols.append(col_event)
                    col_pairs.append({io, partner})
        return cols