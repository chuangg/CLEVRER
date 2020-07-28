import os
import numpy as np
import json
import argparse
from pprint import pprint
import pycocotools._mask as _mask
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--idx_video', type=int, default=375)
parser.add_argument('--idx_frame', type=int, default=95)
parser.add_argument('--read_src', default='derender_proposals')

args = parser.parse_args()


def decode(rleObjs):
    if type(rleObjs) == list:
        return _mask.decode(rleObjs)
    else:
        return _mask.decode([rleObjs])[:,:,0]


with open(os.path.join(args.read_src, 'sim_%05d.json' % args.idx_video)) as f:
    data = json.load(f)

pprint(data)

'''
print(data['video_name'])

frame = data['frames'][args.idx_frame]
print('frame_name', frame['frame_filename'])
print('frame_index', frame['frame_index'])

objects = frame['objects']
print(len(objects))

# pprint(objects)

for i in range(len(objects)):
    print(objects[i]['material'], objects[i]['color'], objects[i]['shape'])
    mask = decode(objects[i]['mask'])
    print(np.sum(mask))
    mask = cv2.resize(mask, (120, 80), interpolation=cv2.INTER_NEAREST)
    # cv2.imwrite('mask_%d.png' % i, mask * 255)
'''
