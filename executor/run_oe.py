"""
Run symbolic reasoning on open-ended questions
"""
import os
import json
from tqdm import tqdm
import argparse

from executor import Executor
from simulation import Simulation


parser = argparse.ArgumentParser()
parser.add_argument('--n_progs', required=True)
parser.add_argument('--use_event_ann', default=1, type=int)
parser.add_argument('--use_in', default=0, type=int)  # Interaction network for dynamics prediction
args = parser.parse_args()



if args.use_event_ann != 0:
    raw_motion_dir = 'data/propnet_preds/with_edge_supervision'
else:
    raw_motion_dir = 'data/propnet_preds/without_edge_supervision'
if args.use_in:
    raw_motion_dir = 'data/propnet_preds/interaction_network'


question_path = './data/validation.json'
program_path = 'data/parsed_programs/oe_{}pg_val_new.json'.format(args.n_progs)

with open(program_path) as f:
    parsed_pgs = json.load(f)
with open(question_path) as f:
    anns = json.load(f)

total, correct = 0, 0

pbar = tqdm(range(5000))

for ann_idx in pbar:
    question_scene = anns[ann_idx]
    file_idx = ann_idx + 10000 
    ann_path = os.path.join(raw_motion_dir, 'sim_%05d.json' % file_idx)

    sim = Simulation(ann_path, use_event_ann=(args.use_event_ann != 0))
    exe = Executor(sim)

    for q_idx, q in enumerate(question_scene['questions']):
        q_type = q['question_type']
        if q_type != 'descriptive':
            continue
        question = q['question']
        parsed_pg = parsed_pgs[str(file_idx)]['questions'][q_idx]['program']
        pred = exe.run(parsed_pg, debug=False)
        ans = q['answer']
        if pred == ans:
            correct += 1
        total += 1

    pbar.set_description('acc: {:f}%%'.format(float(correct)*100/total))
