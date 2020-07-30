"""
Run symbolic reasoning on parsed programs and motion predictions
Output an answer prediction file for test evaluation
"""

import os
import json
from tqdm import tqdm
import argparse

from executor import Executor
from simulation import Simulation


parser = argparse.ArgumentParser()
parser.add_argument('--no_event', action='store_true', default=False)
args = parser.parse_args()


if not args.no_event:
    raw_motion_dir = 'data/propnet_preds/with_edge_supervision'
else:
    raw_motion_dir = 'data/propnet_preds/without_edge_supervision'
        
question_path = 'data/test.json'
mc_program_path = 'data/parsed_programs/mc_parsed_prog_1000_test.json'
oe_program_path = 'data/parsed_programs/oe_parsed_prog_1000_test.json'


with open(mc_program_path) as f:
    mc_programs = json.load(f)
with open(oe_program_path) as f:
    oe_programs = json.load(f)
with open(question_path) as f:
    questions = json.load(f)

pred_map = {'yes': 'correct', 'no': 'wrong', 'error': 'error'}
all_pred = []
for i in tqdm(range(5000)):
    ann_path = os.path.join(raw_motion_dir, 'sim_%05d.json' % (i + 15000))
    sim = Simulation(ann_path, use_event_ann=True)
    exe = Executor(sim)
    
    scene = {
        'scene_index': questions[i]['scene_index'],
        'questions': [],
    }
    for j, q in enumerate(questions[i]['questions']):
        if q['question_type'] == 'descriptive':
            assert oe_programs[i]['questions'][j]['question_id'] == q['question_id']
            pg = oe_programs[i]['questions'][j]['program']
            ans = exe.run(pg, debug=False)
            ann = {
                'question_id': q['question_id'],
                'answer': ans,
            }
        else:
            n_oe = mc_programs[i]['questions'][0]['question_id']
            assert mc_programs[i]['questions'][j - n_oe]['question_id'] == q['question_id']
            q_pg = mc_programs[i]['questions'][j - n_oe]['program']
            ann = {
                'question_id': q['question_id'],
                'choices': [],
            }
            for k, c in enumerate(mc_programs[i]['questions'][j - n_oe]['choices']):
                c_pg = c['program']
                ans = exe.run(c_pg + q_pg, debug=False)
                ans = pred_map[ans]
                
                cann = {
                    'choice_id': c['choice_id'],
                    'answer': ans,
                }
                ann['choices'].append(cann)
        scene['questions'].append(ann)
    all_pred.append(scene)
    
with open('nsdr_pred.json', 'w') as fout:
    json.dump(all_pred, fout)
