# CLEVRER


This repository holds the codes for the paper

> 
**CLEVRER: Collision Events for Video Representation and Reasoning**,
Kexin Yi and Chuang Gan and Yunzhu Li and Pushmeet Kohli and Jiajun Wu and Antonio Torralba and Joshua B. Tenenbaum, *ICLR*, 2020.
>
[[Arxiv Preprint](https://arxiv.org/pdf/1910.01442)]
[[Project Website](http://clevrer.csail.mit.edu/)]


# Usage Guide

## Prerequisites
<!-- [[back to top](#CLEVRER)] -->
The codebase is written in Python.
There are a few dependencies to run the code. 

Run the command to install dependencies.

```
pip install -r requirements
```



## Code & Data Preparation

### Get the code
<!-- [[back to top](#CLEVRER)] -->

Use git to clone this repository
```
git clone git@github.com:chuangg/CLEVRER.git
```
The code mainly consists of two parts, including the dynamics predictor and the program executor. 
- dynamics predictor: we provide the implementation of the model in folder `temporal-reasoning`.
- programe executor: the code is provided in folder `executor`. 

### Get the data
To help reproduced the reported results, we provide all the required data, including visual masks, parsed programs and dynamic predictions.
#### Data for the program executor
>
The **parsed programes** can be found under the path `./executor/data/parsed_program/`.

The **dynamic predictions** can be found [here][propnet_preds], including two versions:

- *with_edge_supervision*: the dynamics predictor is trained `with supervisions` in edges of the graph neural network.
- *without_edge_supervision*: the dynamics predictor is trained `without supervisions` in edges of the graph neural network.
>
Please extract the archieved file you download using `tar -zxvf <file_path>` and place them under the path ``executor/data`` \
(*e.g.,* `executor/data/propnet_preds/with(without)_edge_supervision`).
#### Data for the dynamics predictor
>
The **results of video frame parser** (visual masks) can be found [here][proposals].

Please download **videos** from [project page](http://clevrer.csail.mit.edu/), and extract frames before training/testing the dynamics predictor.
>
## Testing the NS-DR model
>
Go to the executor folder:
```
cd ./executor
```
Before starting, please check and modify the path of the dynamic predictions in `line 22 and 24` in `executor/run_oe.py` and `executor/run_mc.py`. Make sure the path is valid.

For open-ended questions:
```
python run_oe.py --n_progs 1000
```

For multiple-choice questions:
```
python run_mc.py --n_progs 1000
```
>

## Training/Testing the dynamics predictor
[[back to top](#CLEVRER)]

We provide the code for the `dynamics predictor` to generate *dynamics predictions* as input of the `programe executor`.
>
Before training the dynamics predictor, please make sure that you have extracted the vidoe frames (named format **frame_00000.png**) and download the proposals. 
Please organize the files with the same structure as (or you can modify the frame path in `data.py` and `eval.py`.):
```
video_frames/
├─sim_00000/frame_00000.png, frame_00001.png,... 
├─sim_00001/frame_00000.png, frame_00001.png,...
├─sim_00002/...
├─...
├─...
└sim_19999/...

processed_proposals/
├─sim_00000.json
├─sim_00001.json
├─sim_00002.json
├─...
├─...
└ sim_19999.json
```
`Note`: the index of videos/frames is starting from 0 (*i,e,. 00000*).

#### Training
The training scripts can be found in `temporal-reasoning/scripts/`, and the main arguments in the scripts including:
- gen_valid_idx: set to 1 at the first training. 
- **data_dir**: the directory of extracted frames.
- **label_dir**: the directory of the downloaded proposals.
- resume_epoch/iter: the checkpoint information (set to 0 if no checkpoints.)

The name of scripts `train*.sh` is self-explained.
More details about arguments can be found in `./temporal-reasoning/train.py`.
>
Start training
```
cd ./temporal-reasoning 
bash scripts/train.sh   
```


The models and log will be save in a folder with a name like *files_CLEVR_pn_pstep2* in `temporal-reasonging` folder.

#### Testing
>
The tesing scripts can also be found in `temporal-reasoning/scripts`. One training script corresponds to one testing script. The testing script contains serveral arguments:
- des_dir: the directory for saving the output dynamic predictions.
- st_idx: the starting index for a data split (10000 for validation set; 15000 for testing set).
- ed_idx: the ending index for a data split (15000 for validation set; 20000 for testing set).
- epoch/iter: the checkpoint information.

More details about arguments can be found in `./temporal-reasoning/train.py`.
>
Start testing
```
cd ./temporal-reasoning
bash scripts/eval.sh
```
The dynamic predictions will be saved under the directory as you set in `--des_dir` in `eval.sh`. 

You can then feed these dynamic predictions as input of the `program executor` and enjoy.

[[back to top](#CLEVRER)]

## Citation
Please cite the following paper if you feel this repository useful.
```
@inproceedings{CLEVRER2020ICLR,
  author    = {Kexin Yi and
               Chuang Gan and
               Yunzhu Li and
               Pushmeet Kohli and
               Jiajun Wu and
               Antonio Torralba and
               Joshua B. Tenenbaum},
  title     = {{CLEVRER:} Collision Events for Video Representation and Reasoning},
  booktitle = {ICLR},
  year      = {2020}
}
```



[propnet_preds]:https://drive.google.com/file/d/1u2OdG59Zl1PqNAnXZjDVMmhXSy3czR44/view?usp=sharing
[proposals]:xxxx