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
[[back to top](#CLEVRER)]
The codebase is written in Python.
There are a few dependencies to run the code. 

Run the command to install dependencies.

```
pip install -r requirements
```



## Code & Data Preparation

### Get the code
[[back to top](#CLEVRER)]

Use git to clone this repository and its submodules
```
git clone git@github.com:chuangg/CLEVRER.git
```
The code mainly consists of two parts, including the dynamic predictor and the program executor. 
- dynamic predictor: we provide the implementation of the model in folder `temporal-reasoning`.
- programe executor: the code is provided in folder `executor`. we provide the parsed programs and dynamic predictions and you can feed them into the program executor to reproduce the reported results in the paper.

### Get the data
To help reproduced the reported results, we provide the parsed programs and dynamic predictions.
>
The parsed programes can be found under the path `.executor/data/parsed_program/`.

The dynamic predictions can be found [here][propnet_preds], including two versions:

- with_edge_supervision: the dynamic model is trained `with edge supervision` in the graph neural network.
- without_edge_supervision: the dynamic model is trained `without supervision` in edges of the graph neural network.
>
Please extract the archieved file you download using `tar -zxvf <file_path>` and place them in the ``data`` folder.

## Testing the NS-DR model
>
Go to the executor folder:
```
cd ./executor
```
Before starting, please check and modify the path of dynamic prediction in `line 22 and 24` in `run_oe.py` and `run_mc.py`. Make sure the path is valid.

For open-ended questions:
```
python run_oe.py --n_progs 1000
```

For multiple-choice questions:
```
python run_mc.py --n_progs 1000
```
>

## Training the dynamic predictor
[[back to top](#CLEVRER)]

TODO


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




[pytorch]:https://pytorch.org/
[cocoapi]:https://github.com/cocodataset/cocoapi
[parsed_prgs]: xxxxxxxx
[propnet_preds]:https://drive.google.com/file/d/1u2OdG59Zl1PqNAnXZjDVMmhXSy3czR44/view?usp=sharing