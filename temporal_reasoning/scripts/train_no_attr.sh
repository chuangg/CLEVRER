CUDA_VISIBLE_DEVICES=2 python train.py \
--use_attr 0 \
--edge_superv 0 \
--gen_valid_idx 0 \
--data_dir /data/vision/billf/scratch/kyi/projects/temporal-physics-reasoning/data/clevrer/ver1.0/frames \
--label_dir /data/vision/billf/scratch/kyi/projects/temporal-physics-reasoning/arxiv_dec_2018/data/derender/processed_proposals \
--resume_epoch 1 \
--resume_iter 300000
