CUDA_VISIBLE_DEVICES=2 python train.py \
--gen_valid_idx 1 \
--data_dir /data/vision/billf/scratch/kyi/projects/temporal-physics-reasoning/data/clevrer/ver1.0/frames \
--label_dir /data/vision/billf/scratch/kyi/projects/temporal-physics-reasoning/arxiv_dec_2018/data/derender/processed_proposals \
--resume_epoch 0 \
--resume_iter 0
