CUDA_VISIBLE_DEVICES=3 python train.py \
--gen_valid_idx 0 \
--edge_superv 0 \
--data_dir /data/vision/billf/scratch/kyi/projects/temporal-physics-reasoning/data/clevrer/ver1.0/frames \
--label_dir /data/vision/billf/scratch/kyi/projects/temporal-physics-reasoning/arxiv_dec_2018/data/derender/processed_proposals \
--resume_epoch 5 \
--resume_iter 350000
