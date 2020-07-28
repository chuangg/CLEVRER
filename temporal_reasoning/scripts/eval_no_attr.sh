CUDA_VISIBLE_DEVICES=0 python eval.py \
    --use_attr 0 \
    --edge_superv 0 \
    --des_dir propnet_predictions \
    --data_dir /data/vision/billf/scratch/kyi/projects/temporal-physics-reasoning/data/clevrer/ver1.0/frames \
    --label_dir /data/vision/billf/scratch/kyi/projects/temporal-physics-reasoning/arxiv_dec_2018/data/derender/processed_proposals \
    --video 0 \
    --st_idx 10000 \
    --ed_idx 20000 \
    --epoch 5 \
    --iter 700000 \
		

