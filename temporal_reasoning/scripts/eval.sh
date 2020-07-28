CUDA_VISIBLE_DEVICES=0 python eval.py \
	--des_dir propnet_predictions_v1.0 \
	--data_dir /data/vision/billf/scratch/kyi/projects/temporal-physics-reasoning/data/clevrer/ver1.0/frames \
	--label_dir /data/vision/billf/scratch/kyi/projects/temporal-physics-reasoning/arxiv_dec_2018/data/derender/processed_proposals \
	--video 0 \
	--st_idx 10000 \
	--ed_idx 15000 \
	--epoch 5 \
	--iter 800000 
		

