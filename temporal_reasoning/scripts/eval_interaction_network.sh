CUDA_VISIBLE_DEVICES=1 python eval.py \
	--pstep 1 \
	--des_dir propnet_predictions_interaction_network \
	--data_dir /data/vision/billf/scratch/kyi/projects/temporal-physics-reasoning/data/clevrer/ver1.0/frames \
	--label_dir /data/vision/billf/scratch/kyi/projects/temporal-physics-reasoning/arxiv_dec_2018/data/derender/processed_proposals \
	--video 0 \
	--st_idx 15000 \
	--ed_idx 20000 \
	--epoch 3 \
	--iter 650000 \
		

