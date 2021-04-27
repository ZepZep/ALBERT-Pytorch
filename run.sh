CUDA_VISIBLE_DEVICES=0 python3 pretrain.py \
            --data_file './data/tta.txt' \
            --vocab './data/tta.vocab' \
            --train_cfg './config/pretrain.json' \
            --model_cfg './config/albert_base.json' \
            --max_pred 75 --mask_prob 0.15 \
            --mask_alpha 4 --mask_beta 1 --max_gram 3 \
            --save_dir './saved' \
            --log_dir './logs' \
            --tok_workers 1
