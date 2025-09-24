python cls_train.py \
    --model_type tcn \
    --fold 0 \
    --data MITECG \
    --seed 42 \
    --epoch 1

# python baseline.py \
#     --model_type tcn \
#     --fold 0 \
#     --data MITECG \
#     --seed 42

python main.py \
    --train_type            ppo_v3 \
    --nb_transform          r_p \
    --terminate_type        rel_ce_diff \
    --dataset               MITECG \
    --split                 0 \
    --mask_type             mean \
    --epochs                5 \
    --ppo_epochs            4 \
    --max_segment           5 \
    --note                  '_' \
    --seg_dist              cat_cat \
    --entropy_coef          0.01 \
    --weights               1.0,0.3 \
    --backbone              tcn,cnn \
    --batch_size            256 \
    --rollout_len           1024 \
    --seed                  42 \
    --threshold             0.3