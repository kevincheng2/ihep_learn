export CUDA_VISIBLE_DEVICES=3

python -u run.py \
  --is_training 0 \
  --root_path ./dataset/ \
  --model_id trace_96_96 \
  --model LSTM \
  --features SYSCALL \
  --seq_len 96 \
  --label_len 76 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 393 \
  --itr 1 \
  --train_epochs 10 \
  --gpu 3 \
  --batch_size 80 \
  --learning_rate 0.001 \
  --criterion CEL \
