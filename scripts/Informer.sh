export CUDA_VISIBLE_DEVICES=3


# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --model_id trace_96_96 \
#   --model Informer \
#   --features SYSCALL \
#   --seq_len 96 \
#   --label_len 76 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 21 \
#   --c_out 393 \
#   --itr 1 \
#   --train_epochs 64 \
#   --gpu 0 \
#   --batch_size 32 \
#   --learning_rate 0.0001 \


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id trace_96_128 \
  --model Informer \
  --features SYSCALL \
  --seq_len 96 \
  --label_len 96 \
  --pred_len 128 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 393 \
  --itr 1 \
  --train_epochs 64 \
  --gpu 0 \
  --batch_size 28 \
  --learning_rate 0.0001 \


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id trace_96_192 \
  --model Informer \
  --features SYSCALL \
  --seq_len 96 \
  --label_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 393 \
  --itr 1 \
  --train_epochs 64 \
  --gpu 0 \
  --batch_size 24 \
  --learning_rate 0.0001 \


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --model_id trace_96_268 \
  --model Informer \
  --features SYSCALL \
  --seq_len 96 \
  --label_len 96 \
  --pred_len 268 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 393 \
  --itr 1 \
  --train_epochs 64 \
  --gpu 0 \
  --batch_size 20 \
  --learning_rate 0.0001 \
