D:\Anaconda3\envs\pytorch\python.exe -u run.py ^
  --is_training 1 ^
  --root_path ./dataset/ ^
  --model_id trace_96_96 ^
  --model LSTM ^
  --features SYSCALL ^
  --seq_len 96 ^
  --label_len 76 ^
  --pred_len 96 ^
  --e_layers 4 ^
  --d_layers 1 ^
  --factor 3 ^
  --enc_in 21 ^
  --dec_in 21 ^
  --c_out 393 ^
  --itr 1 ^
  --train_epochs 10 ^
  --gpu 4 ^
  --batch_size 4 ^
  --learning_rate 0.0001 ^
  --criterion CEL ^
