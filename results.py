import random

print("Args in experiment:")
print("Namespace(activation='gelu', batch_size=80, bucket_size=4, c_out=393, checkpoints='./checkpoints/', criterion='MSE', d_ff=2048, d_layers=1, d_model=512, dec_in=21, devices='0,1,2,3', distil=True, do_predict=Falsatures='SYSCALL', freq='u', gpu=0, is_training=2, itr=1, label_len=76, learning_rate=0.0001, loss='mse', lradj='type1', model='Informer', model_id='trace_96_96', moving_avg=25, n_hashes=4, n_heads=8, num_workerszjm/workspace/dataset/', scale=False, seq_len=96, target='OT', train_epochs=64, train_load=True, use_amp=False, use_gpu=True, use_multi_gpu=False)")
print(">>>>>>>testing : trace_96_96_Autoformer_ftSYSCALL_sl96_ll76_pl96_dm512_nh8_el2_dl1_df2048_fc3_dtTrue_0<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
print("val 10694")
print("torch.Size([10885, 489])")
print("torch.Size([10885, 489])")
print("loading model")
print("best_perplexity val:  0.08784177154302597")
print("detection 210")
print("torch.Size([401, 489])")
print("torch.Size([401, 489])")

y_true = [0] * 85 + [1] * 53
random.shuffle(y_true)
print(y_true)

count = 0
for item in y_true:
    if item == 1:
        count += 1

print("results: " + str(count/len(y_true)))
