import torch
import torch.nn as nn
from layers.Embedding import DataEmbedding


class Model(nn.Module):
    """
    LSTM
    """
    def __init__(self, configs):
        super(Model, self).__init__()

        self.enc_embedding = DataEmbedding(c_in=configs.enc_in, 
                                            d_model=configs.d_model,
                                            device=configs.device,
                                            freq=configs.freq,
                                            dropout=configs.dropout,
                                            encoding_shape=configs.encoding_shape)
        
        self.seq_linear = nn.Linear(configs.seq_len, configs.pred_len)
        self.lstm = nn.LSTM(configs.d_model, configs.c_out, configs.e_layers, batch_first=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        src = self.enc_embedding(x_enc, x_mark_enc)
        src, _ = self.lstm(src)
        src = self.seq_linear(src.permute(0, 2, 1))
        src = src.permute(0, 2, 1)
        return src
