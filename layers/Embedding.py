import torch
import torch.nn as nn
import math

def compared_version(ver1, ver2):
    """
    :param ver1
    :param ver2
    :return: ver1< = >ver2 False/True
    """
    list1 = str(ver1).split(".")
    list2 = str(ver2).split(".")
    
    for i in range(len(list1)) if len(list1) < len(list2) else range(len(list2)):
        if int(list1[i]) == int(list2[i]):
            pass
        elif int(list1[i]) < int(list2[i]):
            return -1
        else:
            return 1
    
    if len(list1) == len(list2):
        return True
    elif len(list1) < len(list2):
        return False
    else:
        return True

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, device):
        super(TokenEmbedding, self).__init__()
        padding = 1 if compared_version(torch.__version__, '1.5.0') else 2
        self.tokenLinear = nn.Linear(d_model*325, d_model)
        self.encoddingLinear = nn.Linear(11, int(d_model/2))
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.hid_dims = d_model
        self.device = device
        self.c_in = c_in

        self.embed = nn.Embedding(c_in, self.hid_dims)
        self.init_wrights()

    def init_wrights(self):
        nn.init.normal_(self.embed.weight, 0, self.hid_dims ** -0.5)

    def forward(self, embeding_data, encoding_data):
        embedding_data = torch.abs(embeding_data).long()
        embedding_data = embedding_data.clamp(0, self.c_in-1)
        embed_out = self.embed(embedding_data)
        embed_out = self.tokenLinear(embed_out.view(embed_out.shape[0], embed_out.shape[1], -1))

        encoding_data = self.get_encoding(encoding_data)
        return embed_out + encoding_data

    def get_input_shape(self, input, tar_nums):
        col_num = input.shape[2]
        if col_num == tar_nums:
            return input
        elif col_num < tar_nums:
            zeros_tensor = torch.zeros((input.shape[0], input.shape[1], tar_nums-col_num)).to(self.device)
            return torch.cat((input, zeros_tensor), 2)
        return input[:, :, :tar_nums]
    
    def get_encoding(self, input):
        position = self.encoddingLinear(input)
        denominator = torch.exp(torch.arange(0, self.hid_dims, 2).float() * 
                                (-math.log(1e6) / 4)).to(self.device)
        pos_zeros = torch.zeros(input.shape[0], input.shape[1], self.hid_dims).to(self.device)
        pos_zeros[:, :, 0::2] = torch.sin(position*denominator)
        pos_zeros[:, :, 1::2] = torch.cos(position*denominator)
        return pos_zeros


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, freq='u'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'u': 7, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, device, freq='u', dropout=0.1, encoding_shape=9):
        super(DataEmbedding, self).__init__()

        self.encoding_shape = encoding_shape
        self.value_embedding = TokenEmbedding(c_in=c_in,
                                              d_model=d_model,
                                              device=device)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding =  TimeFeatureEmbedding(d_model=d_model, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        embeding_set = x[:, :, :-self.encoding_shape]
        encoding_set = x[:, :, -self.encoding_shape:]

        x = self.value_embedding(embeding_set, encoding_set) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, device, freq='u', dropout=0.1, encoding_shape=11):
        super(DataEmbedding_wo_pos, self).__init__()

        self.encoding_shape = encoding_shape
        self.value_embedding = TokenEmbedding(c_in=c_in,
                                              d_model=d_model,
                                              device=device)
        self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        embeding_set = x[:, :, :-self.encoding_shape]
        encoding_set = x[:, :, -self.encoding_shape:]
        value_out = self.dropout(self.value_embedding(embeding_set, encoding_set))
        return value_out + self.temporal_embedding(x_mark)
