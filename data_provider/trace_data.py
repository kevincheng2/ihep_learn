import os
import re
import sys
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features


def in_list(list_data, tag):
    for item in list_data:
        if tag in item:
            return True
    return False


def mkdir_if_not(path):
    if not os.path.exists(path):
        os.mkdir(path)


def transfer_number(input_str):
    if type(input_str) != str:
        return input_str

    pattern = r'^[0-9a-fA-F]+$'
    if re.match(pattern, input_str):
        return int(input_str, 16)
    else:
        return 0


def read_file(root_path, col_list, data_nums=sys.maxsize, max_len=200):
    df = pd.DataFrame(columns=col_list)

    idx = 0
    with open(root_path, encoding='utf-8') as f:
        while True:
            if idx > data_nums:
                break
            else:
                idx = idx + 1
            line = f.readline()
            if not line:
                break
            js_data = json.loads(line)
            loc_data = []
            for key, val in js_data.items():
                val = str(val).strip()
                if len(val) > max_len:
                    val = val[:max_len]
                loc_data.append(val)
            df.loc[len(df.index)] = loc_data
    return df


def split_args(args_list, args_col_list, val_max=40):
    """
    将 evt_args 列根据空格拆分成多个列
    :param args_list:
    :param col_list:
    :return:
        拆分后重建的 dataframe
    """
    print("删除args中的无用字符")
    args_col_list_tmp = []
    for idx, row in args_list.iterrows():
        args_str = row["evt_args"].split(' ')
        for arg in args_str:
            if '=' not in arg:
                continue
            if not in_list(args_col_list_tmp, arg.split('=')[0]):
                args_col_list_tmp.append(arg.split('=')[0])

    print("开始构建args dataframe")
    df = pd.DataFrame(columns=args_col_list)
    for idx, row in args_list.iterrows():
        args_str = row["evt_args"].split(' ')
        loc_data = [pd.NA for x in range(0, len(args_col_list))]
        for arg in args_str:
            if '=' not in arg:
                continue
            key = arg.split('=')[0].strip()
            if key not in args_col_list:
                continue
            val = re.sub(r'[^A-Za-z0-9]', "", arg.split('=')[1].strip())
            if len(val) > val_max:
                val = val[:val_max]
            loc_data[args_col_list.index(key)] = val
        df.loc[len(df.index)] = loc_data
    return df


def missing_value(df_data, proportion=0.01):
    """
    删除空值超过给定比例(proportion)的列值
    :return:
        待删除的列名
    """
    df_notnull = df_data.notnull().sum()
    del_list = []
    for key, val in df_notnull.items():
        if val < len(df_data)*proportion:
            del_list.append(key)

    return del_list


def get_target_shape(input, tar_nums):
    rows, columns = input.shape
    if columns == tar_nums:
        return input
    elif columns < tar_nums:
        zero_pd = pd.DataFrame(np.zeros((rows, tar_nums-columns)))
        return pd.concat([input, zero_pd], axis=1)
    else:
        return input.iloc[:, :tar_nums]


def onehot_code(df_data, index_json):
    """
    :param 
        df_data: 待编码的数据，dataframe 类型
        index_json(dict(tuple))：列名、编号、编码列数的对照关系
    :return:
     merge_pd(dataframe)：编码后的矩阵
    """
    # col_names = df_data.columns.tolist()
    # col_idx = {}
    # merge_pd = pd.DataFrame()
    # for idx, name in enumerate(col_names):
    #     col_onehot = pd.get_dummies(df_data[[name]], columns=[name], dtype=int)
    #     merge_pd = pd.concat([merge_pd, col_onehot], axis=1)
    #     col_idx[name] = (idx, len(col_onehot.columns))
    # return merge_pd, col_idx

    index_json = dict(sorted(index_json.items(), key=lambda item: item[1][0]))
    merge_pd = pd.DataFrame()
    for key, value in index_json.items():
        col_onehot = pd.get_dummies(df_data[[key]], columns=[key], dtype=int)
        col_onehot = get_target_shape(input=col_onehot, tar_nums=int(value[1]))
        merge_pd = pd.concat([merge_pd, col_onehot], axis=1)
    return merge_pd


def building_dataset(original_path, save_path, standard_path):
    if not os.path.exists(original_path) or not os.path.exists(standard_path):
        print(f"file {original_path} or {standard_path} not exist.")
        return

    mkdir_if_not(save_path)
    # 读取原始数据
    col_list = ["evt_args", "evt_cpu", "evt_deltatime", "evt_dir", "evt_time", "fd_name",
                "fd_num", "fd_type", "proc_fdopencount", "proc_name", "proc_pid", "proc_vmsize",
                "syscall_type", "thread_ismain", "thread_tid", "thread_vmsize", "user_name",
                "user_uid", "hostname"]
    df_data = read_file(root_path=original_path, col_list=col_list)
    print("文件读取成功")

    # 对参数数据进行拆分
    args_encod_col = ["length", "addr"]
    args_embed_col = ["res", "mode", "flags", "ino", "dev", "prot", "dirfd"]
    df_args = split_args(args_list=df_data[['evt_args']], args_col_list=args_encod_col+args_embed_col)
    print("args字段拆分成功")

    df_data = df_data.drop(columns='evt_args', axis=1)
    df_data = pd.concat([df_data, df_args], axis=1)

    # 删除缺失值过多的列
    df_data = df_data.replace(to_replace="None", value=np.nan)
    df_data = df_data.dropna(axis=0, subset=['syscall_type'])
    df_data = df_data.reset_index(drop=True)
    del_list = missing_value(df_data, proportion=0.001)
    df_data = df_data.drop(columns=del_list, axis=1)

    # 根据数据类型，分别进行 embedding 和 encoding 处理
    df_data["addr_dec"] = df_data["addr"].apply(transfer_number)
    col_names_list = df_data.columns.tolist()

    all_encoding_col = ["evt_cpu", "evt_deltatime", "evt_time", "fd_num", "proc_fdopencount", "proc_pid",
                        "proc_vmsize", "thread_tid", "thread_vmsize", "user_uid", "ptid", "vm_size",
                        "vm_rss", "vm_swap"]
    all_embedding_col = ["evt_dir", "fd_name", "fd_type", "proc_name", "thread_ismain", "user_name", "hostname"]
    all_encoding_col.extend(["addr_dec", "length"])
    all_embedding_col.extend(args_embed_col)

    encoding_col = []
    embedding_col = []
    for col_name in col_names_list:
        if col_name in all_encoding_col:
            encoding_col.append(col_name)
        elif col_name in all_embedding_col:
            embedding_col.append(col_name)

    # pd_encode = df_data[encoding_col]
    pd_embed = df_data[embedding_col]

    # 对部分数据进行 onthot 编码处理
    print("开始进行 onehost 编码")
    with open(os.path.join(standard_path, "index.json"), 'r') as f:
        index_json = json.load(f)

    data_onehot = onehot_code(pd_embed, index_json)

    pd_encod = df_data[encoding_col]
    label = df_data["syscall_type"].to_frame()

    data_onehot.to_csv(os.path.join(save_path, "embedding_onehot.csv"), index=False, header=False)
    pd_encod.to_csv(os.path.join(save_path, "encoding_data.csv"), index=False)
    label.to_csv(os.path.join(save_path, "label_data.csv"), index=False)


def load_dataset_info(standard_path, device):
    if not os.path.exists(standard_path):
        print(f"file path {standard_path} is not exist")
        return

    vocab_dict = load_vocab(os.path.join(standard_path, "vocab.txt"))
    with open(os.path.join(standard_path, "index.json"), 'r') as f:
        index_json = json.load(f)

    return index_json, vocab_dict, len(vocab_dict)


def load_vocab(path):
    count = 0
    vocab_dict = {}
    with open(path, 'r') as file:
        for line in file:
            if len(line) > 1:
                vocab_dict[line.replace('\n', "")] = count
                count += 1
    return vocab_dict


def get_data_index_json(file_path):
    if not os.path.exists(file_path):
        print(f"file path {file_path} is not exist")
        return

    vocab_dict = load_vocab(os.path.join(file_path, "vocab.txt"))
    with open(os.path.join(file_path, "index.json"), 'r') as f:
        index_json = json.load(f)
    return index_json, vocab_dict, len(vocab_dict)


class TraceDataset(Dataset):
    def __init__(self, trace_path, device, batch_size=64, size=None, freq='u', scale=True, flag='train'):
        super().__init__()
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.freq = freq
        self.device = device
        self.file_path = trace_path
        
        self.type_map = {'train': 0.8, 'val': 0.1, 'test': 0.1}
        flag_map = {'train': 0, 'test': 1, 'val': 2}
        assert flag in ['train', 'test', 'val']
        self.set_type = flag_map[flag]
        self.flag = flag
        self.scale = scale

        self.index_json, self.vocab_dict, self.out_syscall_len = get_data_index_json(self.file_path)
        self.load_dataset()

    def load_dataset_info(self):
        if not os.path.exists(self.file_path):
            print(f"file path {self.file_path} is not exist")
            return

        vocab_dict = load_vocab(os.path.join(self.file_path, "vocab.txt"))
        with open(os.path.join(self.file_path, "index.json"), 'r') as f:
            index_json = json.load(f)

        return index_json, vocab_dict, len(vocab_dict)

    def load_dataset(self):
        trace_dir = os.path.abspath(os.path.join(self.file_path, "format_data"))
        if not os.path.exists(trace_dir) :
            print(f"file path /{trace_dir}/ is not exist ")
            return
        
        self.scaler = StandardScaler()
        # traces_path = os.path.join(trace_dir, "traces_dataset.csv")
        traces_path = os.path.join(trace_dir, "trace_all_normal.csv")

        traces_dataset = pd.read_csv(traces_path, encoding='utf-8')
        traces_dataset.fillna("0", inplace=True)
        traces_dataset.fillna("-1", inplace=True)

        # label 增加一列，标签对应的数字
        traces_dataset['nums'] = traces_dataset.apply(lambda row: self.vocab_dict[row['syscall_type']], axis=1)

        traces_dataset.loc[traces_dataset['dirfd'].notnull(), 'dirfd'] = 1

        # 开始处理数据
        data_stamp = time_features(pd.to_datetime(traces_dataset['evt_time'].values, unit='ns'), freq=self.freq)
        # trans_set = traces_dataset.drop(["evt_time", "evt_deltatime", "syscall_type"], axis=1).values
        trans_set = traces_dataset.drop(["evt_time", "syscall_type"], axis=1).values
        data = torch.tensor(trans_set.astype(float), dtype=torch.float32)
        
        if self.scale:
            data = self.scaler.fit_transform(data)
        data_stamp = data_stamp.transpose(1, 0)

        # 划分数据集
        train_len = int(len(data) * self.type_map['train'])
        test_len = int(len(data) * self.type_map['test'])
        data_begin = [0, train_len-self.seq_len, train_len+test_len-self.seq_len]
        # data_end = [train_len, train_len+test_len, len(data)]
        data_end = [len(data), train_len+test_len, len(data)]
        begin_index = data_begin[self.set_type]
        end_index = data_end[self.set_type]

        self.data_x = data[begin_index:end_index]
        self.data_y = data[begin_index:end_index]
        self.data_stamp = data_stamp[begin_index:end_index]

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


if __name__ == '__main__':
    # file_path = r"./trace_dataset"
    # save_dataset(file_path=file_path)

    # current_folder = os.getcwd()
    # original_dir = os.path.join(current_folder, "trace_dataset", "original_data")
    # save_dir = os.path.join(current_folder, "trace_dataset", "format_data")
    # standard_dir = os.path.join(current_folder, "trace_dataset", "standard_data")

    # original_dir_list = os.listdir(original_dir)
    # for dir_name in original_dir_list:
    #     original_path = os.path.join(original_dir, dir_name)
    #     dir_name = dir_name.split(".")[0]
    #     save_path = os.path.join(save_dir, dir_name)
    #     print("=" * 50 + f"\n{original_path:^50s}\n" + "=" * 50)
    #     building_dataset(original_path=original_path, save_path=save_path, standard_path=standard_dir)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # current_folder = os.getcwd()
    # trace_dir = os.path.abspath(os.path.join(current_folder, "trace_dataset"))

    # standard_path = os.path.join(trace_dir, "standard_data")
    # save_dir = os.path.join(trace_dir, "format_data")
    # index_json, vocab_dict, n_syscall = load_dataset_info(standard_path=standard_path, device=device)

    # format_file_path = os.path.join(save_dir, "BES_399865")
    # data_set_path = r"/mnt/workspace/Autoformer/dataset"
    data_set_path = r"C:\Users\dell\Documents\vsCode\Autoformer\dataset"
    dataset = TraceDataset(trace_path=data_set_path, device=device)
    # train_iter = dataset.get_varlen_iter()

    # for batch, (data, target, seq_len) in enumerate(train_iter):
    #     print(batch)
    #     print(data.shape)
    #     print(target.shape)
    #     print(seq_len)
    #     break

    # embeding_dataset, encoding_dataset, index_json = load_dataset("./")
    # print(embeding_dataset)
    # print(encoding_dataset)
    # print(index_json)
