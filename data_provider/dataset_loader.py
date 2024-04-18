from data_provider.trace_data import TraceDataset, get_data_index_json
from torch.utils.data import DataLoader


def data_provider(args, flag):
    # trace_path, device, batch_size, seq_len, label_len, pred_len, num_workers, scale=True
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
    else:
        shuffle_flag = True
        drop_last = True

    data_set = TraceDataset(
        trace_path=args.root_path,
        device=args.device,
        batch_size=args.batch_size,
        size=[args.seq_len, args.label_len, args.pred_len],
        scale=args.scale,
        flag=flag
    )
    print(flag, len(data_set))
    data_set.show_shape()
    
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
