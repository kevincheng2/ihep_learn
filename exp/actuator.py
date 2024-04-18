import os
import time
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import sklearn.metrics as skmetrics
import warnings
import torch.nn.functional as F

warnings.filterwarnings('ignore')

from data_provider.dataset_loader import data_provider
from models import Informer, Autoformer, Transformer, Reformer, LSTM
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric


class Actuator(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        args.device = self.device
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'LSTM': LSTM,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                if self.args.criterion == "CEL":
                    batch_y = torch.max(batch_y, dim=-1)[1]
                    outputs = outputs.permute(0, 2, 1)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        if self.args.train_load:
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))
            print("load model success.")

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.criterion)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss, train_acc = [], []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                if self.args.criterion == "CEL":
                    batch_y = torch.max(batch_y, dim=-1)[1]
                    outputs = outputs.permute(0, 2, 1)

                loss = criterion(outputs, batch_y)
                loss.requires_grad_(True)

                if self.args.criterion == "CEL":
                    acc_outputs = torch.max(outputs.permute(0, 2, 1), dim=-1)[1]
                    acc_y = batch_y
                else:
                    acc_outputs = torch.max(outputs, dim=-1)[1]
                    acc_y = torch.max(batch_y, dim=-1)[1]

                acc = float((acc_outputs == acc_y).sum().item()) / acc_outputs.numel()

                train_loss.append(loss.item())
                train_acc.append(acc)

                if (i + 1) % 100 == 0:
                    print("\titers: {0:4d}, epoch: {1:2d} | loss: {2:.7f} | acc: {3:.2%}".format(i + 1, epoch + 1, loss.item(), acc), end=" ")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('| speed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_acc = np.average(train_acc)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0:2d}, Steps: {1:1d} | Train Loss: {2:.7f} Train Acc: {3:.4%} Vali Loss: {4:.7f} Test Loss: {5:.7f}".format(
                epoch + 1, train_steps, train_loss, train_acc, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        return

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                pred = torch.max(outputs, dim=-1)[1]    # .squeeze()
                true = torch.max(batch_y, dim=-1)[1]    # .squeeze()
            
                pred = pred.detach().cpu().numpy()
                true = true.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def odd_detection(self, setting):
        train_data, train_loader = self._get_data(flag='val')

        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))

        folder_path = './detection_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        criterion = nn.CrossEntropyLoss(reduction="none")
        criterion.cuda(self.device)

        self.model.eval()
        accuracy, precision, recall, fscore = [], [], [], []
        perplexity_list = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                batch_y_c = torch.max(batch_y, dim=-1)[1]
                outputs_c = outputs.permute(0, 2, 1)

                loss = criterion(outputs_c, batch_y_c)
                loss = loss.reshape(batch_y_c.shape)
                loss = torch.sum(loss, 1)
                perplexity_res = (loss - torch.mean(loss))/torch.std(loss)
                perplexity_list.extend(torch.exp(perplexity_res).cpu().detach().tolist())

                pred = torch.max(outputs, dim=-1)[1]    # .squeeze()
                true = torch.max(batch_y, dim=-1)[1]    # .squeeze()
                pred = pred.detach().cpu().numpy()
                true = true.detach().cpu().numpy()

                for idx in range(self.args.batch_size):
                    y_pred = pred[idx].flatten()
                    y_true = true[idx].flatten()
                    precision.append(
                        skmetrics.precision_score(y_true, y_pred, average='micro')
                    )
                    recall.append(skmetrics.recall_score(y_true, y_pred, average='micro'))  
                    accuracy.append(skmetrics.accuracy_score(y_true, y_pred))
                    fscore.append(skmetrics.f1_score(y_true, y_pred, average='micro'))

        id_best_threshold = np.argmax(fscore)
        # thresholds = np.arange(min(perplexity_list), max(perplexity_list),
        #                        step=(max(perplexity_list) - min(perplexity_list)) / 100)

        best_perplexity = perplexity_list[id_best_threshold]

        print("best_perplexity val: ", best_perplexity)

        odd_data, odd_loader = self._get_data(flag='detection')
        odd_perplexity = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(odd_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                batch_y_c = torch.max(batch_y, dim=-1)[1]
                outputs_c = outputs.permute(0, 2, 1)

                loss = criterion(outputs_c, batch_y_c)
                loss = loss.reshape(batch_y_c.shape)
                loss = torch.sum(loss, 1)
                perplexity_res = (loss - torch.mean(loss))/torch.std(loss)
                odd_perplexity.extend(torch.exp(perplexity_res).cpu().detach().tolist())

                # pred = torch.max(outputs, dim=-1)[1]    # .squeeze()
                # true = torch.max(batch_y, dim=-1)[1]    # .squeeze()
            
                # pred = pred.detach().cpu().numpy()
                # true = true.detach().cpu().numpy()

                # for idx in range(self.args.batch_size):
                #     y_pred = pred[idx].flatten()
                #     y_true = true[idx].flatten()
                #     precision.append(
                #         skmetrics.precision_score(y_true, y_pred, average='micro')
                #     )
                #     recall.append(skmetrics.recall_score(y_true, y_pred, average='micro'))
                #     accuracy.append(skmetrics.accuracy_score(y_true, y_pred))
                #     fscore.append(skmetrics.f1_score(y_true, y_pred, average='micro'))

        # normal_data, normal_loader = self._get_data(flag='val')
        # normal_perplexity = []

        # self.model.eval()
        # with torch.no_grad():
        #     for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(normal_loader):
        #         batch_x = batch_x.float().to(self.device)
        #         batch_y = batch_y.float().to(self.device)

        #         batch_x_mark = batch_x_mark.float().to(self.device)
        #         batch_y_mark = batch_y_mark.float().to(self.device)

        #         outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

        #         batch_y_c = torch.max(batch_y, dim=-1)[1]
        #         outputs_c = outputs.permute(0, 2, 1)

        #         loss = criterion(outputs_c, batch_y_c)
        #         loss = loss.reshape(batch_y_c.shape)
        #         loss = torch.sum(loss, 1)
        #         perplexity_res = (loss - torch.mean(loss))/torch.std(loss)
        #         normal_perplexity.extend(torch.exp(perplexity_res).cpu().detach().tolist())

                # pred = torch.max(outputs, dim=-1)[1]    # .squeeze()
                # true = torch.max(batch_y, dim=-1)[1]    # .squeeze()
            
                # pred = pred.detach().cpu().numpy()
                # true = true.detach().cpu().numpy()

                # for idx in range(self.args.batch_size):
                #     y_pred = pred[idx].flatten()
                #     y_true = true[idx].flatten()
                #     precision.append(
                #         skmetrics.precision_score(y_true, y_pred, average='micro')
                #     )
                #     recall.append(skmetrics.recall_score(y_true, y_pred, average='micro'))
                #     accuracy.append(skmetrics.accuracy_score(y_true, y_pred))
                #     fscore.append(skmetrics.f1_score(y_true, y_pred, average='micro'))

        # odd_test = odd_perplexity + normal_perplexity
        # y_true = [0] * len(odd_perplexity) + [1] * len(normal_perplexity)
        y_pred = [0 if p > best_perplexity else 1 for p in odd_perplexity]

        print(y_pred)

        count = 0
        for item in y_pred:
            if item == 1:
                count += 1
        
        print("results: " + str(count/len(y_pred)))

        # precision = skmetrics.precision_score(y_true, y_pred, average='micro')
        # recall = skmetrics.recall_score(y_true, y_pred, average='micro')
        # accuracy = skmetrics.accuracy_score(y_true, y_pred)
        # fscore = skmetrics.f1_score(y_true, y_pred, average='micro')
        # auroc = skmetrics.roc_auc_score(y_true, y_pred, average='micro')

        # print(f"{'    AUROC':30}: {auroc:68.2%}")
        # print(f"{'    Recall':30}: {recall:68.2%}")
        # print(f"{'    Precision':30}: {precision:68.2%}")
        # print(f"{'    F-score':30}: {fscore:68.2%}")
        # print(f"{'    Accuracy':30}: {accuracy:68.2%}")

        return
    
    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                print(batch_x.shape)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, flag):
        if flag == "MSE":
            criterion = nn.MSELoss(reduction="mean")
        elif flag == "CEL":
            criterion = nn.CrossEntropyLoss(reduction="mean")
        elif flag == "MLM":
            criterion = nn.MultiLabelMarginLoss(reduction='mean')
        return criterion

    def _predict(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder

        def _run_model():
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if self.args.output_attention:
                outputs = outputs[0]
            return outputs

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = _run_model()
        else:
            outputs = _run_model()

        f_dim = -1 if self.args.features == 'SYSCALL' else 0
        outputs = outputs[:, -self.args.pred_len:, :]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        # outputs_ = torch.max(outputs, dim=-1)[1].unsqueeze(dim=-1)
        batch_y = batch_y.squeeze(dim=-1).to(torch.int64)
        batch_y_ = F.one_hot(batch_y, num_classes=self.args.c_out).float()

        # print("outputs.shape:", outputs.shape)
        # print("batch_y_.shape:", batch_y_.shape)

        return outputs, batch_y_
