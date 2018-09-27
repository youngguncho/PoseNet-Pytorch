import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from model import model_parser
from model import PoseLoss
from pose_utils import *
from itertools import islice


def get_query_items(data_loader, query_list):
    data_size = len(query_list)
    images = torch.zeros((data_size, 3, 224, 224))
    poses = torch.zeros((data_size, 7))

    for i, idx in enumerate(query_list):
        (image, pose, _, _) = data_loader.dataset.__getitem__(idx)
        images[i, :] = image
        poses[i, :] = pose

    return images, poses


class Solver():
    def __init__(self, data_loader, config):
        self.data_loader = data_loader
        self.config = config

        # do not use dropout if not bayesian mode
        # if not self.config.bayesian:
        #     self.config.dropout_rate = 0.0

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = model_parser(self.config.model, self.config.fixed_weight, self.config.dropout_rate,
                                  self.config.bayesian)

        self.criterion = PoseLoss(self.device, self.config.sx, self.config.sq, self.config.learn_beta)

        self.print_network(self.model, self.config.model)
        self.data_name = self.config.image_path.split('/')[-1]
        # self.data_name = 'NCLT_cam4_2seqs_3m'
        self.model_save_path = 'models_%s' % self.data_name
        self.summary_save_path = 'summary_%s' % self.data_name

        if self.config.pretrained_model:
            self.load_pretrained_model()

        if self.config.sequential_mode:
            self.set_sequential_mode()

    # Inner Functions #
    def set_sequential_mode(self):
        if self.config.sequential_mode == 'model':
            self.model_save_path = 'models/%s/models_%s' % (self.config.sequential_mode, self.config.model)
            self.summary_save_path = 'summaries/%s/summary_%s' % (self.config.sequential_mode, self.config.model)
        elif self.config.sequential_mode == 'fixed_weight':
            self.model_save_path = 'models/%s/models_%d' % (self.config.sequential_mode, int(self.config.fixed_weight))
            self.summary_save_path = 'summaries/%s/summary_%d' % (
            self.config.sequential_mode, int(self.config.fixed_weight))
        elif self.config.sequential_mode == 'batch_size':
            self.model_save_path = 'models/%s/models_%d' % (self.config.sequential_mode, self.config.batch_size)
            self.summary_save_path = 'summaries/%s/summary_%d' % (self.config.sequential_mode, self.config.batch_size)
        elif self.config.sequential_mode == 'learning_rate':
            self.model_save_path = 'models/%s/models_%f' % (self.config.sequential_mode, self.config.lr)
            self.summary_save_path = 'summaries/%s/summary_%f' % (self.config.sequential_mode, self.config.lr)
        elif self.config.sequential_mode == 'beta':
            self.model_save_path = 'models/%s/models_%d' % (self.config.sequential_mode, self.config.beta)
            self.summary_save_path = 'summaries/%s/summary_%d' % (self.config.sequential_mode, self.config.beta)
        else:
            assert 'Unvalid sequential mode'

    def load_pretrained_model(self):
        model_path = self.model_save_path + '/%s_net.pth' % self.config.pretrained_model
        self.model.load_state_dict(torch.load(model_path))
        print('Load pretrained network: ', model_path)

    def print_network(self, model, name):
        num_params = 0
        for param in model.parameters():
            num_params += param.numel()

        print('*' * 20)
        print(name)
        print(model)
        print('*' * 20)

    def loss_func(self, input, target):
        diff = torch.norm(input - target, dim=1)
        diff = torch.mean(diff)
        return diff

    def calc_negative_distances(self, feat_out, pos_true):
        batch_size = feat_out.size(0)

        query_idx = [i for i in range(batch_size)]

        pair_list = []
        for idx in query_idx:
            pn_list = []
            neg_list = np.array([n for n in range(batch_size) if n != idx])

            pos_anchor = pos_true[idx, :]
            pos_neg = pos_true[neg_list, :]
            pos_diff = F.pairwise_distance(pos_anchor, pos_neg)

            pos_diff = pos_diff.cpu().data.numpy()

            # To discard near node to the anchor node
            neg_list = neg_list[np.where(pos_diff > 10)]  # 앵커와 10m 이내에 있는 neg set들의 인덱스 리스트
            # print("len(false_list)", false_list.size)
            #
            # if false_list.size > 0:
            #
            #     print("false_list", false_list)
            #     filt_neg_list = []
            #
            #     for k in range(len(neg_list)):
            #         print("k", k)
            #         if k in false_list:
            #             continue
            #
            #         filt_neg_list.append(neg_list[k])
            #
            #     neg_list = filt_neg_list

            feat_anchor = feat_out[idx, :]
            feat_neg = feat_out[neg_list, :]

            neg_dist = F.pairwise_distance(feat_anchor, feat_neg)

            min_dist, min_idx = torch.min(neg_dist.unsqueeze(0), dim=1)

            pair_list.append([neg_list[min_idx], min_dist])

        return pair_list

    def evaluate(self):
        f = open(self.summary_save_path + '/test_result.csv', 'w')

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)
        self.model.eval()

        if self.config.test_model is None:
            test_model_path = self.model_save_path + '/best_net.pth'
        else:
            test_model_path = self.model_save_path + '/{}_net.pth'.format(self.config.test_model)

        print('Load pretrained model: ', test_model_path)
        self.model.load_state_dict(torch.load(test_model_path))

        total_pos_loss = 0
        total_ori_loss = 0
        pos_loss_arr = []
        ori_loss_arr = []
        true_pose_list = []
        estim_pose_list = []
        estim_feat_list = []
        if self.config.bayesian:
            pred_mean = []
            pred_var = []

        num_data = len(self.data_loader)

        for i, (inputs, poses) in enumerate(self.data_loader):
            print(i)

            inputs = inputs.to(self.device)

            # forward
            if self.config.bayesian:
                num_bayesian_test = 100
                pos_array = torch.Tensor(num_bayesian_test, 3)
                ori_array = torch.Tensor(num_bayesian_test, 4)

                for i in range(num_bayesian_test):
                    pos_single, ori_single, _ = self.model(inputs)
                    pos_array[i, :] = pos_single
                    ori_array[i, :] = F.normalize(ori_single, p=2, dim=1)

                pose_quat = torch.cat((pos_array, ori_array), 1).detach().cpu().numpy()
                pred_pose, pred_var = fit_gaussian(pose_quat)

                pos_var = np.sum(pred_var[:3])
                ori_var = np.sum(pred_var[3:])

                pos_out = pred_pose[:3]
                ori_out = pred_pose[3:]
            else:
                pos_out, ori_out, feat_out = self.model(inputs)
                pos_out = pos_out.squeeze(0).detach().cpu().numpy()
                ori_out = F.normalize(ori_out, p=2, dim=1)
                ori_out = quat_to_euler(ori_out.squeeze(0).detach().cpu().numpy())
                print('pos out', pos_out)
                print('ori_out', ori_out)

            pos_true = poses[:, :3].squeeze(0).numpy()
            ori_true = poses[:, 3:].squeeze(0).numpy()

            ori_true = quat_to_euler(ori_true)
            print('pos true', pos_true)
            print('ori true', ori_true)
            loss_pos_print = array_dist(pos_out, pos_true)
            loss_ori_print = array_dist(ori_out, ori_true)

            true_pose_list.append(np.hstack((pos_true, ori_true)))
            estim_feat_list.append(feat_out.squeeze(0).detach().cpu().numpy())

            if loss_pos_print < 20:
                estim_pose_list.append(np.hstack((pos_out, ori_out)))

            # ori_out = F.normalize(ori_out, p=2, dim=1)
            # ori_true = F.normalize(ori_true, p=2, dim=1)
            #
            # loss_pos_print = F.pairwise_distance(pos_out, pos_true, p=2).item()
            # loss_ori_print = F.pairwise_distance(ori_out, ori_true, p=2).item()

            # loss_pos_print = F.l1_loss(pos_out, pos_true).item()
            # loss_ori_print = F.l1_loss(ori_out, ori_true).item()

            # loss_pos_print = self.loss_func(pos_out, pos_true).item()
            # loss_ori_print = self.loss_func(ori_out, ori_true).item()

            print(pos_out)
            print(pos_true)

            total_pos_loss += loss_pos_print
            total_ori_loss += loss_ori_print

            pos_loss_arr.append(loss_pos_print)
            ori_loss_arr.append(loss_ori_print)

            if self.config.bayesian:
                print('{}th Error: pos error {:.3f} / ori error {:.3f}'.format(i, loss_pos_print, loss_ori_print))
                print('{}th std: pos / ori', pos_var, ori_var)
                f.write('{},{},{},{}\n'.format(loss_pos_print, loss_ori_print, pos_var, ori_var))

            else:
                print('{}th Error: pos error {:.3f} / ori error {:.3f}'.format(i, loss_pos_print, loss_ori_print))


        # position_error = sum(pos_loss_arr)/len(pos_loss_arr)
        # rotation_error = sum(ori_loss_arr)/len(ori_loss_arr)
        position_error = np.median(pos_loss_arr)
        rotation_error = np.median(ori_loss_arr)

        print('=' * 20)
        print('Overall median pose errer {:.3f} / {:.3f}'.format(position_error, rotation_error))
        print('Overall average pose errer {:.3f} / {:.3f}'.format(np.mean(pos_loss_arr), np.mean(ori_loss_arr)))
        f.close()

        if self.config.save_result:
            f_true = self.summary_save_path + '/pose_true.csv'
            f_estim = self.summary_save_path + '/pose_estim.csv'
            f_feat = self.summary_save_path + '/feat_estim.csv'
            np.savetxt(f_true, true_pose_list, delimiter=',')
            np.savetxt(f_estim, estim_pose_list, delimiter=',')
            np.savetxt(f_feat, estim_feat_list, delimiter=',', fmt='%.4f')

        if self.config.sequential_mode:
            f = open(self.summary_save_path + '/test.csv', 'w')
            f.write('{},{}'.format(position_error, rotation_error))
            f.close()
            # return position_error, rotation_error

