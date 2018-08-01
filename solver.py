import os
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from model import model_parser
from tensorboardX import SummaryWriter


class Solver():
    def __init__(self, data_loader, config):
        self.data_loader = data_loader
        self.config = config
        self.model = model_parser(self.config.model, self.config.fixed_weight)

        self.print_network(self.model, self.config.model)
        self.model_save_path = 'models_%s' % self.config.model
        self.summary_save_path = 'summary_%s' % self.config.model

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
            self.summary_save_path = 'summaries/%s/summary_%d' % (self.config.sequential_mode, int(self.config.fixed_weight))
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

    def train(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=50)

        num_epochs = self.config.num_epochs

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # Setup for tensorboard
        use_tensorboard = self.config.use_tensorboard
        if use_tensorboard:
            if not os.path.exists(self.summary_save_path):
                os.makedirs(self.summary_save_path)
            writer = SummaryWriter(log_dir=self.summary_save_path)

        since = time.time()
        n_iter = 0

        # For pretrained network
        start_epoch = 1
        if self.config.pretrained_model:
            start_epoch = int(self.config.pretrained_model)

        # Pre-define variables to get the best model
        best_train_loss = 10000
        best_val_loss = 10000
        best_train_model = None
        best_val_model = None

        for epoch in range(start_epoch, num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs-1))
            print('-'*20)

            error_train = []
            error_val = []

            for phase in ['train', 'val']:

                if phase == 'train':
                    scheduler.step()
                    self.model.train()
                else:
                    self.model.eval()

                data_loader = self.data_loader[phase]

                for i, (inputs, poses) in enumerate(data_loader):

                    inputs = inputs.to(device)
                    poses = poses.to(device)

                    # Zero the parameter gradient
                    optimizer.zero_grad()

                    # forward
                    pos_out, ori_out = self.model(inputs)

                    pos_true = poses[:, :3]
                    ori_true = poses[:, 3:]

                    beta = self.config.beta
                    ori_out = F.normalize(ori_out, p=2, dim=1)
                    ori_true = F.normalize(ori_true, p=2, dim=1)

                    loss_pos = F.mse_loss(pos_out, pos_true)
                    loss_ori = F.mse_loss(ori_out, ori_true)

                    loss = loss_pos + beta * loss_ori

                    loss_print = loss.item()
                    loss_ori_print = loss_ori.item()
                    loss_pos_print = loss_pos.item()

                    if use_tensorboard:
                        if phase == 'train':
                            error_train.append(loss_print)
                            writer.add_scalar('loss/overall_loss', loss_print, n_iter)
                            writer.add_scalar('loss/position_loss', loss_pos_print, n_iter)
                            writer.add_scalar('loss/rotation_loss', loss_ori_print, n_iter)
                        elif phase == 'val':
                            error_val.append(loss_print)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        n_iter += 1

                    print('{}th {} Loss: total loss {:.3f} / pos loss {:.3f} / ori loss {:.3f}'.format(i, phase, loss_print, loss_pos_print, loss_ori_print))

            # For each epoch
            error_train = sum(error_train) / len(error_train)
            error_val = sum(error_val) / len(error_val)

            if phase == 'train':
                if (epoch+1) % self.config.model_save_step == 0:
                    if error_train < best_train_loss:
                        best_train_loss = error_train
                        best_train_model = epoch
                    if error_val < best_val_loss:
                        best_val_loss = error_val
                        best_val_model = epoch

                    save_filename = self.model_save_path + '/%s_net.pth' % epoch
                    # save_path = os.path.join('models', save_filename)
                    torch.save(self.model.cpu().state_dict(), save_filename)
                    if torch.cuda.is_available():
                        self.model.to(device)

            print('Train and Validaion error {} / {}'.format(error_train, error_val))
            print('=' * 40)
            print('=' * 40)

            if use_tensorboard:
                writer.add_scalars('loss/trainval', {'train':error_train,
                                                     'val':error_val}, epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        if self.config.sequential_mode:
            f = open(self.summary_save_path + '/train.csv', 'w')

            f.write('{},{}\n{},{}'.format(best_train_loss, best_train_model,
                                          best_val_loss, best_val_model))
            f.close()
            # return (best_train_loss, best_train_model), (best_val_loss, best_val_model)

    def test(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(device)

        print('Load pretrained model!')
        self.model.load_state_dict(torch.load(self.model_save_path + '/175_net.pth'))

        total_pos_loss = 0
        total_ori_loss = 0
        pos_loss_arr = []
        ori_loss_arr = []

        num_data = len(self.data_loader)

        for i, (inputs, poses) in enumerate(self.data_loader):
            print(i)

            inputs = inputs.to(device)
            poses = poses.to(device)

            # forward
            pos_out, ori_out = self.model(inputs)

            pos_true = poses[:, :3]
            ori_true = poses[:, 3:]

            ori_out = F.normalize(ori_out, p=2, dim=1)
            ori_true = F.normalize(ori_true, p=2, dim=1)

            loss_pos_print = F.pairwise_distance(pos_out, pos_true, p=2).item()
            loss_ori_print = F.pairwise_distance(ori_out, ori_true, p=2).item()

            total_pos_loss += loss_pos_print
            total_ori_loss += loss_ori_print

            pos_loss_arr.append(loss_pos_print)
            ori_loss_arr.append(loss_ori_print)

            print('{}th Error: pos error {:.3f} / ori error {:.3f}'.format(i, loss_pos_print, loss_ori_print))

        position_error = sum(pos_loss_arr)/len(pos_loss_arr)
        rotation_error = sum(ori_loss_arr)/len(ori_loss_arr)
        print('=' * 20)
        print('Overall pose errer {:.3f} / {:.3f}'.format(position_error, rotation_error))

        if self.config.sequential_mode:
            f = open(self.summary_save_path + '/test.csv', 'w')
            f.write('{},{}'.format(position_error, rotation_error))
            f.close()
            # return position_error, rotation_error

