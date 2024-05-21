import argparse
import copy
import json
import logging
import os
import pickle
import sys
import time
from collections import OrderedDict  
from copy import deepcopy

import torch
import numpy as np
from argparse import ArgumentParser
import src.approach.our_ot as ot
from src.loggers.exp_logger import ExperimentLogger
from src.datasets.exemplars_dataset import ExemplarsDataset
from src.networks.lenet import LeNetArch
from src.networks.network import LLL_Net
from utils.utils import construct_log, print_args


def get_config():
    parser1 = argparse.ArgumentParser()
    parser1.add_argument('--num_epoch', default=1, type=int,
                         help='number of epochs (default 48)')
    parser1.add_argument('--batch_size', default=32, type=int,
                         help='size of each mini batch (default 50)')
    parser1.add_argument('--learning_rate', default=0.01, type=float,
                         help='learning rate (default 0.000075)')
    parser1.add_argument('--epoch_per_averaging', default=1, type=int,
                         help='number of batches per averaging (default 1)')
    parser1.add_argument('--model_name', default='simplemnistnet', type=str,
                         help='the name of the model (default simplemnistnet)')
    parser1.add_argument('--device', default='cpu', type=str,
                         help='the device to train the model, either cpu or cuda (default cpu)')
    parser1.add_argument('--dataset', default='cifar10', type=str,
                         help='the dataset to train the model (default mnist)')
    parser1.add_argument('--need_customized_dataset', default=False, type=bool,
                         help='whether or not you need customized dataset (default: False)')
    parser1.add_argument('--iid', default=True, type=bool,
                         help='whether or not you need iid dataset or not (default: True)')
    parser1.add_argument('--to_download', default=True, type=bool,
                         help='whether or not the dataset needs to be downloaded (default: True)')
    parser1.add_argument('--ensemble_step', default=0.7, type=float,
                         help='the ensemble weight used in gm-based fusion (default: 0.5)')
    parser1.add_argument('--training_mode', default='ot', type=str,
                         help='whether to use traditional averaging or fusion-based averaging, \
                            can be traditional or fusion or fusion_slice or ot (default: traditional)')
    args1 = parser1.parse_args()
    if args1.device not in ['cpu', 'cuda']:
        raise NotImplementedError
    args1.device = torch.device('cpu') if args1.device == 'cpu' else torch.device('cuda')

    args1.batch_size_train = args1.batch_size
    args1.batch_size_test = args1.batch_size
    args1.act_num_samples = 200
    args1.clip_gm = False
    args1.clip_max = 5
    args1.clip_min = 0
    args1.clipgrad = 10000
    args1.lr_patience = 5
    args1.lr_min = 3e-4
    args1.lr_factor = 3
    args1.multi_softmax = False
    args1.fix_bn = False
    args1.eval_on_train = False
    args1.correction = True
    args1.dataset = "cifar10"
    args1.debug = False
    args1.dist_normalize = True
    args1.ensemble_step = 0.6
    args1.eval_aligned = False
    args1.exact = 2
    args1.geom_ensemble_type = "wts"
    args1.gpu_id = -1
    args1.ground_metric = "euclidean"
    args1.ground_metric_eff = False
    args1.ground_metric_normalize = "log"
    args1.importance = None
    args1.normalize_wts = False
    args1.num_models = 2
    args1.not_squared = True
    args1.past_correction = True
    args1.prediction_wts = True
    args1.proper_marginals = False
    args1.reg = 0.01
    args1.skip_last_layer = False
    args1.softmax_temperature = 1
    args1.unbalanced = False
    args1.weight = [0.5, 0.5]
    args1.width_ratio = 1
    args1.max = True

    return args1


class Inc_Learning_Appr:
    def __init__(self, args, model, device, nepochs=1, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger: ExperimentLogger = None, exemplars_dataset: ExemplarsDataset = None):
        self.model = model
        self.model2 = LeNetArch()
        self.model2 = LLL_Net(self.model2, True)
        self.model2.add_head(2)
        self.device = device
        self.nepochs = nepochs
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.momentum = momentum
        self.wd = wd
        self.multi_softmax = multi_softmax
        self.logger = logger
        self.exemplars_dataset = exemplars_dataset
        self.warmup_epochs = wu_nepochs
        self.warmup_lr = lr * wu_lr_factor
        self.warmup_loss = torch.nn.CrossEntropyLoss()
        self.fix_bn = fix_bn
        self.eval_on_train = eval_on_train
        self.tt = 0
        self.optimizer = None
        self.config = args
        self.fisher0 = {}
        self.fisher1 = {}
        self.old_model = None

        self.output_dir = args.output if args.output else './output'
        os.makedirs(self.output_dir, exist_ok=True)
        construct_log(os.path.join(self.output_dir, 'train.log'), only_file=args.ssh)
        self._logger = logging.getLogger('train')
        self._logger.info(sys.argv)
        print_args(args, self._logger)

        args_save = copy.deepcopy(args)
        del args_save.device
        with open(os.path.join(self.output_dir, 'args.json'), "w") as f:
            json.dump(args_save.__dict__, f, indent=2)
        self.pickle_log = {'train': {}, 'eval': {}, 'test': {}, 'eval_noema': {}}

    @staticmethod
    def extra_parser(args):
        parser = ArgumentParser()
        parser.add_argument('--lamb', default=1, type=float, required=False,
                            help='Forgetting-intransigence trade-off (default=%(default)s)')
        # Define how old and new fisher is fused, by default it is a 50-50 fusion
        parser.add_argument('--alpha', default=0.5, type=float, required=False,
                            help='EWC alpha (default=%(default)s)')
        parser.add_argument('--fi-sampling-type', default='max_pred', type=str, required=False,
                            choices=['true', 'max_pred', 'multinomial'],
                            help='Sampling type for Fisher information (default=%(default)s)')
        parser.add_argument('--fi-num-samples', default=-1, type=int, required=False,
                            help='Number of samples for Fisher information (-1: all available) (default=%(default)s)')
        return parser.parse_known_args(args)

    @staticmethod
    def exemplars_dataset_class():
        return None

    def _get_optimizer(self, model):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and len(model.heads) > 1:
            params = list(model.model.parameters()) + list(model.heads[-1].parameters())
        else:
            params = model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)


    def train(self, t, trn_loader, val_loader):
        self.pre_train_process(t, trn_loader)
        self.train_loop(t, trn_loader, val_loader)
        self.post_train_process(t, trn_loader)
        self.model2.add_head(2)
        self.model2.to(self.config.device)

    def pre_train_process(self, t, trn_loader):
        if self.warmup_epochs and t > 0:
            self.optimizer = torch.optim.SGD(self.model.heads[-1].parameters(), lr=self.warmup_lr)
            for e in range(self.warmup_epochs):
                warmupclock0 = time.time()
                self.model.heads[-1].train()
                for images, targets in trn_loader:
                    outputs = self.model(images.to(self.device))
                    loss = self.warmup_loss(outputs[t], targets.to(self.device) - self.model.task_offset[t])
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.heads[-1].parameters(), self.clipgrad)
                    self.optimizer.step()
                warmupclock1 = time.time()
                with torch.no_grad():
                    total_loss, total_acc_taw = 0, 0
                    self.model.eval()
                    for images, targets in trn_loader:
                        outputs = self.model(images.to(self.device))
                        loss = self.warmup_loss(outputs[t], targets.to(self.device) - self.model.task_offset[t])
                        pred = torch.zeros_like(targets.to(self.device))
                        for m in range(len(pred)):
                            this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
                            pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
                        hits_taw = (pred == targets.to(self.device)).float()
                        total_loss += loss.item() * len(targets)
                        total_acc_taw += hits_taw.sum().item()
                total_num = len(trn_loader.dataset.labels)
                trn_loss, trn_acc = total_loss / total_num, total_acc_taw / total_num
                warmupclock2 = time.time()
                print('| Warm-up Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, warmupclock1 - warmupclock0, warmupclock2 - warmupclock1, trn_loss, 100 * trn_acc))
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=trn_loss, group="warmup")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * trn_acc, group="warmup")

    def train_loop(self, t, trn_loader, val_loader):
        lr = self.lr
        best_loss = np.inf
        patience = self.lr_patience
        target_model = self.model if t == 0 else self.model2
        best_model = target_model.get_copy()
        self.optimizer = self._get_optimizer(target_model)

        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(t, trn_loader, target_model)
            clock1 = time.time()
            if self.eval_on_train:
                train_loss, train_acc, _ = self.eval(t, trn_loader, target_model)
                self.pickle_log['eval'][j] = dict(result)

                with open(os.path.join(self.output_dir, "log.pkl"), "wb") as f:
                    pickle.dump(self.pickle_log, f)
                clock2 = time.time()
                print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
            else:
                print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

            clock3 = time.time()
            valid_loss, valid_acc, _,result = self.eval(t, val_loader, target_model)
            clock4 = time.time()

            print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = target_model.get_copy()
                patience = self.lr_patience
                print(' *', end='')
            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        print()
                        break
                    patience = self.lr_patience
                    self.optimizer.param_groups[0]['lr'] = lr
                    target_model.set_state_dict(best_model)
            self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
            self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
            print()

        target_model.set_state_dict(best_model)
        if t == 0:
            self.model2.set_state_dict(best_model)

    def _federated_averaging_ewc(self):
        parameters, fishers = ot.get_wassersteinized_layers_modularized_ewc(
            args=self.config, networks=[self.model, self.model2], fishers=[self.fisher0, self.fisher1])
        idx = 0
        for n, p in self.fisher0.items():
            self.fisher0[n] = fishers[idx]
            idx = idx + 1
        state_dict_1 = {}
        state_dict_2 = {}
        for n, p in self.model.model.named_parameters():
            state_dict_1[n] = p
        for n, p in self.model2.model.named_parameters():
            state_dict_2[n] = p
        for n, p in self.model.heads.named_parameters():
            state_dict_1[n] = p
        for n, p in self.model2.heads.named_parameters():
            state_dict_2[n] = p
        state_dict_11 = {}
        state_dict_22 = {}
        for idx, (name, _) in enumerate(state_dict_1.items()):
            # print(name)
            if idx >= (len(state_dict_1) - len(self.model.heads)):
                name1 = 'heads.' + str(idx - (len(state_dict_1) - len(self.model.heads))) + '.' + 'weight'
                # print('change',name1)
                if idx != (len(state_dict_1) - 1):
                    if idx == (len(state_dict_1) - len(self.model.heads)):
                        state_dict_11[name1] = state_dict_1[name]
                    else:
                        state_dict_11[name1] = state_dict_1[name]
                else:
                    name2 = '0.weight'
                    state_dict_11[name1] = state_dict_2[name2]

            else:
                name1 = 'model.' + name
                state_dict_11[name1] = parameters[idx]
        self.model.load_state_dict(state_dict_11)
        self.model2.load_state_dict(state_dict_11)

    def _federated_averaging_ot(self):
        parameters, loss = ot.get_wassersteinized_layers_modularized(
            args=self.config, networks=[self.model, self.model2])
        state_dict_1 = {}
        state_dict_2 = {}
        for n, p in self.model.model.named_parameters():
            state_dict_1[n] = p
        for n, p in self.model2.model.named_parameters():
            state_dict_2[n] = p
        for n, p in self.model.heads.named_parameters():
            state_dict_1[n] = p
        for n, p in self.model2.heads.named_parameters():
            state_dict_2[n] = p
        state_dict_11 = {}
        state_dict_22 = {}
        for idx, (name, _) in enumerate(state_dict_1.items()):
            if idx >= (len(state_dict_1) - len(self.model.heads)):
                name1 = 'heads.' + str(idx - (len(state_dict_1) - len(self.model.heads))) + '.' + 'weight'
                if idx != (len(state_dict_1) - 1):
                    if idx == (len(state_dict_1) - len(self.model.heads)):
                        state_dict_11[name1] = state_dict_1[name]
                    else:
                        state_dict_11[name1] = state_dict_1[name]
                else:
                    state_dict_11[name1] = state_dict_2[name]

            else:
                name1 = 'model.' + name
                state_dict_11[name1] = parameters[idx]

        self.model.load_state_dict(state_dict_11)
        self.model2.load_state_dict(state_dict_11)

    def _federated_averaging_traditional(self):
        state_dict_1 = self.model.state_dict()
        state_dict_2 = self.model2.state_dict()
        state_dict_11 = {}
        for idx, (name, _) in enumerate(state_dict_1.items()):
            if idx >= len(state_dict_1) - len(self.model.heads):
                name1 = 'heads.' + str(idx - (len(state_dict_1) - len(self.model.heads))) + '.' + 'weight'
                if idx != (len(state_dict_1) - 1):
                    state_dict_11[name1] = state_dict_1[name]
                else:
                    name2 = 'heads.0.weight'
                    state_dict_11[name1] = state_dict_2[name2]
            else:
                name1 = name
                state_dict_11[name1] = (state_dict_1[name] + state_dict_2[name]) / 2

        self.model.load_state_dict(state_dict_11)
        self.model2.load_state_dict(state_dict_11)

    def _federated_averaging_ot_test(self):
        parameters, loss = ot.get_wassersteinized_layers_modularized(
            args=self.config, networks=[self.model, self.model2])

        return loss

    def post_train_process(self, t, trn_loader):

        if t != 0:
            if self.config.training_mode == 'ot':
                self.loss0 = self._federated_averaging_ot()
            elif self.config.training_mode == 'traditional':
                self._federated_averaging_traditional()

        self.old_model = deepcopy(self.model)
        self.old_model.freeze_all()

    def train_epoch(self, t, trn_loader, model):
        model.train()
        if self.fix_bn and t > 0:
            model.freeze_bn()
        for images, targets in trn_loader:
            outputs_old = None
            if t > 0:
                outputs_old = self.old_model(images.to(self.device))
            outputs = model(images.to(self.device))
            loss = self.criterion(t, outputs, targets.to(self.device), outputs_old)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipgrad)
            self.optimizer.step()

    def train_epoch1(self, t, trn_loader, model):
        model.train()
        if self.fix_bn and t > 0:
            model.freeze_bn()
        for images, targets in trn_loader:
            outputs = model(images.to(self.device))
            outputs = torch.cat(outputs, dim=1)
            loss = self.criterion1(t, outputs, targets.to(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipgrad)
            self.optimizer.step()

    def eval(self, t, val_loader, model):
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            model.eval()
            for images, targets in val_loader:
                outputs = model(images.to(self.device))
                loss = self.criterion1(t, outputs, targets.to(self.device))
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        loss = (total_loss / total_num)
        Taw = (total_acc_taw / total_num)
        Tag = (total_acc_tag / total_num)
        log_name = 'Validate_eval'
        self._logger.info('Log_name:{} '
                          'loss: {loss:#.4g};  '
                          'Taw: {Taw:#.4g}; '
                          'Tag: {Tag:#.4g}; '.format(
            log_name,  loss=(total_loss / total_num), Taw=(total_acc_taw / total_num),Tag=( total_acc_tag / total_num))
        )

        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num, OrderedDict([ ('loss', loss), ('Taw', Taw), ('Tag', Tag)])

    def eval1(self, t, val_loader):
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward current model
                outputs = self.model(images.to(self.device))  # output为一个list list中的shape为[64,2]
                loss = self.criterion(t, outputs, targets.to(self.device))
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def calculate_metrics(self, outputs, targets):
        pred = torch.zeros_like(targets.to(self.device))
        for m in range(len(pred)):
            this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
            pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
        hits_taw = (pred == targets.to(self.device)).float()
        if self.multi_softmax:
            outputs = [torch.nn.functional.log_softmax(output, dim=1) for output in outputs]
            pred = torch.cat(outputs, dim=1).argmax(1)
        else:
            pred = torch.cat(outputs, dim=1).argmax(1)
        hits_tag = (pred == targets.to(self.device)).float()
        return hits_taw, hits_tag

    def cross_entropy(self, outputs, targets, exp=1.0, size_average=True, eps=1e-5):
        out = torch.nn.functional.softmax(outputs, dim=1)
        tar = torch.nn.functional.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        if size_average:
            ce = ce.mean()
        return ce

    def criterion(self, t, outputs, targets, outputs_old=None):
        loss = 0
        if t > 0:
            self.loss0 = self._federated_averaging_ot_test()
            loss += self.loss0
        return loss + torch.nn.functional.cross_entropy(outputs[t], targets.long() - self.model.task_offset[t])
