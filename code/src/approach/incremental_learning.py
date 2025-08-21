import argparse
import logging
import os
import pickle
import time
from collections import OrderedDict
import importlib
import torch
import numpy as np
from copy import deepcopy  
from argparse import ArgumentParser
import src.approach.our_ot as ot
from src.loggers.exp_logger import ExperimentLogger
from src.datasets.exemplars_dataset import ExemplarsDataset
from src.networks.lenet import LeNetArch
from src.networks.vggnet import VggNet
from src.networks import tvmodels, allmodels, set_tvmodel_head_var
# from src.networks.vggnet import VggNet
from src.networks.network import LLL_Net
from src.approach.our_ot import fuse_bn_recursively

def get_activation(model, input_data):
    activations = []
    def hook(module, input, output):
        activations.append(output)
    target_layer = model.model.relu
    hook_handle = target_layer.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(input_data)
    hook_handle.remove()
    return activations[0] if activations else None

def get_activation1(model, input_data, layer_num):
    activations = []
    def hook(module, input, output):
        activations.append(output)
    target_layer = model.model.layer4[layer_num]
    hook_handle = target_layer.register_forward_hook(hook)
    with torch.no_grad():
        _ = model(input_data)
    hook_handle.remove()
    return activations[0] if activations else None 


def get_config():
    '''
    get the configurations from commandline with parser
    '''
    parser1 = argparse.ArgumentParser()
    parser1.add_argument('--device', default='cuda', type=str,
                         help='the device to train the model, either cpu or cuda (default cpu)')
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

    args1.clip_gm = False
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
    args1.ground_metric = "cosine"
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
    """Basic class for implementing incremental learning approaches"""

    def __init__(self, model, device, args, logger: ExperimentLogger = None,
                exemplars_dataset: ExemplarsDataset = None):
        # Khởi tạo self.model với LLL_Net
        self.model = LLL_Net(model, remove_last_layer=True)
        self.model.heads = torch.nn.ModuleList()
        self.model.add_head(10)  # Head cho task đầu tiên (CIFAR-10)
        self.model.to(device)

        # Khởi tạo self.model2
        net1 = getattr(importlib.import_module(name='src.networks'), args.network)
        init_model = net1()
        self.model2 = LLL_Net(init_model, remove_last_layer=True)
        self.model2.heads = torch.nn.ModuleList()
        self.model2.add_head(10)
        self.model2.to(device)

        # Khởi tạo trọng số Conv2d (tùy chọn, nếu cần)
        for m in self.model2.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        self.device = device
        self.nepochs = args.nepochs
        self.lr = args.learning_rate
        self.lr_min = args.lr_min
        self.lr_factor = args.lr_factor
        self.lr_patience = args.lr_patience
        self.clipgrad = args.clipping
        self.momentum = args.momentum
        self.wd = args.wd
        self.multi_softmax = args.multi_softmax
        self.logger = logger
        self.exemplars_dataset = exemplars_dataset
        self.warmup_epochs = 0
        self.warmup_lr = self.lr * args.warmup_lr_factor
        self.warmup_loss = torch.nn.CrossEntropyLoss()
        self.fix_bn = args.fix_bn
        self.eval_on_train = args.eval_on_train
        self.tt = 0
        self.optimizer = None
        self.config = args
        self.fisher0 = {}
        self.fisher1 = {}
        self.old_model = None
        self.old_model1 = None
        self._logger = logging.getLogger('train')
        self.al = args.al
        self.decay_mile_stone = [80, 120]
        self.lr_decay = 0.1
        

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        return parser.parse_known_args(args)

    @staticmethod
    def exemplars_dataset_class():
        """Returns a exemplar dataset to use during the training if the approach needs it
        :return: ExemplarDataset class or None
        """
        return None

    def _get_optimizer(self, model):
        """Returns the optimizer"""
        if len(model.heads) > 1:
            # if there are no exemplars, previous heads are not modified
            params = list(model.model.parameters()) + list(model.heads[-1].parameters())
        else:
            params = model.parameters()
        # return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)
        return torch.optim.Adam(params, lr=self.lr)

    def train(self, t, trn_loader, val_loader):
        """Main train structure"""
        
        # self.old_model = deepcopy(self.model)
        # self.model.freeze_all()
        # self.old_model.freeze_all()
        self.pre_train_process(t, trn_loader)
        self.train_loop(t, trn_loader, val_loader)
        self.post_train_process(t, trn_loader)
        self.model2.add_head(10)
        self.model2.to(self.device)
        # for m in self.model2.modules():
        #     # print('m',m)
        #     if isinstance(m, torch.nn.Conv2d):
        #         torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""

        # Warm-up phase
        if self.warmup_epochs and t > 0:
            self.optimizer = torch.optim.SGD(self.model.heads[-1].parameters(), lr=self.warmup_lr)
            # Loop epochs -- train warm-up head
            for e in range(self.warmup_epochs):
                warmupclock0 = time.time()
                self.model.heads[-1].train()  # 训练
                for images, targets in trn_loader:
                    features = self.model(images.to(self.device), return_features=True)
                    outputs = self.model.heads[t](features)
                    loss = self.warmup_loss(outputs, targets.to(self.device) - self.model.task_offset[t])
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
                # self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=trn_loss, group="warmup")
                # self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * trn_acc, group="warmup")

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""
        lr = self.lr
        best_loss = np.inf
        best_acc = 0
        patience = self.lr_patience
        target_model =  self.model2
        self.optimizer = self._get_optimizer(self.model2)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.decay_mile_stone, gamma=self.lr_decay)
        # log_name = 'former'
        # for n, p in target_model.model.named_parameters():
        #     self._logger.info(
        #                         'logger: {};'
        #                         'task: {task:};'
        #                         'weigths: {weights:};  '.format(
        #             log_name, task=t, weights=p)
        #         )

        
        best_model = self.model2.get_copy()
        # best_model1 = target_model.get_copy()
        idxs = 0
        # for n, p in target_model.model.named_parameters():
        #     if idxs == 0:
        #         print(p)
        #     idxs = idxs+1
        # self.optimizer = self._get_optimizer(self.model2)
        # print(best_model.named_parameters())
        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(t, trn_loader, self.model2)
            clock1 = time.time()
            if self.eval_on_train:
                train_loss, train_acc, _, result = self.eval(t, trn_loader, self.model2)
                clock2 = time.time()
                print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
            else:
                print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc, _, result = self.eval(t, val_loader, self.model2)
            clock4 = time.time()
            scheduler.step()
            print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')
            print()
        best_model = self.model2.get_copy()
        self.model2.set_state_dict(best_model)
        print('='*200)

        if t == 0:
            self.model.set_state_dict(best_model)
        return 

    def _federated_averaging_ewc(self):
        '''
        average the parameters of two models based on ot_based_fusion method
        '''
        # the following code can be replaced by other functions
        parameters, fisherss = ot.get_wassersteinized_layers_modularized_ewc(
            args=self.config, device=self.device, networks=[self.model, self.model2],
            fishers=[self.fisher0, self.fisher1])
        
        state_dict_1 = {}
        state_dict_2 = {}
        for n, p in self.model.model.named_parameters():
            # print('n1',p)
            state_dict_1[n] = p
        for n, p in self.model2.model.named_parameters():
            # print('n2',p)
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
                if idx != (len(state_dict_1) - 1):
                    state_dict_11[name1] = state_dict_1[name]
                else:
                    name3 = '0.weight'
                    # state_dict_22[name3] = state_dict_1[name]
                    state_dict_11[name1] = state_dict_2[name]

            else:
                name1 = 'model.' + name
                state_dict_11[name1] = parameters[idx]
        self.model.load_state_dict(state_dict_11)
        self.model2.load_state_dict(state_dict_11)

    def _federated_averaging_ot(self):
        '''
        average the parameters of two models based on ot_based_fusion method
        '''
        # the following code can be replaced by other functions
        parameters, loss = ot.get_wassersteinized_layers_modularized(
            args=self.config, device=self.device, networks=[self.model, self.model2])
        state_dict_1 = {}
        state_dict_2 = {}
        for n in self.model.model.state_dict():
            # print('n1',n)
            if 'num_batches_tracked' in n:
                continue
            state_dict_1[n] = self.model.model.state_dict()[n]
        print(len(state_dict_1))
        for n in self.model2.model.state_dict():
            # print('n2',n)
            if 'num_batches_tracked' in n:
                continue
            state_dict_2[n] = self.model2.model.state_dict()[n]
        for n, p in self.model.heads.named_parameters():
            # print('n1',n)
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
                    name3 = '0.weight'
                    state_dict_11[name1] = state_dict_2[name]

            else:
                name1 = 'model.' + name
                state_dict_11[name1] = parameters[idx]
                # state_dict_22[name1] = parameters[idx]
        self.model.load_state_dict(state_dict_11)  
        self.model2.load_state_dict(state_dict_11)

    def _federated_averaging_ot1(self):
        '''
        average the parameters of two models based on ot_based_fusion method
        '''
        # the following code can be replaced by other functions
        parameters, loss = ot.get_wassersteinized_layers_modularized1(
            args=self.config, device=self.device, networks=[self.old_model1, self.model])

        state_dict_1 = {}
        state_dict_2 = {}
        for n, p in self.old_model1.model.named_parameters():
            # print('n1',p)
            state_dict_1[n] = p
        for n, p in self.model.model.named_parameters():
            # print('n2',p)
            state_dict_2[n] = p
        for n, p in self.old_model1.heads.named_parameters():
            state_dict_1[n] = p
        for n, p in self.model.heads.named_parameters():
            state_dict_2[n] = p
        state_dict_11 = {}
        state_dict_22 = {}
        for idx, (name, _) in enumerate(state_dict_2.items()):
            # print(name)
            if idx >= (len(state_dict_2) - len(self.model.heads)):
                name1 = 'heads.' + str(idx - (len(state_dict_2) - len(self.model.heads))) + '.' + 'weight'
                # print('change',name1)
                if idx != (len(state_dict_2) - len(self.model.heads)):
                    state_dict_11[name1] = state_dict_2[name]
                else:
                    name3 = '0.weight'
                    state_dict_11[name1] = state_dict_1[name]

            else:
                name1 = 'model.' + name
                state_dict_11[name1] = parameters[idx]
        self.model.load_state_dict(state_dict_11)
        self.model2.load_state_dict(state_dict_11)

    def _federated_averaging_ot_test(self):
        # the following code can be replaced by other functions
        parameters, loss = ot.get_wassersteinized_layers_modularized(
            args=self.config, device=self.device, networks=[self.model, self.model2])
        return loss

    def _federated_averaging_traditional(self):
        state_dict_1 = self.old_model.state_dict()
        state_dict_2 = self.model2.state_dict()
        state_dict_11 = {}
        for idx, (name, _) in enumerate(state_dict_1.items()):
            if idx >= len(state_dict_1) - len(self.old_model.heads):
                name1 = 'heads.' + str(idx - (len(state_dict_1) - len(self.old_model.heads))) + '.' + 'weight'
                if idx != (len(state_dict_1) - 1):
                    state_dict_11[name1] = 0.6 * state_dict_1[name] + 0.4 * state_dict_2[name]
                else:
                    state_dict_11[name1] = state_dict_2[name]
            else:
                name1 = name
                state_dict_11[name1] = 0.6 * state_dict_1[name] + 0.4 * state_dict_2[name]
        print('***' * 200)
        self.model.load_state_dict(state_dict_11)
        self.model2.load_state_dict(state_dict_11)

    def correct(self, t, trn_loader, val_loader):
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)
        lr = self.lr
        best_loss = np.inf
        patience = self.lr_patience
        target_model = self.model
        target_model.freeze_backbone()
        best_model = target_model.get_copy()
        self.optimizer = self._get_optimizer(target_model)

        # Loop epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch1(t, trn_loader, target_model)
            clock1 = time.time()
            if self.eval_on_train:
                train_loss, train_acc, _ = self.eval(t, trn_loader, target_model)
                clock2 = time.time()
                print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
            else:
                print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc, _ = self.eval(t, val_loader, target_model)
            clock4 = time.time()
            print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')
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
            print()
        target_model.set_state_dict(best_model)
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        pass

    def train_epoch(self, t, trn_loader, model):
        """Runs a single epoch"""
        model.train()
        if self.fix_bn and t > 0:
            model.freeze_bn()
        for images, targets in trn_loader:
            outputs_old = None
            if t > 0:
                outputs_old = self.old_model(images.to(self.device))
            features = model(images.to(self.device), return_features=True)
            outputs = model.heads[t](features)
            loss = self.criterion(t, [outputs], targets.to(self.device), outputs_old)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clipgrad)
            self.optimizer.step()

    def train_epoch1(self, t, trn_loader, model):
        model.train()
        if self.fix_bn and t > 0:
            model.freeze_bn()
        for images, targets in trn_loader:
            features = model(images.to(self.device), return_features=True)
            outputs = model.heads[t](features)
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
                # Forward current model
                features = model(images.to(self.device), return_features=True)
                outputs = [model.heads[i](features) for i in range(len(model.heads))]  # Lấy outputs từ tất cả heads
                loss = self.criterion1(t, outputs, targets.to(self.device))
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        loss = total_loss / total_num
        Taw = total_acc_taw / total_num
        Tag = total_acc_tag / total_num
        log_name = 'Validate_eval'
        self._logger.info('logger:{} ; '
                          'task: {task:#.4g};'
                          'loss: {loss:#.4g};  '
                          'Taw: {Taw:#.4g};  '
                          'Tag: {Tag:#.4g};  '.format(
            log_name, task=t, loss=loss, Taw=Taw, Tag=Tag)
        )
        return loss, Taw, Tag, OrderedDict([('task', t), ('loss', loss), ('Taw', Taw), ('Tag', Tag)])
    
    def eval1(self, t, val_loader):
        """Contains the evaluation code"""
        list1 = np.array(0)
        list2 = np.array(0)
        with torch.no_grad():
            self.model.eval()
            k = 0
            layer_num = 1
            for images, targets in val_loader:
                k = k+1
                outputs = self.model(images.to(self.device))
                images = images.to(self.device)
                activations = get_activation(self.model, images)
                activations1 = get_activation1(self.model, images, layer_num)

                if activations is not None:
                    activations_np = activations.cpu().numpy()
                    activations_np = np.reshape(activations_np, [activations_np.shape[1], activations_np.shape[0], -1])
                    channel_activations = activations_np.mean(axis=(1,2))
                    list1 = list1 + channel_activations
                    num_channels = channel_activations.shape[0]

                if activations1 is not None:
                    activations_np = activations1.cpu().numpy()
                    activations_np = np.reshape(activations_np, [activations_np.shape[1], activations_np.shape[0], -1])
                    channel_activations = activations_np.mean(axis=(1,2))
                    list2 = list2 + channel_activations
                    num_channels = channel_activations.shape[0]
            channel_activations = list1 / k
            channel_activations1 = list2 / k
            
            np.save('datas2/data_%d.npy'%t, channel_activations)
            np.save('datas2/data1_%d.npy'%t, channel_activations1)
            


        return 

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
            loss += self.cross_entropy(torch.cat(outputs[:t], dim=1),
                                       torch.cat(outputs_old[:t], dim=1), exp=0.5)
        return loss + torch.nn.functional.cross_entropy(outputs[t], targets.long() - self.model.task_offset[t])

    def criterion1(self, t, outputs, targets):
        """Returns the loss value"""

        return torch.nn.functional.cross_entropy(outputs[t], targets.long() - self.model.task_offset[t])