import time
import torch
import numpy as np
from argparse import ArgumentParser
from copy import deepcopy
from src.loggers.exp_logger import ExperimentLogger
from src.datasets.exemplars_dataset import ExemplarsDataset
from src.networks.lenet import LeNetArch
from src.networks.network import LLL_Net


class Inc_Learning_Appr:
    """Basic class for implementing incremental learning approaches"""

    def __init__(self, model, device, nepochs=1, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, logger: ExperimentLogger = None, exemplars_dataset: ExemplarsDataset = None):
        self.model = model
        self.device = device
        self.nepochs = nepochs
        self.decay_mile_stone = [80,120]
        self.lr_decay = 0.1
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
        self.optimizer = None

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

    def _get_optimizer(self):
        """Returns the optimizer"""
        # return torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def train(self, t, trn_loader, val_loader, consolidated_masks, curr_task_masks=None):
        """Main train structure"""
        self.pre_train_process(t, trn_loader)
        self.train_loop(t, trn_loader, val_loader, consolidated_masks, curr_task_masks)
        self.post_train_process(t, trn_loader)

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""

        # Warm-up phase
        if self.warmup_epochs and t > 0:
            self.optimizer = torch.optim.SGD(self.model.heads[-1].parameters(), lr=self.warmup_lr)
            # Loop epochs -- train warm-up head
            for e in range(self.warmup_epochs):
                warmupclock0 = time.time()
                self.model.heads[-1].train()   #训练
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

    def train_loop(self, t, trn_loader, val_loader, consolidated_masks, curr_task_masks):
        """Contains the epochs loop"""
        lr = self.lr
        best_loss = np.inf
        best_acc = 0
        patience = self.lr_patience
        best_model = deepcopy(self.model.state_dict())
        # best_model = self.model.get_copy()
        self.optimizer = self._get_optimizer()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.decay_mile_stone, gamma=self.lr_decay)
        # Loop epochs
        for e in range(1,self.nepochs+1):
            # Train
            clock0 = time.time()
            self.train_epoch(t, trn_loader, consolidated_masks)
            clock1 = time.time()
            if self.eval_on_train:
                train_loss, train_acc, _ = self.eval(t, trn_loader,curr_task_masks,mode='test')
                clock2 = time.time()
                print('| Epoch {:3d}, time={:5.1f}s/{:5.1f}s | Train: loss={:.3f}, TAw acc={:5.1f}% |'.format(
                    e + 1, clock1 - clock0, clock2 - clock1, train_loss, 100 * train_acc), end='')
                self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=train_loss, group="train")
                self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * train_acc, group="train")
            else:
                print('| Epoch {:3d}, time={:5.1f}s | Train: skip eval |'.format(e + 1, clock1 - clock0), end='')

            # Valid
            clock3 = time.time()
            valid_loss, valid_acc, _ = self.eval(t, val_loader, curr_task_masks=None,mode='valid')
            clock4 = time.time()
            scheduler.step()
            print(' Valid: time={:5.1f}s loss={:.3f}, TAw acc={:5.1f}% |'.format(
                clock4 - clock3, valid_loss, 100 * valid_acc), end='')
            # scheduler.step()
            self.logger.log_scalar(task=t, iter=e + 1, name="loss", value=valid_loss, group="valid")
            self.logger.log_scalar(task=t, iter=e + 1, name="acc", value=100 * valid_acc, group="valid")

            # # Adapt learning rate - patience scheme - early stopping regularization
            # if valid_acc >= best_acc:
            #     # if the loss goes down, keep it as the best model and end line with a star ( * )
            #     best_acc = valid_acc
            #     best_model = self.model.get_copy()
            #     patience = self.lr_patience
            #     print(' *', end='')
            # if valid_loss <= best_loss:
            #     best_loss = valid_loss
            #     best_acc = valid_acc
            #     best_model = self.model.get_copy()
            #     patience = self.lr_patience
            #     print(' *', end='')
            # elif valid_acc >= best_acc:
            #     best_loss = valid_loss
            #     best_acc = valid_acc
            #     best_model = self.model.get_copy()
            #     patience = self.lr_patience
            #     print(' *', end='')
            # else:
            #     # if the loss does not go down, decrease patience
            #     patience -= 1
            #     if patience <= 0:
            #         # if it runs out of patience, reduce the learning rate
            #         lr = self.lr
            #         print(' lr={:.1e}'.format(lr), end='')
            #         if lr < self.lr_min:
            #             # if the lr decreases below minimum, stop the training session
            #             print()
            #             break
            #         # reset patience and recover best model so far to continue training
            #         patience = self.lr_patience
            #         self.optimizer.param_groups[0]['lr'] = lr
            #         self.model.set_state_dict(best_model)
            # # self.logger.log_scalar(task=t, iter=e + 1, name="patience", value=patience, group="train")
            # # self.logger.log_scalar(task=t, iter=e + 1, name="lr", value=lr, group="train")
            print()
        best_model = deepcopy(self.model.state_dict())
        self.model.load_state_dict(deepcopy(best_model))

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        pass

    def train_epoch(self, t, trn_loader,consolidated_masks):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            # Forward current model
            outputs = self.model(images.to(self.device), t, mask=None, mode="train") 
            loss = self.criterion(t, outputs, targets.to(self.device))
            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Continual Subnet no backprop
            curr_head_keys = ["last.{}.weight".format(t), "last.{}.bias".format(t)]
            if consolidated_masks is not None and consolidated_masks != {}: # Only do this for tasks 1 and beyond
                # if args.use_continual_masks:
                for key in consolidated_masks.keys():
                    # Skip if not task head is not for curent task
                    if 'last' in key:
                        if key not in curr_head_keys:
                            continue

                    # Determine wheter it's an output head or not
                    key_split = key.split('.')
                    if 'last' in key_split or len(key_split) == 2:
                        if 'last' in key_split:
                            module_attr = key_split[-1]
                            task_num = int(key_split[-2])
                            module_name = '.'.join(key_split[:-2])

                        else:
                            module_attr = key_split[1]
                            module_name = key_split[0]

                        # Zero-out gradients
                        if (hasattr(getattr(self.model, module_name), module_attr)):
                            if (getattr(getattr(self.model, module_name), module_attr) is not None):
                                getattr(getattr(self.model, module_name), module_attr).grad[consolidated_masks[key] == 1] = 0

                    else:
                        module_attr = key_split[-1]

                        # Zero-out gradients
                        curr_module = getattr(getattr(self.model, key_split[0])[int(key_split[1])], key_split[2])
                        if hasattr(curr_module, module_attr):
                            if getattr(curr_module, module_attr) is not None:
                                getattr(curr_module, module_attr).grad[consolidated_masks[key] == 1] = 0
            # # Continual Subnet no backprop
            # curr_head_keys = ["last.{}.weight".format(t), "last.{}.bias".format(t)]
            # if consolidated_masks is not None and consolidated_masks != {}: # Only do this for tasks 1 and beyond
            #     # if args.use_continual_masks:
            #     for key in consolidated_masks.keys():
            #         # print('key',key)
            #         # Skip if not task head is not for curent task
            #         if 'last' in key:
            #             if key not in curr_head_keys:
            #                 continue

            #         # Determine whether it's an output head or not
            #         if (len(key.split('.')) == 4):  # e.g. last.1.weight
            #             module_name, task_num, _, module_attr = key.split('.')
            #             # curr_module = getattr(model, module_name)[int(task_num)]
            #         else: # e.g. fc1.weight
            #             module_name, module_attr = key.split('.')
            #             # curr_module = getattr(model, module_name)

            #         # Zero-out gradients
            #         if (hasattr(getattr(self.model, module_name), module_attr)):
            #             if (getattr(getattr(self.model, module_name), module_attr) is not None):
            #                 getattr(getattr(self.model, module_name), module_attr).grad[consolidated_masks[key] == 1] = 0
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def eval(self, t, val_loader,curr_task_masks=None,mode="test"):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            per_task_masks = {}
            for images, targets in val_loader:
                # Forward current 
                # Save the per-task-dependent masks

                # per_task_masks[t] = self.model.get_masks(t)

                outputs = self.model(images.to(self.device), t, mask=curr_task_masks, mode=mode)   #output为一个list list中的shape为[64,2]
                loss = self.criterion(t, outputs, targets.to(self.device))
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def calculate_metrics(self, outputs, targets):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        """任务可知 和 任务不可知"""
        pred = torch.zeros_like(targets.to(self.device))
        # Task-Aware Multi-Head
        for m in range(len(pred)):
            this_task = (self.model.task_cls.cumsum(0) <= targets[m]).sum()
            pred[m] = outputs[this_task][m].argmax() + self.model.task_offset[this_task]
        hits_taw = (pred == targets.to(self.device)).float()
        # print("hits_taw",pred[0])
        # Task-Agnostic Multi-Head
        if self.multi_softmax:
            outputs = [torch.nn.functional.log_softmax(output, dim=1) for output in outputs]
            pred = torch.cat(outputs, dim=1).argmax(1)
        else:
            pred = torch.cat(outputs, dim=1).argmax(1)
        hits_tag = (pred == targets.to(self.device)).float()
        return hits_taw, hits_tag

    def criterion(self, t, outputs, targets):
        """Returns the loss value"""
        return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
