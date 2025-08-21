import argparse
from copy import deepcopy

from torch import nn

from src.networks.lenet import LeNetArch
import torch
import itertools
from argparse import ArgumentParser
import src.approach.our_ot as ot  
from src.datasets.exemplars_dataset import ExemplarsDataset
from src.approach.incremental_learning import Inc_Learning_Appr
from src.networks.network import LLL_Net
import torch.nn.functional as F


class Appr(Inc_Learning_Appr):
    def __init__(self, model, device, args, logger,exemplars_dataset):
        super(Appr, self).__init__(model, device, args,logger,exemplars_dataset)

        self.lamb = 1
        self.alpha = 0.5
        self.sampling_type = 'max_pred'
        self.num_samples = -1
        self.model_old = None
        feat_ext = self.model.model
        feat_ext1 = self.model2.model
        self.older_params = {n: p.clone().detach() for n, p in feat_ext.named_parameters() if p.requires_grad}
        self.new_params = {n: p.clone().detach() for n, p in feat_ext1.named_parameters() if p.requires_grad}
        self.fisher0 = {n: torch.zeros(p.shape).to(self.device) for n, p in feat_ext.named_parameters()
                       if p.requires_grad}
        self.fisher1 = {n: torch.zeros(p.shape).to(self.device) for n, p in feat_ext1.named_parameters()
                       if p.requires_grad}
        self.loss0 = 0


    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

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

    def _get_optimizer(self,model):
        """Returns the optimizer"""
        if len(model.heads) > 1:
            params = list(model.model.parameters()) + list(model.heads[-1].parameters())
        else:
            params = model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)





    def train_loop(self, t, trn_loader, val_loader):
        super().train_loop(t, trn_loader, val_loader)

    def get_activation(name,output_data):
        def hook(model,input,output):
            output_data[name] = output.detach()
        return hook

    def compute_fisher_matrix_diag(self, trn_loader,model):
        fisher = {n: (torch.zeros(p.shape).to(self.device)) for n, p in model.model.named_parameters()
                  if p.requires_grad}
        n_samples_batches = (self.num_samples // trn_loader.batch_size + 1) if self.num_samples > 0 \
            else (len(trn_loader.dataset) // trn_loader.batch_size)
        model.train()



        for images, targets in itertools.islice(trn_loader, n_samples_batches):
            outputs = model.forward(images.to(self.device))

            if self.sampling_type == 'true':
                preds = targets.to(self.device)
            elif self.sampling_type == 'max_pred':
                preds = torch.cat(outputs, dim=1).argmax(1).flatten()
            elif self.sampling_type == 'multinomial':
                probs = torch.nn.functional.softmax(torch.cat(outputs, dim=1), dim=1)
                preds = torch.multinomial(probs, len(targets)).flatten()

            loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), preds)
            self.optimizer.zero_grad()
            loss.backward()
            for n, p in model.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2) * len(targets)
        n_samples = n_samples_batches * trn_loader.batch_size
        fisher = {n: (p / n_samples) for n, p in fisher.items()}
        return fisher

    def post_train_process(self, t, trn_loader):
        self.old_model = deepcopy(self.model2)
        self.old_model.eval()
        self.old_model.freeze_all()
        
        if t>0:

            if self.config.training_mode == 'ot':
                self.loss0 = self._federated_averaging_ot()
            elif self.config.training_mode == 'traditional':
                self._federated_averaging_traditional()
            elif self.config.training_mode == 'ewc':
                self._federated_averaging_ewc()


        

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

    def criterion(self, t, outputs, targets,outputs_old=None):
        loss = 0
        
        if t > 0:
            loss += self.lamb * self.cross_entropy(torch.cat(outputs[:t], dim=1),
                                                   torch.cat(outputs_old[:t], dim=1), exp=0.5)
        return loss + torch.nn.functional.cross_entropy(outputs[t], targets.long() - self.model.task_offset[t])
