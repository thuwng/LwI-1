import logging
import os
import pickle
import sys
import time
import torch
from collections import OrderedDict
import argparse
import importlib
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce   
import yaml
import src.utils as utils
import src.approach
# from search_new import construct_log
from src.utils import setup_seed, print_args, construct_log
from src.loggers.exp_logger import MultiLogger
from src.loggers.disk_logger import Logger
from src.datasets.data_loader import get_loaders
from src.datasets.dataset_config import dataset_config
from src.last_layer_analysis import last_layer_analysis
from src.networks import tvmodels, allmodels, set_tvmodel_head_var

_logger = logging.getLogger('train')
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

class MergeInfoMessagesHandler(logging.Handler):
    def __init__(self):
        super(MergeInfoMessagesHandler, self).__init__()
        self.messages = []

    def emit(self, record):
        if record.levelno == logging.INFO:
            self.messages.append(record.getMessage())
        else:
            # Log the accumulated INFO messages as one line
            if self.messages:
                merged_message = ' - '.join(self.messages)
                self.messages = []  # Reset the messages list
                self.record = record
                self.record.message = merged_message
                super(MergeInfoMessagesHandler, self).emit(self.record)


def main(argv=None):
    tstart = time.time()

    parser = argparse.ArgumentParser(description='')

    # miscellaneous args
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU (default=%(default)s)')
    parser.add_argument('--results-path', type=str, default='../results',
                        help='Results path (default=%(default)s)')
    parser.add_argument('--exp-name', default=None, type=str,
                        help='Experiment name (default=%(default)s)')
    parser.add_argument('--log', default=['disk'], type=str, choices=['disk', 'tensorboard'],
                        help='Loggers used (disk, tensorboard) (default=%(default)s)', nargs='*', metavar="LOGGER")
    parser.add_argument('--save-models', action='store_true',
                        help='Save trained models (default=%(default)s)')
    parser.add_argument('--last-layer-analysis', action='store_true',
                        help='Plot last layer analysis (default=%(default)s)')
    parser.add_argument('--no-cudnn-deterministic', action='store_true',
                        help='Disable CUDNN deterministic (default=%(default)s)')
    # dataset args
    parser.add_argument('--datasets', default=['cifar10'], type=str, choices=list(dataset_config.keys()),
                        help='Dataset or datasets used (default=%(default)s)', nargs='+', metavar="DATASET")
    parser.add_argument('--num-workers', default=1, type=int, required=False,
                        help='Number of subprocesses to use for dataloader (default=%(default)s)')
    parser.add_argument('--pin-memory', default=False, type=bool, required=False,
                        help='Copy Tensors into CUDA pinned memory before retusrning them (default=%(default)s)')
    parser.add_argument('--batch-size', default=64, type=int, required=False,
                        help='Number of samples per batch to load (default=%(default)s)')
    parser.add_argument('--num-tasks', default=5, type=int, required=False,
                        help='Number of tasks per dataset (default=%(default)s)')
    parser.add_argument('--nc-first-task', default=None, type=int, required=False,
                        help='Number of classes of the first task (default=%(default)s)')
    parser.add_argument('--use-valid-only', action='store_true',
                        help='Use validation split instead of test (default=%(default)s)')
    parser.add_argument('--stop-at-task', default=0, type=int, required=False,
                        help='Stop training after specified task (default=%(default)s)')
    # model args
    parser.add_argument('--network', default='resnet32', type=str, choices=allmodels,
                        help='Network architecture used (default=%(default)s)', metavar="NETWORK")
    parser.add_argument('--keep-existing-head', action='store_true',
                        help='Disable removing classifier last layer (default=%(default)s)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained backbone (default=%(default)s)')
    # training argsW
    parser.add_argument('--approach', default='ours', type=str, choices=src.approach.__all__,
                        help='Learning approach used (default=%(default)s)', metavar="APPROACH")
    parser.add_argument('--nepochs', default=200, type=int, required=False,
                        help='Number of epochs per training session (default=%(default)s)')
    parser.add_argument('--learning_rate', default=0.1, type=float, required=False,
                        help='Starting learning rate (default=%(default)s)')
    parser.add_argument('--lr-min', default=1e-5, type=float, required=False,
                        help='Minimum learning rate (default=%(default)s)')
    parser.add_argument('--lr-factor', default=3, type=float, required=False,
                        help='Learning rate decreasing factor (default=%(default)s)')
    parser.add_argument('--lr-patience', default=10, type=int, required=False,
                        help='Maximum patience to wait before decreasing learning rate (default=%(default)s)')
    parser.add_argument('--clipping', default=10000, type=float, required=False,
                        help='Clip gradient norm (default=%(default)s)')
    parser.add_argument('--momentum', default=0.0, type=float, required=False,
                        help='Momentum factor (default=%(default)s)')
    parser.add_argument('--weight-decay', default=0.0, type=float, required=False,
                        help='Weight decay (L2 penalty) (default=%(default)s)')
    parser.add_argument('--warmup-nepochs', default=0, type=int, required=False,
                        help='Number of warm-up epochs (default=%(default)s)')
    parser.add_argument('--warmup_lr_factor', default=1.0, type=float, required=False,
                        help='Warm-up learning rate factor (default=%(default)s)')
    parser.add_argument('--multi-softmax', action='store_true',
                        help='Apply separate softmax for each task (default=%(default)s)')
    parser.add_argument('--fix-bn', action='store_true',
                        help='Fix batch normalization after first task (default=%(default)s)')
    parser.add_argument('--eval-on-train', action='store_true',
                        help='Show train loss and accuracy (default=%(default)s)')
    # gridsearch args
    parser.add_argument('--gridsearch-tasks', default=-1, type=int,
                        help='Number of tasks to apply GridSearch (-1: all tasks) (default=%(default)s)')
    parser.add_argument('--ensemble_step', default=0.68, type=float,
                        help='the ensemble weight used in gm-based fusion (default: 0.5)')
    parser.add_argument('--al', default=1, type=float,
                        help='')
    parser.add_argument('--we', default=0.8, type=float,
                        help='the ensemble weight used in gm-based fusion (default: 0.5)')
    parser.add_argument('--layers', default=5, type=int,
                        help='the ensemble layers used in gm-based fusion (default: 5)')
    parser.add_argument('--ensemble_step_diff', default=0.9, type=float,
                        help='the difference ensemble weight used in gm-based fusion (default: 0.5)')
    parser.add_argument('--reg', default=0.01, type=float,
                        help='the control between sinkhorn algorithm and Hungarian algorithm')
    parser.add_argument('--training_mode', default='ot', type=str,
                        help='whether to use traditional averaging or fusion-based averaging, \
                            can be traditional or fusion or fusion_slice or ot (default: traditional)')
   ###################### myargs #####################
    parser.add_argument('-negative_sample', '--negative_sample', type=float, default=1.0, help='')
    parser.add_argument('-redcircle_dis_thresh', '--redcircle_dis_thresh', type=float, default=10.0, help='')
    parser.add_argument('-pad_size', '--pad_size', type=int, default=None, help='')
    parser.add_argument('--val_fold_i', type=int, default=3, help="val fold")
    parser.add_argument('--data_in_type', type=str, default='cam_gold_shot', help="name of the host")
    parser.add_argument('--checkpoint_data_in_type', type=str, default='cam_shot', help="name of the host")
    parser.add_argument('--model_in_type', type=str, default='passage', help="[concat, passage]")
    parser.add_argument('--passage_middle_layer', type=int, default=3, help="")
    parser.add_argument('--dataset_used', type=str, default='auto_annoted_cut', help="[manually_annoted_filtered, manually_annoted, auto_annoted_cut]")
    parser.add_argument('--seed', type=int, default=100, help="")
    parser.add_argument('--output', type=str, default="/output",
                        help="the dir of source code")
    # Args -- Incremental Learning Framework
    args, extra_args = parser.parse_known_args(argv)
    args.clip_gm = False
    args.multi_softmax = False
    args.fix_bn = False
    args.eval_on_train = False
    args.correction = True
    # args.dataset = "imagenet_256"
    args.debug = False
    args.dist_normalize = True
    args.eval_aligned = False
    args.exact = 0
    args.geom_ensemble_type = "wts"
    args.gpu_id = 0
    args.ground_metric = "cosine"
    args.ground_metric_eff = True
    args.ground_metric_normalize = "log"
    args.importance = None
    args.normalize_wts = False
    args.num_models = 2
    args.not_squared = True
    args.past_correction = True
    args.prediction_wts = True
    args.proper_marginals = False
    args.skip_last_layer = False
    args.softmax_temperature = 1
    args.unbalanced = False
    args.width_ratio = 1
    args.max = True
    args.wd = 0
    args.act_num_samples = 200

    # args.results_path = os.path.expanduser(args.results_path)
    args.results_path = '/kaggle/working/results'
    args.output = '/kaggle/working/output'
    os.makedirs(args.results_path, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)
    setup_seed(args.seed)

    base_kwargs = args
    if args.no_cudnn_deterministic:
        print('WARNING: CUDNN Deterministic will be disabled.')
        utils.cudnn_deterministic = False

    utils.seed_everything(seed=args.seed)
    print('=' * 108)
    print('Arguments =')
    for arg in np.sort(list(vars(args).keys())):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 108)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    pickle_log = {'train': {}, 'eval': {}, 'test': {}, 'eval_noema': {}}
    # Args -- Network
    from src.networks.network import LLL_Net
    if args.network in tvmodels:  # torchvision models
        tvnet = getattr(importlib.import_module(name='torchvision.models'), args.network)
        if args.network == 'googlenet':
            init_model = tvnet(pretrained=args.pretrained, aux_logits=False)
        else:
            init_model = tvnet(pretrained=args.pretrained)
        set_tvmodel_head_var(init_model)
    else:
        net = getattr(importlib.import_module(name='src.networks'), args.network)
        init_model = net()
    from src.approach.incremental_learning import Inc_Learning_Appr
    try:
        module = importlib.import_module(name='src.approach.' + args.approach)
        Appr = getattr(module, 'Appr', None)
        if not Appr or not issubclass(Appr, Inc_Learning_Appr):
            raise ValueError(f"Class {args.approach}.Appr must be a subclass of Inc_Learning_Appr")
    except ImportError as e:
        raise ImportError(f"Failed to import module src.approach.{args.approach}: {str(e)}")
    appr_args, extra_args = Appr.extra_parser(extra_args)
    for arg in np.sort(list(vars(appr_args).keys())):
        print('\t' + arg + ':', getattr(appr_args, arg))
    print('=' * 108)

    from src.datasets.exemplars_dataset import ExemplarsDataset
    Appr_ExemplarsDataset = Appr.exemplars_dataset_class()
    if Appr_ExemplarsDataset:
        assert issubclass(Appr_ExemplarsDataset, ExemplarsDataset)
        appr_exemplars_dataset_args, extra_args = Appr_ExemplarsDataset.extra_parser(extra_args)
        print('Exemplars dataset arguments =')
        for arg in np.sort(list(vars(appr_exemplars_dataset_args).keys())):
            print('\t' + arg + ':', getattr(appr_exemplars_dataset_args, arg))
        print('=' * 108)
    else:
        appr_exemplars_dataset_args = argparse.Namespace()

    # Args -- GridSearch
    if args.gridsearch_tasks > 0:
        from src.gridsearch import GridSearch
        gs_args, extra_args = GridSearch.extra_parser(extra_args)
        Appr_finetuning = getattr(importlib.import_module(name='src.approach.test'), 'Appr')
        assert issubclass(Appr_finetuning, Inc_Learning_Appr)
        GridSearch_ExemplarsDataset = Appr.exemplars_dataset_class()
        print('GridSearch arguments =')
        for arg in np.sort(list(vars(gs_args).keys())):
            print('\t' + arg + ':', getattr(gs_args, arg))
        print('=' * 108)

    assert len(extra_args) == 0, "Unused args: {}".format(' '.join(extra_args))
    ####################################################################################################################

    # Log all arguments
    full_exp_name = reduce((lambda x, y: x[0] + y[0]), args.datasets) if len(args.datasets) > 0 else args.datasets[0]
    full_exp_name += '_' + args.approach
    if args.exp_name is not None:
        full_exp_name += '_' + args.exp_name
    logger = MultiLogger(args.results_path, full_exp_name, loggers=args.log, save_models=args.save_models)
    logger.log_args(argparse.Namespace(**args.__dict__, **appr_args.__dict__, **appr_exemplars_dataset_args.__dict__))

    # Loaders
    trn_loader, val_loader, tst_loader, taskcla = get_loaders(args.datasets, args.num_tasks, args.nc_first_task,
                                                              args.batch_size, num_workers=args.num_workers,
                                                              pin_memory=args.pin_memory)
    if args.use_valid_only:
        tst_loader = val_loader
    max_task = 20
    net = LLL_Net(init_model, remove_existing_head=not args.keep_existing_head)
    first_train_ds = trn_loader[0].dataset
    transform, class_indices = first_train_ds.transform, first_train_ds.class_indices
    appr = Appr(net, device, logger=logger, exemplars_dataset=None, **vars(base_kwargs))

    # GridSearch
    if args.gridsearch_tasks > 0:
        appr_ft = Appr_finetuning(net, device, base_kwargs,None,None)
        gridsearch = GridSearch(appr_ft, args.seed, gs_args.gridsearch_config, gs_args.gridsearch_acc_drop_thr,
                                gs_args.gridsearch_hparam_decay, gs_args.gridsearch_max_num_searches)

    # Loop tasks
    print("taskcla",taskcla)
    acc_taw = np.zeros((max_task, max_task))
    acc_tag = np.zeros((max_task, max_task))
    forg_taw = np.zeros((max_task, max_task))
    forg_tag = np.zeros((max_task, max_task))


    for t, (_, ncla) in enumerate(taskcla):
        if t >= max_task:
            continue

        print('*' * 108)
        print('Task {:2d}'.format(t))
        print('*' * 108)

        net.add_head(taskcla[t][1])
        net.to(device)

        # GridSearch
        if t < args.gridsearch_tasks:

            print('LR GridSearch')
            best_ft_acc, best_ft_lr = gridsearch.search_lr(appr.model, t, trn_loader[t], val_loader[t])
            # Apply to approach
            appr.lr = best_ft_lr
            gen_params = gridsearch.gs_config.get_params('general')
            for k, v in gen_params.items():
                if not isinstance(v, list):
                    setattr(appr, k, v)

            print('Trade-off GridSearch')
            best_tradeoff, tradeoff_name = gridsearch.search_tradeoff(args.approach, appr,
                                                                      t, trn_loader[t], val_loader[t], best_ft_acc)
            if tradeoff_name is not None:
                setattr(appr, tradeoff_name, best_tradeoff)

            print('-' * 108)

        # Train
        appr.train(t, trn_loader[t], val_loader[t])
        print('-' * 108)


        # Test
        for u in range(t + 1):
            test_loss, acc_taw[t, u], acc_tag[t, u],result = appr.eval(u, tst_loader[u],net)

            if u < t:
                forg_taw[t, u] = acc_taw[:t, u].max(0) - acc_taw[t, u]
                forg_tag[t, u] = acc_tag[:t, u].max(0) - acc_tag[t, u]
            print('>>> Test on task {:2d} : loss={:.3f} | TAw acc={:5.1f}%, forg={:5.1f}%'
                  '| TAg acc={:5.1f}%, forg={:5.1f}% <<<'.format(u, test_loss,
                                                                 100 * acc_taw[t, u], 100 * forg_taw[t, u],
                                                                 100 * acc_tag[t, u], 100 * forg_tag[t, u]))
            _logger.info(
                     'task: {task:#.4g};'
                     'iter: {iter:#.4g};'
                     'Taw: {Taw:#.4g};  '
                     'Tag: {Tag:#.4g};  '
                     'forg_taw: {forg_taw:#.4g};  '
                     'forg_tag: {forg_tag:#.4g};  '.format(task=t, iter=u, Taw=100*acc_taw[t, u], Tag=100*acc_tag[t, u], forg_taw=100*forg_taw[t, u],forg_tag=100*forg_tag[t, u]))

    metrics = []
    for name, metric in zip(['TAwa', 'TAga', 'TAwf', 'TAgf'], [acc_taw, acc_tag, forg_taw, forg_tag]):
        _logger.info('*' * 108)
        _logger.info('{}'.format(name))
        # _logger.info('name: {name: };'.format(name))
        for i in range(metric.shape[0]):
            # _logger.info('\t', end='')
            for j in range(metric.shape[1]):
                # pass
                _logger.info('{:5.1f}% '.format(100 * metric[i, j]))
            if np.trace(metric) == 0.0:
                if i > 0:
                    # pass
                    r1 = 100 * metric[i, :i].mean()
                    _logger.info('Avg.:{:5.1f}% '.format(100 * metric[i, :i].mean()))
            else:
                # pass
                r1 = 100 * metric[i, :i + 1].mean()
                _logger.info('Avg.:{:5.1f}% '.format(100 * metric[i, :i + 1].mean()))
            # _logger.info()
            if i==(metric.shape[0]-1):
                metrics.append(r1)

    # for name , metrcic in zip(['TAw Acc', 'TAg Acc', 'TAw Forg', 'TAg Forg'],metrics):
    result = OrderedDict([('TAwa',metrics[0]),('TAga',metrics[1]),('TAwf',metrics[2]),('TAgf',metrics[3])])
    return acc_taw, acc_tag, forg_taw, forg_tag, logger.exp_path
    ####################################################################################################################


if __name__ == '__main__':
    main()

