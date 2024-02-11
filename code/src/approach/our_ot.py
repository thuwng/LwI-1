import math
import logging
import ot
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch.nn as nn
from src.approach.our_groundmetric import GroundMetric
torch.set_printoptions(profile="full")
from copy import deepcopy

# import compute_activations

def compute_optimal_transport(M, r, c, lam, eplison=1e-8):
    n, m = M.shape  # 8, 5
    P = np.exp(lam * M)  # (8, 5)
    P /= P.sum() 
    # print('P', P)
    u = np.zeros(n)  # (8, )
    # normalize this matrix
    while np.max(np.abs(u - P.sum(1))) > eplison: 
        u = P.sum(1) 
        # print('u',u)
        P *= (r / u).reshape((-1, 1)) 
        # print('pu', P)
        v = P.sum(0) 
        # print('v',v)
        P *= (c / v).reshape((1, -1)) 
        # print('pv', P)

    return P, np.sum(P * M)


def sinkhorn_torch(M, a, b, lambda_sh, numItermax=5000, stopThr=.5e-2, cuda=False):
    if cuda:
        u = (torch.ones_like(a) / a.size()[0]).double().cuda()
        v = (torch.ones_like(b)).double().cuda()
    else:
        u = (torch.ones_like(a) / a.size()[0])
        v = (torch.ones_like(b))

    K = torch.exp(-M * lambda_sh)
    err = 1
    cpt = 0
    while err > stopThr and cpt < numItermax:
        u = torch.div(a, torch.matmul(K, torch.div(b, torch.matmul(u.t(), K).t())))
        cpt += 1
        if cpt % 20 == 1:
            v = torch.div(b, torch.matmul(K.t(), u))
            u = torch.div(a, torch.matmul(K, v))
            bb = torch.mul(v, torch.matmul(K.t(), u))
            err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))

    sinkhorn_divergences = torch.sum(torch.mul(u, torch.matmul(torch.mul(K, M), v)), dim=0)
    return sinkhorn_divergences


def sinkhorn(M, r=None, c=None, gamma=1.0, eps=1.0e-6, maxiters=1000, logspace=False):
    M = torch.from_numpy(M)
    H, W = M.shape
    # assert r is None or r.shape == (B, H) or r.shape == (1, H)
    # assert c is None or c.shape == (B, W) or c.shape == (1, W)
    assert not logspace or torch.all(M > 0.0)

    r = 1.0 / H if r is None else r.unsqueeze(dim=2)
    c = 1.0 / W if c is None else c.unsqueeze(dim=1)

    if logspace:
        P = torch.pow(M, gamma)
    else:
        P = torch.exp(1.0 * gamma * (M - torch.amin(M, 1, keepdim=True)))

    for i in range(maxiters):
        alpha = torch.sum(P, 1)
        # Perform division first for numerical stability
        P = P / alpha.view(H, 1) * r

        beta = torch.sum(P, 1)
        if torch.max(torch.abs(beta - c)) <= eps:
            break
        P = P / beta.view(1, W) * c
    P = P.numpy()
    return P


def sinkhorn1(dist_matrix, regularization_param, max_iters=1000, tolerance=1e-4):
    n, m = dist_matrix.shape
    K = np.exp(-dist_matrix / regularization_param)
    u = np.ones(n)
    v = np.ones(m)
    num_iterations = 0

    while num_iterations < max_iters:
        u_prev = u
        v_prev = v
        u = 1 / np.dot(K, v)
        v = 1 / np.dot(K.T, u)

        num_iterations += 1
        if np.allclose(u, u_prev, rtol=0, atol=tolerance) and np.allclose(v, v_prev, rtol=0, atol=tolerance):
            break
    matching_matrix = np.outer(u, v) * K
    row_indices, col_indices = linear_sum_assignment(-matching_matrix)
    n = len(row_indices)
    m = len(col_indices)
    matching_matrix = np.zeros((n, m), dtype=float)

    for i in range(n):
        matching_matrix[i, col_indices[i]] = 1
    return matching_matrix, num_iterations


def cost_matrix(x, y, p=2):
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
    return c


def get_histogram(args, idx, cardinality, layer_name, activations=None, return_numpy=True, float64=False):
    if activations is None:
        if not args.unbalanced:
            return np.ones(cardinality) / cardinality
        else:
            return np.ones(cardinality)
    else:
        print(activations[idx].keys())
        unnormalized_weights = activations[idx][layer_name.split('.')[0]]
        print("For layer {},  shape of unnormalized weights is ".format(layer_name), unnormalized_weights.shape)
        unnormalized_weights = unnormalized_weights.squeeze()
        assert unnormalized_weights.shape[0] == cardinality

        if return_numpy:
            if float64:
                return torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0).data.cpu().numpy().astype(
                    np.float64)
            else:
                return torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0).data.cpu().numpy()
        else:
            return torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0)


def get_wassersteinized_layers_modularized_tests(args, device, networks, activations=None, eps=1e-7, test_loader=None):

    avg_aligned_layers = []
    T_var = None
    T1_var = None
    ground_metric_object = GroundMetric(args)

    if args.eval_aligned:
        model0_aligned_layers = []

    if args.gpu_id == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu_id))
    loss = 0
    params1 = {}
    params2 = {}
    for n, p in networks[0].model.named_parameters():
        params1[n] = p
    for n, p in networks[1].model.named_parameters():
        params2[n] = p
    num_layers = len(params1)
    TT = 0
    for idx, ((layer0_name), (layer1_name)) in \
            enumerate(zip(params1, params2)):
        fc_layer0_weight = params2[layer1_name]
        fc_layer1_weight = params1[layer0_name]
        assert fc_layer0_weight.shape == fc_layer1_weight.shape
        previous_layer_shape = fc_layer1_weight.shape

        mu_cardinality = fc_layer0_weight.shape[0]
        nu_cardinality = fc_layer1_weight.shape[0]

        layer_shape = fc_layer0_weight.shape

        if len(layer_shape) > 2:
            is_conv = True
            fc_layer0_weight_data = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], fc_layer0_weight.shape[1], -1)
            fc_layer1_weight_data = fc_layer1_weight.data.view(fc_layer1_weight.shape[0], fc_layer1_weight.shape[1], -1)
        else:
            is_conv = False
            fc_layer0_weight_data = fc_layer0_weight.data
            fc_layer1_weight_data = fc_layer1_weight.data

        if idx == 0:
            if is_conv:
                M = ground_metric_object.process(fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
                                                 fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
            else:
                M = ground_metric_object.process(fc_layer0_weight_data, fc_layer1_weight_data)

            aligned_wt = fc_layer0_weight_data
        else:

            if is_conv:
                T_var_conv = T_var.unsqueeze(0).repeat(fc_layer0_weight_data.shape[2], 1, 1)
                aligned_wt = torch.bmm(fc_layer0_weight_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)
                M = ground_metric_object.process(
                    aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                    fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
                )
            else:

                if fc_layer0_weight.data.shape[1] != T_var.shape[0]:
                    fc_layer0_unflattened = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], T_var.shape[0],
                                                                       -1).permute(2, 0, 1)
                    aligned_wt = torch.bmm(
                        fc_layer0_unflattened,
                        T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1)
                    ).permute(1, 2, 0)
                    aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                else:
                    aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)
                M = ground_metric_object.process(aligned_wt, fc_layer1_weight)
            if args.skip_last_layer and idx == (num_layers - 1):
                if args.ensemble_step != 0.5:
                    avg_aligned_layers.append((1 - args.ensemble_step) * aligned_wt +
                                              args.ensemble_step * fc_layer1_weight)
                else:
                    avg_aligned_layers.append((aligned_wt + fc_layer1_weight) / 2)
                return avg_aligned_layers

        if args.importance is None or (idx == num_layers - 1):
            mu = get_histogram(args, 0, mu_cardinality, layer0_name)
            nu = get_histogram(args, 1, nu_cardinality, layer1_name)
        else:
            # mu = _get_neuron_importance_histogram(args, aligned_wt, is_conv)
            mu = _get_neuron_importance_histogram(args, fc_layer0_weight_data, is_conv)
            nu = _get_neuron_importance_histogram(args, fc_layer1_weight_data, is_conv)
            # Alexanderia

        cpuM = M.data.cpu().numpy()

        if args.exact == 0:
            T = ot.emd(mu, nu, -cpuM)
            T1 = ot.emd(mu,nu,-cpuM)
        elif args.exact == 1:
            T, _ = sinkhorn1(cpuM, 1)
            args.correction = False
        else:
            T = ot.bregman.sinkhorn_log(mu, nu, cpuM, reg=args.reg, numItermax=20000)
            TT = ot.bregman.sinkhorn2(mu,nu, cpuM, reg=args.reg, numItermax=20000)
            loss = loss+TT
        if args.gpu_id != -1:
            T_var = torch.from_numpy(T).to(device).float()
            T1_var = torch.from_numpy(T1).to(device).float()
        else:
            T_var = torch.from_numpy(T).to(device).float()
            T1_var = torch.from_numpy(T1).to(device).float()

        if args.correction:
            if not args.proper_marginals:
                if args.gpu_id != -1:
                    marginals = torch.ones(T_var.shape[0]).to(device) / T_var.shape[0]
                    marginals1 = torch.ones(T1_var.shape[0]).to(device) / T1_var.shape[0]
                else:
                    marginals = torch.ones(T_var.shape[0]).to(device) / T_var.shape[0]
                    marginals1 = torch.ones(T1_var.shape[0]).to(device) / T1_var.shape[0]
                marginals = torch.diag(1.0 / (marginals + eps))
                marginals1 = torch.diag(1.0 / (marginals1 + eps))
                T_var = torch.matmul(T_var, marginals)
                T1_var = torch.matmul(T1_var, marginals1)

            else:
                marginals_beta = T_var.t() @ torch.ones(T_var.shape[0], dtype=T_var.dtype).to(device)
                marginals_beta1 = T1_var.t() @ torch.ones(T1_var.shape[0], dtype=T1_var.dtype).to(device)
                marginals = (1 / (marginals_beta + eps))
                marginals1 = (1 / (marginals_beta1 + eps))
                T_var = T_var * marginals
                T_var = T1_var * marginals1
        
        if args.debug:
            if idx == (num_layers - 1):
                print("there goes the last transport map: \n ", T_var)
            else:
                print("there goes the transport map at layer {}: \n ".format(idx), T_var)

            print("Ratio of trace to the matrix sum: ", torch.trace(T_var) / torch.sum(T_var))

        # print("Here, trace is {} and matrix sum is {} ".format(torch.trace(T_var), torch.sum(T_var)))
        setattr(args, 'trace_sum_ratio_{}'.format(layer0_name), (torch.trace(T_var) / torch.sum(T_var)).item())

        if args.past_correction:

            t_fc0_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))
        else:
            t_fc0_model = torch.matmul(T_var.t(), fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1))


        geometric_fc = ((args.ensemble_step) * t_fc0_model +
                        (1-args.ensemble_step) * fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
        if is_conv and layer_shape != geometric_fc.shape:
            geometric_fc = geometric_fc.view(layer_shape)
        avg_aligned_layers.append(geometric_fc)


    return avg_aligned_layers, loss



def get_wassersteinized_layers_modularized(args, device, networks, activations=None, eps=1e-7, test_loader=None):


    avg_aligned_layers = []
    T_var = None
    ground_metric_object = GroundMetric(args)

    if args.eval_aligned:
        model0_aligned_layers = []

    if args.gpu_id == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu_id))
    loss = 0
    params1 = {}
    params2 = {}
    idx = 0 
    for n in networks[0].model.state_dict():
        if 'num_batches_tracked' in n:
            continue
        params1[n] = networks[0].model.state_dict()[n]

    for n in networks[1].model.state_dict():
        if 'num_batches_tracked' in n:
            continue
        params2[n] = networks[1].model.state_dict()[n]
    num_layers = len(params1)
    TT = 0
    idxx = 0
    for idx, ((layer0_name), (layer1_name)) in \
            enumerate(zip(params1, params2)):
        fc_layer0_weight = params1[layer0_name]
        fc_layer1_weight = params2[layer1_name]
        layer_shape = fc_layer0_weight.shape
        if 'num_batches_tracked' in layer0_name:
            continue

        if 'shortcut' in layer0_name:
            if idxx == 0:
                M1 = ground_metric_object.process(fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
                                                  fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
                cpuM1 = M1.data.cpu().numpy()
                mu = get_histogram(args, 0, mu_cardinality, layer0_name)
                nu = get_histogram(args, 1, nu_cardinality, layer1_name)
                if idx >= (num_layers - 1):
                    cpuM1 = -cpuM1
                    ensemble_step = args.ensemble_step_diff
                else:
                    cpuM1 = cpuM1
                    ensemble_step = args.ensemble_step

                if args.exact == 0:
                    T11 = ot.emd(mu, nu, cpuM1)
                elif args.exact == 1:
                    T11, _ = sinkhorn1(cpuM1, 1)
                    args.correction = False
                else:
                    T11 = ot.bregman.sinkhorn_log(mu, nu, cpuM1, reg=args.reg, numItermax=20000)
                    TT = ot.bregman.sinkhorn2(mu, nu, cpuM1, reg=args.reg, numItermax=20000)
                if args.gpu_id != -1:
                    T_var1 = torch.from_numpy(T11).to(device).float()
                    T1_var = torch.from_numpy(T1).to(device).float()
                else:
                    T_var1 = torch.from_numpy(T11).to(device).float()
                    T1_var = torch.from_numpy(T1).to(device).float()
                if args.correction:
                    if not args.proper_marginals:
                        if args.gpu_id != -1:
                            marginals1 = torch.ones(T_var1.shape[0]).to(device) / T_var1.shape[0]
                        else:
                            marginals1 = torch.ones(T_var1.shape[0]).to(device) / T_var1.shape[0]
                        marginals1 = torch.diag(1.0 / (marginals1 + eps))  # take inverse
                        T_var1 = torch.matmul(T_var1, marginals1)
                    else:

                        marginals_beta1 = T_var1.t() @ torch.ones(T_var1.shape[0], dtype=T_var1.dtype).to(device)
                        marginals1 = (1 / (marginals_beta1 + eps))
                        T_var1 = T_var1 * marginals1

                t_fc0_model = torch.matmul(T_var1.t(), fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1))
            else:
                aligned_wt = torch.matmul(fc_layer0_weight.data, T_var1)
                aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                t_fc0_model = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
            geometric_fc = ((1 - args.ensemble_step) * fc_layer0_weight +
                            args.ensemble_step * fc_layer1_weight)
            avg_aligned_layers.append(geometric_fc)
            continue
        assert fc_layer0_weight.shape == fc_layer1_weight.shape

        mu_cardinality = fc_layer0_weight.shape[0]
        nu_cardinality = fc_layer1_weight.shape[0]

        if len(layer_shape) > 2:
            is_conv = True
            is_bias = False
            fc_layer0_weight_data = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], fc_layer0_weight.shape[1], -1)
            fc_layer1_weight_data = fc_layer1_weight.data.view(fc_layer1_weight.shape[0], fc_layer1_weight.shape[1], -1)
        elif len(layer_shape)==2:
            is_conv = False
            is_bias = False
            fc_layer0_weight_data = fc_layer0_weight.data
            fc_layer1_weight_data = fc_layer1_weight.data
        else:
            is_conv = False
            is_bias = True
            fc_layer0_weight_data = fc_layer0_weight.data.view(fc_layer0_weight.shape[0],-1)
            fc_layer1_weight_data = fc_layer1_weight.data.view(fc_layer1_weight.shape[0],-1)
        
        if idx == 0:

            if is_conv:

                M = ground_metric_object.process(fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
                                fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
            else:

                M = ground_metric_object.process(fc_layer0_weight_data, fc_layer1_weight_data)

            aligned_wt = fc_layer0_weight_data
        else:

            if is_conv:
                T_var_conv = T_var.unsqueeze(0).repeat(fc_layer0_weight_data.shape[2], 1, 1)
                aligned_wt = torch.bmm(fc_layer0_weight_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)
                M = ground_metric_object.process(
                    aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                    fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
                )

            else:
                if is_bias:
                    aligned_wt = torch.matmul(T_var.t(), fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1))
                    aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)

                else:

                    if fc_layer0_weight.data.shape[1] != T_var.shape[0]:

                        fc_layer0_unflattened = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], T_var.shape[0], -1).permute(2, 0, 1)
                        aligned_wt = torch.bmm(
                            fc_layer0_unflattened,
                            T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1)
                        ).permute(1, 2, 0)
                        aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                    else:
                        aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)
                
                    M = ground_metric_object.process(aligned_wt, fc_layer1_weight)

            if args.skip_last_layer and idx == (num_layers - 1):
                if args.ensemble_step != 0.5:
                    avg_aligned_layers.append((1 - args.ensemble_step) * aligned_wt +
                                          args.ensemble_step * fc_layer1_weight)
                else:
                    avg_aligned_layers.append((aligned_wt + fc_layer1_weight)/2)
                return avg_aligned_layers

        if not is_bias:


            if args.importance is None or (idx == num_layers - 1):
                mu = get_histogram(args, 0, mu_cardinality, layer0_name)
                nu = get_histogram(args, 1, nu_cardinality, layer1_name)
            else:
                mu = _get_neuron_importance_histogram(args, fc_layer0_weight_data, is_conv)
                nu = _get_neuron_importance_histogram(args, fc_layer1_weight_data, is_conv)
                assert args.proper_marginals
                
            cpuM = M.data.cpu().numpy()
            if idx>=(num_layers - args.layers):
                cpuM = -cpuM
                ensemble_step = args.ensemble_step_diff
            else:
                cpuM = cpuM
                ensemble_step = args.ensemble_step

            if args.exact == 0:
                T = ot.emd(mu, nu, cpuM)
                T1 = ot.emd(mu,nu,-cpuM)
            elif args.exact == 1:
                T, _ = sinkhorn1(cpuM, 1)
                args.correction = False
            else:
                T = ot.bregman.sinkhorn_log(mu, nu, cpuM, reg=args.reg, numItermax=20000)
                TT = ot.bregman.sinkhorn2(mu,nu, cpuM, reg=args.reg, numItermax=20000)
                loss = loss+TT

            if args.gpu_id!=-1:
                T_var = torch.from_numpy(T).to(device).float()
            else:
                T_var = torch.from_numpy(T).to(device).float()

            if args.correction:
                if not args.proper_marginals:
                    if args.gpu_id != -1:
                        marginals = torch.ones(T_var.shape[0]).to(device) / T_var.shape[0]
                    else:
                        marginals = torch.ones(T_var.shape[0]).to(device) / T_var.shape[0]
                    marginals = torch.diag(1.0/(marginals + eps))  # take inverse
                    T_var = torch.matmul(T_var, marginals)
                else:

                    marginals_beta = T_var.t() @ torch.ones(T_var.shape[0], dtype=T_var.dtype).to(device)
                    marginals = (1 / (marginals_beta + eps))
                    T_var = T_var * marginals
            if args.debug:
                if idx == (num_layers - 1):
                    print("there goes the last transport map: \n ", T_var)
                else:
                    print("there goes the transport map at layer {}: \n ".format(idx), T_var)

                print("Ratio of trace to the matrix sum: ", torch.trace(T_var) / torch.sum(T_var))

            setattr(args, 'trace_sum_ratio_{}'.format(layer0_name), (torch.trace(T_var) / torch.sum(T_var)).item())

            if args.past_correction:
                t_fc0_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))
            else:
                t_fc0_model = torch.matmul(T_var.t(), fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1))

        else:
            t_fc0_model = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)

        if ensemble_step != 0.5:
            geometric_fc = ((1-ensemble_step) * t_fc0_model +
                            ensemble_step * fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
        else:
            geometric_fc = (t_fc0_model + fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))/2
        if is_conv and layer_shape != geometric_fc.shape:
            geometric_fc = geometric_fc.view(layer_shape)
        else:
            geometric_fc = geometric_fc.view(layer_shape)

        avg_aligned_layers.append(geometric_fc)


    return avg_aligned_layers,loss


def fuse_single_conv_bn_pair(block1, block2):
    if isinstance(block1, nn.BatchNorm2d) and isinstance(block2, nn.Conv2d):
        m = block1
        conv = block2
        
        bn_st_dict = m.state_dict()
        conv_st_dict = conv.state_dict()

        # BatchNorm params
        eps = m.eps
        mu = bn_st_dict['running_mean']
        var = bn_st_dict['running_var']
        gamma = bn_st_dict['weight']

        if 'bias' in bn_st_dict:
            beta = bn_st_dict['bias']
        else:
            beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

        # Conv params
        W = conv_st_dict['weight']
        if 'bias' in conv_st_dict:
            bias = conv_st_dict['bias']
        else:
            bias = torch.zeros(W.size(0)).float().to(gamma.device)

        denom = torch.sqrt(var + eps)
        b = beta - gamma.mul(mu).div(denom)
        A = gamma.div(denom)
        bias *= A
        A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

        W.mul_(A)
        bias.add_(b)

        conv.weight.data.copy_(W)

        if conv.bias is None:
            conv.bias = torch.nn.Parameter(bias)
        else:
            conv.bias.data.copy_(bias)
            
        return conv
        
    else:
        return False
    
def fuse_bn_recursively(model):
    previous_name = None
    
    for module_name in model._modules:
        previous_name = module_name if previous_name is None else previous_name # Initialization
        
        conv_fused = fuse_single_conv_bn_pair(model._modules[module_name], model._modules[previous_name])
        if conv_fused:
            model._modules[previous_name] = conv_fused
            model._modules[module_name] = nn.Identity()
            
        if len(model._modules[module_name]._modules) > 0:
            fuse_bn_recursively(model._modules[module_name])
            
        previous_name = module_name

    return model

# input_data = {}
# output_data = {}



def get_wassersteinized_layers_modularized_features(args, device, networks, fishers,activations=None, eps=1e-7, test_loader=None):

    avg_aligned_layers = []
    avg_fisher_layers = []
    T_var = None
    ground_metric_object = GroundMetric(args)

    if args.eval_aligned:
        model0_aligned_layers = []

    if args.gpu_id == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu_id))

    params1 = {}
    params2 = {}
    p1 = {}
    p2 = {}
    # print(networks[0].model.named_parameters())
    for n, p in networks[0].model.named_parameters():
        # print(p.shape)
        params1[n] = p
    for n, p in networks[1].model.named_parameters():
        params2[n] = p
    # print(fishers[0])
    idx = 0
    for n, p in fishers[0].items():
        # print(p.shape)
        if idx==0:
            p1[n] = p11['conv10']
        else:
            p1[n] = p11['conv20']
        idx = idx+1
    idx = 0
    for n, p in fishers[1].items():
        if idx==0:
            p2[n] = p22['conv10']
        else:
            p2[n] = p22['conv20']
        idx = idx+1
    # for n, p in networks[0].heads[0].named_parameters():
    #     params1[n] = p
    # for n, p in networks[1].heads[0].named_parameters():
    #     params2[n] = p
    num_layers = len(params1)
    TT = 0
    # num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
    # for idx, ((layer1_name, fc_layer1_weight), (layer0_name, fc_layer0_weight)) in \
    #         enumerate(zip(networks[0].named_parameters(), networks[1].named_parameters())):
    for idx, ((layer0_name), (layer1_name)) in \
            enumerate(zip(params1, params2)):
        fc_layer0_weight = params1[layer0_name]
        fc_layer1_weight = params2[layer1_name]
        # fisher0 = fishers[1].get(layer1_name)
        # fisher1 = fishers[0].get(layer0_name)
        fisher0 = p1[layer0_name]
        fisher1 = p2[layer1_name]
        # print('layer0_name',layer0_name)
        # print('layer1_name',layer1_name)
        assert fc_layer0_weight.shape == fc_layer1_weight.shape
        # Alexanderia
        # print("Previous layer shape is ", previous_layer_shape)
        previous_layer_shape = fc_layer1_weight.shape

        mu_cardinality = fc_layer0_weight.shape[0]
        nu_cardinality = fc_layer1_weight.shape[0]

        layer_shape = fc_layer0_weight.shape

        if len(layer_shape) > 2:
            is_conv = True
            fc_layer0_weight_data = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], fc_layer0_weight.shape[1], -1)
            fc_layer1_weight_data = fc_layer1_weight.data.view(fc_layer1_weight.shape[0], fc_layer1_weight.shape[1], -1)
            fisher0_weight_data = fisher0.data.view(fisher0.shape[0],fisher0.shape[1],-1)
            fisher1_weight_data = fisher1.data.view(fisher1.shape[0],fisher1.shape[1],-1)
        else:
            is_conv = False
            fc_layer0_weight_data = fc_layer0_weight.data
            fc_layer1_weight_data = fc_layer1_weight.data
            fisher0_weight_data = fisher0.data
            fisher1_weight_data = fisher1.data

        if idx == 0:
            '''
            at the first iteration, need to initialize [is_conv] and [aligned_wt]

            also, 
            '''
            if is_conv:
                M = ground_metric_object.process(fisher0_weight_data.view(fisher0_weight_data.shape[0], -1),
                                                 fisher1_weight_data.view(fisher1_weight_data.shape[0], -1))
                M1 = ground_metric_object.process(fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
                                                 fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
            else:
                M = ground_metric_object.process(fisher0_weight_data, fisher1_weight_data)
                
                M1 = ground_metric_object.process(fc_layer0_weight_data, fc_layer1_weight_data)

            aligned_wt = fc_layer0_weight_data
            aligned_fisher = fisher0_weight_data
        else:

            if is_conv:
                T_var_conv = T_var.unsqueeze(0).repeat(fc_layer0_weight_data.shape[2], 1, 1)  
                aligned_wt = torch.bmm(fc_layer0_weight_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)
                aligned_fisher = torch.bmm(fisher0_weight_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)
                M = ground_metric_object.process(
                    aligned_fisher.contiguous().view(aligned_fisher.shape[0], -1),
                    fisher1_weight_data.view(fisher1_weight_data.shape[0], -1)
                )
                M1 = ground_metric_object.process(
                    aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                    fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
                )
            else:

                if fc_layer0_weight.data.shape[1] != T_var.shape[0]:
                    fc_layer0_unflattened = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], T_var.shape[0],
                                                                       -1).permute(2, 0, 1)
                    aligned_wt = torch.bmm(
                        fc_layer0_unflattened,
                        T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1)
                    ).permute(1, 2, 0)
                    aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                    fisher0_unflattened = fisher0.data.view(fisher0.shape[0], T_var.shape[0],
                                                                       -1).permute(2, 0, 1)
                    aligned_fisher = torch.bmm(
                        fisher0_unflattened,
                        T_var.unsqueeze(0).repeat(fisher0_unflattened.shape[0], 1, 1)
                    ).permute(1, 2, 0)
                    aligned_fisher = aligned_fisher.contiguous().view(aligned_fisher.shape[0], -1)
                else:
                    aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)
                    aligned_fisher = torch.matmul(fisher0.data, T_var)
                M = ground_metric_object.process(aligned_fisher, fisher1)
                
                M1 = ground_metric_object.process(aligned_wt, fc_layer1_weight)
            if args.skip_last_layer and idx == (num_layers - 1):
                # Alexanderia
                # print("Simple averaging of last layer weights. NO transport map needs to be computed")
                # if args.ensemble_step != 0.5:
                avg_aligned_layers.append((1 - args.ensemble_step) * aligned_wt +
                                            args.ensemble_step * fc_layer1_weight)
                avg_fisher_layers.append((1 - args.ensemble_step) * aligned_fisher +
                                            args.ensemble_step * fisher1)
                # else:
                #     avg_aligned_layers.append((aligned_fisher + fisher1) / 2)
                #     avg_fisher_layers.append((1 - ensemble_step) * aligned_fisher +
                #                               args.ensemble_step * fisher1)
                return avg_aligned_layers,avg_fisher_layers

        if args.importance is None or (idx == num_layers - 1):
            mu = get_histogram(args, 0, mu_cardinality, layer0_name)
            nu = get_histogram(args, 1, nu_cardinality, layer1_name)
        else:
            # mu = _get_neuron_importance_histogram(args, aligned_wt, is_conv)
            mu = _get_neuron_importance_histogram(args, fc_layer0_weight_data, is_conv)
            nu = _get_neuron_importance_histogram(args, fc_layer1_weight_data, is_conv)
            # Alexanderia

        # cpuM =( args.we*M+(1-args.we)*M1).data.cpu().numpy()
        cpuM = M.data.cpu().numpy()
        print(cpuM.shape)
        if idx>=(num_layers - 1):
            cpuM = -cpuM
            # print('执行')
            ensemble_step = args.ensemble_step_diff
        else:
            cpuM = cpuM
            ensemble_step = args.ensemble_step
        
        if args.exact == 0:
            T = ot.emd(mu, nu, cpuM)
            T1 = ot.emd(mu,nu,-cpuM)
            # TT = ot.bregman.sinkhorn2(mu,nu, cpuM, reg=args.reg, numItermax=20000)
            # TT1 = ot.bregman.sinkhorn2(mu,nu, -cpuM, reg=args.reg, numItermax=20000)
            # loss = loss+TT
        elif args.exact == 1:
            T, _ = sinkhorn1(cpuM, 1)
            args.correction = False
        else:
            T = ot.bregman.sinkhorn_log(mu, nu, cpuM, reg=args.reg, numItermax=20000)
            TT = ot.bregman.sinkhorn2(mu,nu, cpuM, reg=args.reg, numItermax=20000)
            loss = loss+TT
        '''
        [T_var] is updated by [T]
        '''
        # print('TT',TT)
        # print('TT1',TT1)
        # print('='*108)
        
        if args.gpu_id != -1:
            T_var = torch.from_numpy(T).to(device).float()
            T1_var = torch.from_numpy(T1).to(device).float()
            # print('This is T',T_var)
        else:
            T_var = torch.from_numpy(T).to(device).float()
            T1_var = torch.from_numpy(T1).to(device).float()

        if args.correction:
            if not args.proper_marginals:
                if args.gpu_id != -1:
                    # marginals = torch.mv(T_var.t(), torch.ones(T_var.shape[0]).cuda(args.gpu_id))  # T.t().shape[1] = T.shape[0]
                    marginals = torch.ones(T_var.shape[0]).to(device) / T_var.shape[0]
                    marginals1 = torch.ones(T1_var.shape[0]).to(device) / T1_var.shape[0]
                    # print('margin',marginals)
                else:
                    # marginals = torch.mv(T_var.t(),
                    #                      torch.ones(T_var.shape[0]))  # T.t().shape[1] = T.shape[0]
                    marginals = torch.ones(T_var.shape[0]).to(device) / T_var.shape[0]
                    marginals1 = torch.ones(T1_var.shape[0]).to(device) / T1_var.shape[0]
                    # print('margin', marginals)
                marginals = torch.diag(1.0 / (marginals + eps))  # take inverse
                marginals1 = torch.diag(1.0 / (marginals1 + eps))  # take inverse
                # print('margin1', marginals)
                T_var = torch.matmul(T_var, marginals)
                T1_var = torch.matmul(T1_var, marginals1)

            else:
                '''
                [.t()] method returns the transpose
                '''
                # marginals_alpha = T_var @ torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)
                marginals_beta = T_var.t() @ torch.ones(T_var.shape[0], dtype=T_var.dtype).to(device)
                marginals_beta1 = T1_var.t() @ torch.ones(T1_var.shape[0], dtype=T1_var.dtype).to(device)
                marginals = (1 / (marginals_beta + eps))
                marginals1 = (1 / (marginals_beta1 + eps))
                T_var = T_var * marginals
                T_var = T1_var * marginals1

        if args.debug:
            if idx == (num_layers - 1):
                print("there goes the last transport map: \n ", T_var)
            else:
                print("there goes the transport map at layer {}: \n ".format(idx), T_var)

            print("Ratio of trace to the matrix sum: ", torch.trace(T_var) / torch.sum(T_var))

       # print("Here, trace is {} and matrix sum is {} ".format(torch.trace(T_var), torch.sum(T_var)))
        setattr(args, 'trace_sum_ratio_{}'.format(layer0_name), (torch.trace(T_var) / torch.sum(T_var)).item())

        if args.past_correction:

            t_fc0_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))
            t_fisher_model = torch.matmul(T_var.t(), aligned_fisher.contiguous().view(aligned_fisher.shape[0], -1))
        else:
            t_fc0_model = torch.matmul(T_var.t(), fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1))
            t_fisher_model = torch.matmul(T_var.t(), fisher0_weight_data.view(fisher0_weight_data.shape[0], -1))

        # args.ensemble_step = args.ensemble_step*(1-idx/len())


        geometric_fc = ((1 - ensemble_step) * t_fc0_model +
                        ensemble_step * fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
        geometric_fisher = ((1 - ensemble_step) * t_fisher_model +
                        ensemble_step * fisher1_weight_data.view(fisher1_weight_data.shape[0], -1))

        # else:
        #     geometric_fc = (t_fc0_model + fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))/2
        if is_conv and layer_shape != geometric_fc.shape:
            geometric_fc = geometric_fc.view(layer_shape)
            geometric_fisher = geometric_fisher.view(layer_shape)
        avg_aligned_layers.append(geometric_fc)
        avg_fisher_layers.append(geometric_fisher)


    return avg_aligned_layers,avg_fisher_layers


def get_wassersteinized_layers_modularized1(args, device, networks, activations=None, eps=1e-7, test_loader=None):

    avg_aligned_layers = []
    T_var = None
    T1_var = None
    ground_metric_object = GroundMetric(args)

    if args.eval_aligned:
        model0_aligned_layers = []

    if args.gpu_id == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu_id))
    loss = 0
    params1 = {}
    params2 = {}
    # temp1 = deepcopy(networks[1])
    # temp1 = fuse_bn_recursively(temp1)
    for n, p in networks[0].model.named_parameters():
        # print('n',n)
        params1[n] = p
    print('='*108)
    for n, p in networks[1].model.named_parameters():
        params2[n] = p
    num_layers = len(params1)
    TT = 0
    for idx, ((layer0_name), (layer1_name)) in \
            enumerate(zip(params1, params2)):
        fc_layer0_weight = params1[layer0_name]
        fc_layer1_weight = params2[layer1_name]
        assert fc_layer0_weight.shape == fc_layer1_weight.shape
        previous_layer_shape = fc_layer1_weight.shape

        mu_cardinality = fc_layer0_weight.shape[0]
        nu_cardinality = fc_layer1_weight.shape[0]

        layer_shape = fc_layer0_weight.shape

        if len(layer_shape) > 2:
            is_conv = True
            fc_layer0_weight_data = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], fc_layer0_weight.shape[1], -1)
            fc_layer1_weight_data = fc_layer1_weight.data.view(fc_layer1_weight.shape[0], fc_layer1_weight.shape[1], -1)
        else:
            is_conv = False
            fc_layer0_weight_data = fc_layer0_weight.data
            fc_layer1_weight_data = fc_layer1_weight.data

        if idx == 0:
            if is_conv:
                M = ground_metric_object.process(fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
                                                 fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
            else:
                M = ground_metric_object.process(fc_layer0_weight_data, fc_layer1_weight_data)

            aligned_wt = fc_layer0_weight_data
        else:

            if is_conv:
                T_var_conv = T_var.unsqueeze(0).repeat(fc_layer0_weight_data.shape[2], 1, 1)
                aligned_wt = torch.bmm(fc_layer0_weight_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)
                M = ground_metric_object.process(
                    aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                    fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
                )
            else:

                if fc_layer0_weight.data.shape[1] != T_var.shape[0]:
                    fc_layer0_unflattened = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], T_var.shape[0],
                                                                       -1).permute(2, 0, 1)
                    aligned_wt = torch.bmm(
                        fc_layer0_unflattened,
                        T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1)
                    ).permute(1, 2, 0)
                    aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                else:
                    aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)
                M = ground_metric_object.process(aligned_wt, fc_layer1_weight)
            if args.skip_last_layer and idx == (num_layers - 1):
                # Alexanderia
                # print("Simple averaging of last layer weights. NO transport map needs to be computed")
                if ensemble_step != 0.5:
                    avg_aligned_layers.append((1 - ensemble_step) * aligned_wt +
                                              ensemble_step * fc_layer1_weight)
                else:
                    avg_aligned_layers.append((aligned_wt + fc_layer1_weight) / 2)
                return avg_aligned_layers

        if args.importance is None or (idx == num_layers - 1):
            mu = get_histogram(args, 0, mu_cardinality, layer0_name)
            nu = get_histogram(args, 1, nu_cardinality, layer1_name)
        else:
            # mu = _get_neuron_importance_histogram(args, aligned_wt, is_conv)
            mu = _get_neuron_importance_histogram(args, fc_layer0_weight_data, is_conv)
            nu = _get_neuron_importance_histogram(args, fc_layer1_weight_data, is_conv)
            # Alexanderia

        cpuM = M.data.cpu().numpy()
        if idx>=(num_layers - args.layers):
            cpuM = -cpuM
            ensemble_step = args.ensemble_step_diff1
        else:
            cpuM = cpuM
            ensemble_step = args.ensemble_step1
            
        # else:
            # cpuM = cpuM
            # args.ensemble_step = 0.6
            # args.correction = True

        if args.exact == 0:
            T = ot.emd(mu, nu, cpuM)
            T1 = ot.emd(mu,nu,-cpuM)
        elif args.exact == 1:
            T, _ = sinkhorn1(cpuM, 1)
            args.correction = False
        else:
            if idx>=(num_layers - args.layers):
                T = ot.bregman.sinkhorn_log(mu, nu, cpuM, reg=args.reg, numItermax=20000)
            else:
                T = ot.emd(mu, nu, cpuM)
            TT = ot.bregman.sinkhorn2(mu,nu, cpuM, reg=args.reg, numItermax=20000)
            loss = loss+TT
            T1 = ot.emd(mu,nu,-cpuM)
        
        if args.gpu_id != -1:
            T_var = torch.from_numpy(T).to(device).float()
            T1_var = torch.from_numpy(T1).to(device).float()
        else:
            T_var = torch.from_numpy(T).to(device).float()
            T1_var = torch.from_numpy(T1).to(device).float()

        if args.correction:
            if not args.proper_marginals:
                if args.gpu_id != -1:
                    marginals = torch.ones(T_var.shape[0]).to(device) / T_var.shape[0]
                    marginals1 = torch.ones(T1_var.shape[0]).to(device) / T1_var.shape[0]
                else:
                    marginals = torch.ones(T_var.shape[0]).to(device) / T_var.shape[0]
                    marginals1 = torch.ones(T1_var.shape[0]).to(device) / T1_var.shape[0]
                marginals = torch.diag(1.0 / (marginals + eps))  # take inverse
                marginals1 = torch.diag(1.0 / (marginals1 + eps))  # take inverse
                # print('margin1', marginals)
                T_var = torch.matmul(T_var, marginals)
                T1_var = torch.matmul(T1_var, marginals1)

            else:
                marginals_beta = T_var.t() @ torch.ones(T_var.shape[0], dtype=T_var.dtype).to(device)
                marginals_beta1 = T1_var.t() @ torch.ones(T1_var.shape[0], dtype=T1_var.dtype).to(device)
                marginals = (1 / (marginals_beta + eps))
                marginals1 = (1 / (marginals_beta1 + eps))
                T_var = T_var * marginals
                T_var = T1_var * marginals1

        if args.debug:
            if idx == (num_layers - 1):
                print("there goes the last transport map: \n ", T_var)
            else:
                print("there goes the transport map at layer {}: \n ".format(idx), T_var)

            print("Ratio of trace to the matrix sum: ", torch.trace(T_var) / torch.sum(T_var))

        # print("Here, trace is {} and matrix sum is {} ".format(torch.trace(T_var), torch.sum(T_var)))
        setattr(args, 'trace_sum_ratio_{}'.format(layer0_name), (torch.trace(T_var) / torch.sum(T_var)).item())

        if args.past_correction:

            t_fc0_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))
        else:
            t_fc0_model = torch.matmul(T_var.t(), fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1))


        geometric_fc = ((1 - ensemble_step) * t_fc0_model +
                        ensemble_step * fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))

        if is_conv and layer_shape != geometric_fc.shape:
            geometric_fc = geometric_fc.view(layer_shape)
        avg_aligned_layers.append(geometric_fc)


    return avg_aligned_layers, loss


def get_wassersteinized_layers_modularized_ewc(args, device, networks, fishers,activations=None, eps=1e-7, test_loader=None):

    avg_aligned_layers = []
    avg_fisher_layers = []
    T_var = None
    ground_metric_object = GroundMetric(args)

    if args.eval_aligned:
        model0_aligned_layers = []

    if args.gpu_id == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu_id))

    params1 = {}
    params2 = {}
    p1 = {}
    p2 = {}
    for n, p in networks[0].model.named_parameters():
        # print(p.shape)
        params1[n] = p
    for n, p in networks[1].model.named_parameters():
        params2[n] = p
    # print(fishers[0])
    for n, p in fishers[0].items():
        # print(p.shape)
        p1[n] = p
    for n, p in fishers[1].items():
        p2[n] = p
    # for n, p in networks[0].heads[0].named_parameters():
    #     params1[n] = p
    # for n, p in networks[1].heads[0].named_parameters():
    #     params2[n] = p
    num_layers = len(params1)
    TT = 0
    # num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
    # for idx, ((layer1_name, fc_layer1_weight), (layer0_name, fc_layer0_weight)) in \
    #         enumerate(zip(networks[0].named_parameters(), networks[1].named_parameters())):
    for idx, ((layer0_name), (layer1_name)) in \
            enumerate(zip(params1, params2)):
        fc_layer0_weight = params1[layer0_name]
        fc_layer1_weight = params2[layer1_name]
        # fisher0 = fishers[1].get(layer1_name)
        # fisher1 = fishers[0].get(layer0_name)
        fisher0 = p1[layer0_name]
        fisher1 = p2[layer1_name]
        # print('layer0_name',layer0_name)
        # print('layer1_name',layer1_name)
        assert fc_layer0_weight.shape == fc_layer1_weight.shape
        # Alexanderia
        # print("Previous layer shape is ", previous_layer_shape)
        previous_layer_shape = fc_layer1_weight.shape

        mu_cardinality = fc_layer0_weight.shape[0]
        nu_cardinality = fc_layer1_weight.shape[0]

        layer_shape = fc_layer0_weight.shape

        if len(layer_shape) > 2:
            is_conv = True
            fc_layer0_weight_data = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], fc_layer0_weight.shape[1], -1)
            fc_layer1_weight_data = fc_layer1_weight.data.view(fc_layer1_weight.shape[0], fc_layer1_weight.shape[1], -1)
            fisher0_weight_data = fisher0.data.view(fisher0.shape[0],fisher0.shape[1],-1)
            fisher1_weight_data = fisher1.data.view(fisher1.shape[0],fisher1.shape[1],-1)
        else:
            is_conv = False
            fc_layer0_weight_data = fc_layer0_weight.data
            fc_layer1_weight_data = fc_layer1_weight.data
            fisher0_weight_data = fisher0.data
            fisher1_weight_data = fisher1.data

        if idx == 0:
            '''
            at the first iteration, need to initialize [is_conv] and [aligned_wt]

            also, 
            '''
            if is_conv:
                M = ground_metric_object.process(fisher0_weight_data.view(fisher0_weight_data.shape[0], -1),
                                                 fisher1_weight_data.view(fisher1_weight_data.shape[0], -1))
                M1 = ground_metric_object.process(fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
                                                 fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
            else:
                M = ground_metric_object.process(fisher0_weight_data, fisher1_weight_data)
                
                M1 = ground_metric_object.process(fc_layer0_weight_data, fc_layer1_weight_data)

            aligned_wt = fc_layer0_weight_data
            aligned_fisher = fisher0_weight_data
        else:

            if is_conv:
                T_var_conv = T_var.unsqueeze(0).repeat(fc_layer0_weight_data.shape[2], 1, 1)  
                aligned_wt = torch.bmm(fc_layer0_weight_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)
                aligned_fisher = torch.bmm(fisher0_weight_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)
                M = ground_metric_object.process(
                    aligned_fisher.contiguous().view(aligned_fisher.shape[0], -1),
                    fisher1_weight_data.view(fisher1_weight_data.shape[0], -1)
                )
                M1 = ground_metric_object.process(
                    aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                    fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
                )
            else:

                if fc_layer0_weight.data.shape[1] != T_var.shape[0]:
                    fc_layer0_unflattened = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], T_var.shape[0],
                                                                       -1).permute(2, 0, 1)
                    aligned_wt = torch.bmm(
                        fc_layer0_unflattened,
                        T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1)
                    ).permute(1, 2, 0)
                    aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                    fisher0_unflattened = fisher0.data.view(fisher0.shape[0], T_var.shape[0],
                                                                       -1).permute(2, 0, 1)
                    aligned_fisher = torch.bmm(
                        fisher0_unflattened,
                        T_var.unsqueeze(0).repeat(fisher0_unflattened.shape[0], 1, 1)
                    ).permute(1, 2, 0)
                    aligned_fisher = aligned_fisher.contiguous().view(aligned_fisher.shape[0], -1)
                else:
                    aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)
                    aligned_fisher = torch.matmul(fisher0.data, T_var)
                M = ground_metric_object.process(aligned_fisher, fisher1)
                
                M1 = ground_metric_object.process(aligned_wt, fc_layer1_weight)
            if args.skip_last_layer and idx == (num_layers - 1):
                # Alexanderia
                # print("Simple averaging of last layer weights. NO transport map needs to be computed")
                # if args.ensemble_step != 0.5:
                avg_aligned_layers.append((1 - args.ensemble_step) * aligned_wt +
                                            args.ensemble_step * fc_layer1_weight)
                avg_fisher_layers.append((1 - args.ensemble_step) * aligned_fisher +
                                            args.ensemble_step * fisher1)
                # else:
                #     avg_aligned_layers.append((aligned_fisher + fisher1) / 2)
                #     avg_fisher_layers.append((1 - ensemble_step) * aligned_fisher +
                #                               args.ensemble_step * fisher1)
                return avg_aligned_layers,avg_fisher_layers

        if args.importance is None or (idx == num_layers - 1):
            mu = get_histogram(args, 0, mu_cardinality, layer0_name)
            nu = get_histogram(args, 1, nu_cardinality, layer1_name)
        else:
            # mu = _get_neuron_importance_histogram(args, aligned_wt, is_conv)
            mu = _get_neuron_importance_histogram(args, fc_layer0_weight_data, is_conv)
            nu = _get_neuron_importance_histogram(args, fc_layer1_weight_data, is_conv)
            # Alexanderia

        # cpuM =( args.we*M+(1-args.we)*M1).data.cpu().numpy()
        cpuM = M.data.cpu().numpy()

        # if idx>=(num_layers - 1):
        #     cpuM = -cpuM
        #     # print('执行')
        #     ensemble_step = args.ensemble_step_diff
        # else:
        cpuM = cpuM
        ensemble_step = args.ensemble_step
        
        if args.exact == 0:
            T = ot.emd(mu, nu, cpuM)
            T1 = ot.emd(mu,nu,-cpuM)
            # TT = ot.bregman.sinkhorn2(mu,nu, cpuM, reg=args.reg, numItermax=20000)
            # TT1 = ot.bregman.sinkhorn2(mu,nu, -cpuM, reg=args.reg, numItermax=20000)
            # loss = loss+TT
        elif args.exact == 1:
            T, _ = sinkhorn1(cpuM, 1)
            args.correction = False
        else:
            T = ot.bregman.sinkhorn_log(mu, nu, cpuM, reg=args.reg, numItermax=20000)
            TT = ot.bregman.sinkhorn2(mu,nu, cpuM, reg=args.reg, numItermax=20000)
            loss = loss+TT
        '''
        [T_var] is updated by [T]
        '''
        
        if args.gpu_id != -1:
            T_var = torch.from_numpy(T).to(device).float()
            T1_var = torch.from_numpy(T1).to(device).float()
            # print('This is T',T_var)
        else:
            T_var = torch.from_numpy(T).to(device).float()
            T1_var = torch.from_numpy(T1).to(device).float()

        if args.correction:
            if not args.proper_marginals:
                # think of it as m x 1, scaling weights for m linear combinations of points in X
                if args.gpu_id != -1:
                    # marginals = torch.mv(T_var.t(), torch.ones(T_var.shape[0]).cuda(args.gpu_id))  # T.t().shape[1] = T.shape[0]
                    marginals = torch.ones(T_var.shape[0]).to(device) / T_var.shape[0]
                    marginals1 = torch.ones(T1_var.shape[0]).to(device) / T1_var.shape[0]
                    # print('margin',marginals)
                else:
                    # marginals = torch.mv(T_var.t(),
                    #                      torch.ones(T_var.shape[0]))  # T.t().shape[1] = T.shape[0]
                    marginals = torch.ones(T_var.shape[0]).to(device) / T_var.shape[0]
                    marginals1 = torch.ones(T1_var.shape[0]).to(device) / T1_var.shape[0]
                    # print('margin', marginals)
                marginals = torch.diag(1.0 / (marginals + eps))  # take inverse
                marginals1 = torch.diag(1.0 / (marginals1 + eps))  # take inverse
                # print('margin1', marginals)
                T_var = torch.matmul(T_var, marginals)
                T1_var = torch.matmul(T1_var, marginals1)

            else:
                '''
                [.t()] method returns the transpose
                '''
                # marginals_alpha = T_var @ torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)
                marginals_beta = T_var.t() @ torch.ones(T_var.shape[0], dtype=T_var.dtype).to(device)
                marginals_beta1 = T1_var.t() @ torch.ones(T1_var.shape[0], dtype=T1_var.dtype).to(device)
                marginals = (1 / (marginals_beta + eps))
                marginals1 = (1 / (marginals_beta1 + eps))
                T_var = T_var * marginals
                T_var = T1_var * marginals1

        print('T',T_var[0])
        print('+'*108)
        print('T1',T1_var[0])

        if args.debug:
            if idx == (num_layers - 1):
                print("there goes the last transport map: \n ", T_var)
            else:
                print("there goes the transport map at layer {}: \n ".format(idx), T_var)

            print("Ratio of trace to the matrix sum: ", torch.trace(T_var) / torch.sum(T_var))

       # print("Here, trace is {} and matrix sum is {} ".format(torch.trace(T_var), torch.sum(T_var)))
        setattr(args, 'trace_sum_ratio_{}'.format(layer0_name), (torch.trace(T_var) / torch.sum(T_var)).item())

        if args.past_correction:

            t_fc0_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))
            t_fisher_model = torch.matmul(T_var.t(), aligned_fisher.contiguous().view(aligned_fisher.shape[0], -1))
        else:
            t_fc0_model = torch.matmul(T_var.t(), fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1))
            t_fisher_model = torch.matmul(T_var.t(), fisher0_weight_data.view(fisher0_weight_data.shape[0], -1))

        # args.ensemble_step = args.ensemble_step*(1-idx/len())


        geometric_fc = ((1 - ensemble_step) * t_fc0_model +
                        ensemble_step * fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
        geometric_fisher = ((1 - ensemble_step) * t_fisher_model +
                        ensemble_step * fisher1_weight_data.view(fisher1_weight_data.shape[0], -1))

        # else:
        #     geometric_fc = (t_fc0_model + fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))/2
        if is_conv and layer_shape != geometric_fc.shape:
            geometric_fc = geometric_fc.view(layer_shape)
            geometric_fisher = geometric_fisher.view(layer_shape)
        avg_aligned_layers.append(geometric_fc)
        avg_fisher_layers.append(geometric_fisher)


    return avg_aligned_layers,avg_fisher_layers



def print_stats(arr, nick=""):
    print(nick)
    print("summary stats are: \n max: {}, mean: {}, min: {}, median: {}, std: {} \n".format(
        arr.max(), arr.mean(), arr.min(), np.median(arr), arr.std()
    ))


def get_activation_distance_stats(activations_0, activations_1, layer_name=""):
    if layer_name != "":
        print("In layer {}: getting activation distance statistics".format(layer_name))
    M = cost_matrix(activations_0, activations_1) ** (1 / 2)
    mean_dists = torch.mean(M, dim=-1)
    max_dists = torch.max(M, dim=-1)[0]
    min_dists = torch.min(M, dim=-1)[0]
    std_dists = torch.std(M, dim=-1)

    print("Statistics of the distance from neurons of layer 1 (averaged across nodes of layer 0): \n")
    print("Max : {}, Mean : {}, Min : {}, Std: {}".format(torch.mean(max_dists), torch.mean(mean_dists),
                                                          torch.mean(min_dists), torch.mean(std_dists)))
class Namespace():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def ot_weight_align(w: torch.tensor, anchor: torch.tensor, T_var_pre: torch.tensor, device):
    import ot
    layer_shape = w.shape
    args = Namespace(
        ground_metric='euclidean',
        ground_metric_normalize='none',
        reg=0.01,
        debug=False,
        clip_max=5,
        clip_min=0,
        activation_histograms=True,
        dist_normalize=True,
        act_num_samples=200,
        geom_ensemble_type='acts',
        normalize_wts=False,
        clip_gm=False,
        device=device,
        importance=None,
        unbalanced=False,
        ground_metric_eff=False)
    ground_matrix_object = GroundMetric(params=args)
    if len(layer_shape) > 2:
        is_conv = True
        w = w.view(layer_shape[0], layer_shape[1], -1)
        anchor = anchor.view(layer_shape[0], layer_shape[1], -1)
        if T_var_pre is not None:
            T_var_conv = T_var_pre.unsqueeze(0).repeat(w.shape[2], 1, 1)
            aligned_w = torch.bmm(w.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)
        else:
            aligned_w = w
        M = ground_matrix_object.process(
            aligned_w.contiguous().view(aligned_w.shape[0], -1),
            anchor.view(anchor.shape[0], -1))

    else:
        is_conv = False
        if T_var_pre is not None:
            if layer_shape[1] != T_var_pre.shape[0]:
                w_unflattened = w.view(w.shape[0], T_var_pre.shape[0], -1).permute(2, 0, 1)
                aligned_w = torch.bmm(
                    w_unflattened,
                    T_var_pre.unsqueeze(0).repeat(w_unflattened.shape[0], 1, 1)
                ).permute(1, 2, 0)
                aligned_w = aligned_w.contiguous().view(aligned_w.shape[0], -1)
            else:
                aligned_w = torch.matmul(w, T_var_pre)
        else:
            aligned_w = w
        M = ground_matrix_object.process(aligned_w, anchor)
    # mu = _get_neuron_importance_histogram(args, w, is_conv)
    mu = get_histogram(args, 0, w.shape[0], None)
    # nu = _get_neuron_importance_histogram(args, w, is_conv)
    nu = get_histogram(args, 1, w.shape[0], None)
    cpuM = M.data.cpu().numpy()
    T = ot.emd(mu, nu, cpuM)
    T_var_current = torch.from_numpy(T).float()
    marginals = torch.ones(T_var_current.shape[0]) / T_var_current.shape[0]
    marginals = torch.diag(1.0 / (marginals + 1e-7))  # take inverse

    # print(f'device for T_var_current.t() is {T_var_current.t().device}')
    # print(f'device for aligned_w.contiguous().view(aligned_w.shape[0], -1) is {aligned_w.contiguous().view(aligned_w.shape[0], -1).device}')

    aligned_w = torch.matmul(T_var_current.t(), aligned_w.contiguous().view(aligned_w.shape[0], -1))
    aligned_w = aligned_w.view(layer_shape)
    return aligned_w, T_var_current


# def update_model(args, model, new_params, test=False, test_loader=None, reversed=False, idx=-1):
#     updated_model = get_model_from_name(args, idx=idx)
#     if args.gpu_id != -1:
#         updated_model = updated_model.cuda(args.gpu_id)
#
#     layer_idx = 0
#     model_state_dict = model.state_dict()
#
#     # Alexanderia
#     # print("len of model_state_dict is ", len(model_state_dict.items()))
#     print("len of new_params is ", len(new_params))
#
#     for key, value in model_state_dict.items():
#         print("updated parameters for layer ", key)
#         model_state_dict[key] = new_params[layer_idx]
#         layer_idx += 1
#         if layer_idx == len(new_params):
#             break
#
#     updated_model.load_state_dict(model_state_dict)
#
#     if test:
#         log_dict = {}
#         log_dict['test_losses'] = []
#         final_acc = routines.test(args, updated_model, test_loader, log_dict)
#         print("accuracy after update is ", final_acc)
#     else:
#         final_acc = None
#
#     return updated_model, final_acc


def _check_activation_sizes(args, acts0, acts1):
    if args.width_ratio == 1:
        return acts0.shape == acts1.shape
    else:
        return acts0.shape[-1] / acts1.shape[-1] == args.width_ratio


def process_activations(args, activations, layer0_name, layer1_name):  
    activations_0 = activations[0][layer0_name.replace('.' + layer0_name.split('.')[-1], '')].squeeze(1)
    activations_1 = activations[1][layer1_name.replace('.' + layer1_name.split('.')[-1], '')].squeeze(1)

    # assert activations_0.shape == activations_1.shape
    _check_activation_sizes(args, activations_0, activations_1)

    if args.same_model != -1:
        # sanity check when averaging the same model (with value being the model index)
        assert (activations_0 == activations_1).all()
        print("Are the activations the same? ", (activations_0 == activations_1).all())

    if len(activations_0.shape) == 2:
        activations_0 = activations_0.t()
        activations_1 = activations_1.t()
    elif len(activations_0.shape) > 2:
        reorder_dim = [l for l in range(1, len(activations_0.shape))]
        reorder_dim.append(0)
        print("reorder_dim is ", reorder_dim)
        activations_0 = activations_0.permute(*reorder_dim).contiguous()
        activations_1 = activations_1.permute(*reorder_dim).contiguous()

    return activations_0, activations_1


def _reduce_layer_name(layer_name):
    # print("layer0_name is ", layer0_name) It was features.0.weight
    # previous way assumed only one dot, so now I replace the stuff after last dot
    return layer_name.replace('.' + layer_name.split('.')[-1], '')


def _get_layer_weights(layer_weight, is_conv):
    if is_conv:
        # For convolutional layers, it is (#out_channels, #in_channels, height, width)
        layer_weight_data = layer_weight.data.view(layer_weight.shape[0], layer_weight.shape[1], -1)
    else:
        layer_weight_data = layer_weight.data

    return layer_weight_data


def _process_ground_metric_from_acts(args, is_conv, ground_metric_object, activations):
    print("inside refactored")
    if is_conv:
        if not args.gromov:
            M0 = ground_metric_object.process(activations[0].view(activations[0].shape[0], -1),
                                              activations[1].view(activations[1].shape[0], -1))
        else:
            M0 = ground_metric_object.process(activations[0].view(activations[0].shape[0], -1),
                                              activations[0].view(activations[0].shape[0], -1))
            M1 = ground_metric_object.process(activations[1].view(activations[1].shape[0], -1),
                                              activations[1].view(activations[1].shape[0], -1))

        print("# of ground metric features is ", (activations[0].view(activations[0].shape[0], -1)).shape[1])
    else:
        if not args.gromov:
            M0 = ground_metric_object.process(activations[0], activations[1])
        else:
            M0 = ground_metric_object.process(activations[0], activations[0])
            M1 = ground_metric_object.process(activations[1], activations[1])

    if args.gromov:
        return M0, M1
    else:
        return M0, None


def _custom_sinkhorn(args, mu, nu, cpuM):
    if not args.unbalanced:
        if args.sinkhorn_type == 'normal':
            T = ot.bregman.sinkhorn(mu, nu, cpuM, reg=args.reg)
        elif args.sinkhorn_type == 'stabilized':
            T = ot.bregman.sinkhorn_stabilized(mu, nu, cpuM, reg=args.reg)
        elif args.sinkhorn_type == 'epsilon':
            T = ot.bregman.sinkhorn_epsilon_scaling(mu, nu, cpuM, reg=args.reg)
        # elif args.sinkhorn_type == 'gpu':
        #     T, _ = utils.sinkhorn_loss(cpuM, mu, nu, gpu_id=args.gpu_id, epsilon=args.reg, return_tmap=True)
        else:
            raise NotImplementedError
    else:
        T = ot.unbalanced.sinkhorn_knopp_unbalanced(mu, nu, cpuM, reg=args.reg, reg_m=args.reg_m)
    return T


def _sanity_check_tmap(T):
    if not math.isclose(np.sum(T), 1.0, abs_tol=1e-7):
        print("Sum of transport map is ", np.sum(T))
        raise Exception('NAN inside Transport MAP. Most likely due to large ground metric values')


# def _get_updated_acts_v0(args, layer_shape, aligned_wt, model0_aligned_layers, networks, test_loader, layer_names):
#     '''
#     Return the updated activations of the 0th model with respect to the other one.
#
#     :param args:
#     :param layer_shape:
#     :param aligned_wt:
#     :param model0_aligned_layers:
#     :param networks:
#     :param test_loader:
#     :param layer_names:
#     :return:
#     '''
#     if layer_shape != aligned_wt.shape:
#         updated_aligned_wt = aligned_wt.view(layer_shape)
#     else:
#         updated_aligned_wt = aligned_wt
#
#     updated_model0, _ = update_model(args, networks[0], model0_aligned_layers + [updated_aligned_wt], test=True,
#                                      test_loader=test_loader, idx=0)
#     updated_activations = utils.get_model_activations(args, [updated_model0, networks[1]],
#                                                       config=args.config,
#                                                       layer_name=_reduce_layer_name(layer_names[0]), selective=True)
#
#     updated_activations_0, updated_activations_1 = process_activations(args, updated_activations,
#                                                                        layer_names[0], layer_names[1])
#     return updated_activations_0, updated_activations_1
#
#
# def _get_updated_acts_v1(args, networks, test_loader, layer_names):
#     '''
#     Return the updated activations of the 0th model with respect to the other one.
#
#     :param args:
#     :param layer_shape:
#     :param aligned_wt:
#     :param model0_aligned_layers:
#     :param networks:
#     :param test_loader:
#     :param layer_names:
#     :return:
#     '''
#     updated_activations = utils.get_model_activations(args, networks,
#                                                       config=args.config)
#
#     updated_activations_0, updated_activations_1 = process_activations(args, updated_activations,
#                                                                        layer_names[0], layer_names[1])
#     return updated_activations_0, updated_activations_1


def _check_layer_sizes(args, layer_idx, shape1, shape2, num_layers):
    if args.width_ratio == 1:
        return shape1 == shape2
    else:
        if args.dataset == 'mnist':
            if layer_idx == 0:
                return shape1[-1] == shape2[-1] and (shape1[0] / shape2[0]) == args.width_ratio
            elif layer_idx == (num_layers - 1):
                return (shape1[-1] / shape2[-1]) == args.width_ratio and shape1[0] == shape2[0]
            else:
                ans = True
                for ix in range(len(shape1)):
                    ans = ans and shape1[ix] / shape2[ix] == args.width_ratio
                return ans
        elif args.dataset[0:7] == 'Cifar10':
            assert args.second_model_name is not None
            if layer_idx == 0 or layer_idx == (num_layers - 1):
                return shape1 == shape2
            else:
                if (not args.reverse and layer_idx == (num_layers - 2)) or (args.reverse and layer_idx == 1):
                    return (shape1[1] / shape2[1]) == args.width_ratio
                else:
                    return (shape1[0] / shape2[0]) == args.width_ratio


def _compute_marginals(args, T_var, device, eps=1e-7):
    if args.correction:
        if not args.proper_marginals:
            # think of it as m x 1, scaling weights for m linear combinations of points in X
            marginals = torch.ones(T_var.shape)
            if args.gpu_id != -1:
                marginals = marginals.cuda(args.gpu_id)

            marginals = torch.matmul(T_var, marginals)
            marginals = 1 / (marginals + eps)
            print("marginals are ", marginals)

            T_var = T_var * marginals

        else:
            # marginals_alpha = T_var @ torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)
            marginals_beta = T_var.t() @ torch.ones(T_var.shape[0], dtype=T_var.dtype).to(device)

            marginals = (1 / (marginals_beta + eps))
            print("shape of inverse marginals beta is ", marginals_beta.shape)
            print("inverse marginals beta is ", marginals_beta)

            T_var = T_var * marginals
            # i.e., how a neuron of 2nd model is constituted by the neurons of 1st model
            # this should all be ones, and number equal to number of neurons in 2nd model
            print(T_var.sum(dim=0))
            # assert (T_var.sum(dim=0) == torch.ones(T_var.shape[1], dtype=T_var.dtype).to(device)).all()

        print("T_var after correction ", T_var)
        print("T_var stats: max {}, min {}, mean {}, std {} ".format(T_var.max(), T_var.min(), T_var.mean(),
                                                                     T_var.std()))
    else:
        marginals = None

    return T_var, marginals


def _get_current_layer_transport_map(args, mu, nu, M0, M1, idx, layer_shape, eps=1e-7, layer_name=None):
    print('mu', mu)
    print('mu', nu)
    if not args.gromov:
        cpuM = M0.data.cpu().numpy()
        if args.exact:
            T = ot.emd(mu, nu, cpuM)
        else:
            T = _custom_sinkhorn(args, mu, nu, cpuM)

        if args.print_distances:
            ot_cost = np.multiply(T, cpuM).sum()
            print(f'At layer idx {idx} and shape {layer_shape}, the OT cost is ', ot_cost)
            if layer_name is not None:
                setattr(args, f'{layer_name}_layer_{idx}_cost', ot_cost)
            else:
                setattr(args, f'layer_{idx}_cost', ot_cost)
    else:
        cpuM0 = M0.data.cpu().numpy()
        cpuM1 = M1.data.cpu().numpy()

        assert not args.exact
        T = ot.gromov.entropic_gromov_wasserstein(cpuM0, cpuM1, mu, nu, loss_fun=args.gromov_loss, epsilon=args.reg)

    if not args.unbalanced:
        _sanity_check_tmap(T)

    if args.gpu_id != -1:
        T_var = torch.from_numpy(T).cuda(args.gpu_id).float()
    else:
        T_var = torch.from_numpy(T).float()

    if args.tmap_stats:
        print(
            "Tmap stats (before correction) \n: For layer {}, frobenius norm from the joe's transport map is {}".format(
                layer_name, torch.norm(T_var - torch.ones_like(T_var) / torch.numel(T_var), p='fro')
            ))

    print("shape of T_var is ", T_var.shape)
    print("T_var before correction ", T_var)

    return T_var


def _get_neuron_importance_histogram(args, layer_weight, is_conv, eps=1e-9):
    '''
    this function calculates norm of the vectors of the parameters of previous layer neurons
        for example, suppose the returned list is [hist], then
        ` hist[i] = norm of out-parameters for ith neuron in the previous layer `

    [args.importance] determines which norm to calculate

    [args.unbalanced] = False makes the returned list normalized
    '''
    print('shape of layer_weight is ', layer_weight.shape)
    if is_conv:
        '''
        flatten so that kernels in the same in-channel are combined into a vector
        '''
        layer = layer_weight.contiguous().view(layer_weight.shape[0], -1).cpu().numpy()
    else:
        layer = layer_weight.cpu().numpy()

    if args.importance == 'l1':
        importance_hist = np.linalg.norm(layer, ord=1, axis=-1).astype(
            np.float64) + eps
    elif args.importance == 'l2':
        importance_hist = np.linalg.norm(layer, ord=2, axis=-1).astype(
            np.float64) + eps
    else:
        raise NotImplementedError

    if not args.unbalanced:
        importance_hist = (importance_hist / importance_hist.sum())
        print('sum of importance hist is ', importance_hist.sum())
    # assert importance_hist.sum() == 1.0
    return importance_hist


def get_acts_wassersteinized_layers_modularized(args, networks, activations, eps=1e-7, train_loader=None,
                                                test_loader=None):
    '''
    Average based on the activation vector over data samples. Obtain the transport map,
    and then based on which align the nodes and average the weights!
    Like before: two neural networks that have to be averaged in geometric manner (i.e. layerwise).
    The 1st network is aligned with respect to the other via wasserstein distance.
    Also this assumes that all the layers are either fully connected or convolutional *(with no bias)*
    :param networks: list of networks
    :param activations: If not None, use it to build the activation histograms.
    Otherwise assumes uniform distribution over neurons in a layer.
    :return: list of layer weights 'wassersteinized'
    '''

    avg_aligned_layers = []
    T_var = None
    args.handle_skips = False
    args.update_acts = False
    if args.handle_skips:
        skip_T_var = None
        skip_T_var_idx = -1
        residual_T_var = None
        residual_T_var_idx = -1

    marginals_beta = None
    # print(list(networks[0].parameters()))
    previous_layer_shape = None
    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
    ground_metric_object = GroundMetric(args)

    if args.update_acts or args.eval_aligned:
        model0_aligned_layers = []

    if args.gpu_id == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu_id))

    networks_named_params = list(zip(networks[0].named_parameters(), networks[1].named_parameters()))
    idx = 0
    incoming_layer_aligned = True  # for input
    while idx < num_layers:
        ((layer0_name, fc_layer0_weight), (layer1_name, fc_layer1_weight)) = networks_named_params[idx]
        # for idx,  in \
        #         enumerate(zip(network0_named_params, network1_named_params)):
        print("\n--------------- At layer index {} ------------- \n ".format(idx))
        # layer shape is out x in
        # assert fc_layer0_weight.shape == fc_layer1_weight.shape
        assert _check_layer_sizes(args, idx, fc_layer0_weight.shape, fc_layer1_weight.shape, num_layers)
        print("Previous layer shape is ", previous_layer_shape)
        previous_layer_shape = fc_layer1_weight.shape

        # will have shape layer_size x act_num_samples
        layer0_name_reduced = _reduce_layer_name(layer0_name)
        layer1_name_reduced = _reduce_layer_name(layer1_name)

        print("let's see the difference in layer names", layer0_name.replace('.' + layer0_name.split('.')[-1], ''),
              layer0_name_reduced)
        print(activations[0][layer0_name.replace('.' + layer0_name.split('.')[-1], '')].shape,
              'shape of activations generally')
        # for conv layer I need to make the act_num_samples dimension the last one, but it has the intermediate dimensions for
        # height and width of channels, so that won't work.
        # So convert (num_samples, layer_size, ht, wt) -> (layer_size, ht, wt, num_samples)

        activations_0, activations_1 = process_activations(args, activations, layer0_name, layer1_name)

        # print("activations for 1st model are ", activations_0)
        # print("activations for 2nd model are ", activations_1)

        assert activations_0.shape[0] == fc_layer0_weight.shape[0]
        assert activations_1.shape[0] == fc_layer1_weight.shape[0]

        mu_cardinality = fc_layer0_weight.shape[0]
        nu_cardinality = fc_layer1_weight.shape[0]

        get_activation_distance_stats(activations_0, activations_1, layer0_name)

        layer0_shape = fc_layer0_weight.shape
        layer_shape = fc_layer1_weight.shape
        if len(layer_shape) > 2:
            is_conv = True
        else:
            is_conv = False

        fc_layer0_weight_data = _get_layer_weights(fc_layer0_weight, is_conv)
        fc_layer1_weight_data = _get_layer_weights(fc_layer1_weight, is_conv)

        if idx == 0 or incoming_layer_aligned:
            aligned_wt = fc_layer0_weight_data

        else:

            print("shape of layer: model 0", fc_layer0_weight_data.shape)
            print("shape of layer: model 1", fc_layer1_weight_data.shape)

            print("shape of activations: model 0", activations_0.shape)
            print("shape of activations: model 1", activations_1.shape)

            print("shape of previous transport map", T_var.shape)

            # aligned_wt = None, this caches the tensor and causes OOM
            if is_conv:
                if args.handle_skips:
                    assert len(layer0_shape) == 4
                    # save skip_level transport map if there is block ahead
                    if layer0_shape[1] != layer0_shape[0]:
                        if not (layer0_shape[2] == 1 and layer0_shape[3] == 1):
                            print(f'saved skip T_var at layer {idx} with shape {layer0_shape}')
                            skip_T_var = T_var.clone()
                            skip_T_var_idx = idx
                        else:
                            print(
                                f'utilizing skip T_var saved from layer layer {skip_T_var_idx} with shape {skip_T_var.shape}')
                            # if it's a shortcut (128, 64, 1, 1)
                            residual_T_var = T_var.clone()
                            residual_T_var_idx = idx  # use this after the skip
                            T_var = skip_T_var
                        print("shape of previous transport map now is", T_var.shape)
                    else:
                        if residual_T_var is not None and (residual_T_var_idx == (idx - 1)):
                            T_var = (T_var + residual_T_var) / 2
                            print("averaging multiple T_var's")
                        else:
                            print("doing nothing for skips")
                T_var_conv = T_var.unsqueeze(0).repeat(fc_layer0_weight_data.shape[2], 1, 1)
                aligned_wt = torch.bmm(fc_layer0_weight_data.permute(2, 0, 1), T_var_conv).permute(1, 2, 0)

            else:
                if fc_layer0_weight.data.shape[1] != T_var.shape[0]:
                    # Handles the switch from convolutional layers to fc layers
                    # checks if the input has been reshaped
                    fc_layer0_unflattened = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], T_var.shape[0],
                                                                       -1).permute(2, 0, 1)
                    aligned_wt = torch.bmm(
                        fc_layer0_unflattened,
                        T_var.unsqueeze(0).repeat(fc_layer0_unflattened.shape[0], 1, 1)
                    ).permute(1, 2, 0)
                    aligned_wt = aligned_wt.contiguous().view(aligned_wt.shape[0], -1)
                else:
                    aligned_wt = torch.matmul(fc_layer0_weight.data, T_var)

            #### Refactored ####

            if args.update_acts:
                assert args.second_model_name is None
                # activations_0, activations_1 = _get_updated_acts_v0(args, layer_shape, aligned_wt,
                #                                                     model0_aligned_layers, networks,
                #                                                     test_loader, [layer0_name, layer1_name])

        if args.importance is None or (idx == num_layers - 1):
            mu = get_histogram(args, 0, mu_cardinality, layer0_name)
            nu = get_histogram(args, 1, nu_cardinality, layer1_name)
        else:
            # mu = _get_neuron_importance_histogram(args, aligned_wt, is_conv)
            mu = _get_neuron_importance_histogram(args, fc_layer0_weight_data, is_conv)
            nu = _get_neuron_importance_histogram(args, fc_layer1_weight_data, is_conv)
            print(mu, nu)
            assert args.proper_marginals

        if args.act_bug:
            # bug from before (didn't change the activation part)
            # only for reproducing results from previous version
            M0 = ground_metric_object.process(
                aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
            )
        else:
            # debugged part
            print("Refactored ground metric calc")
            M0, M1 = _process_ground_metric_from_acts(args, is_conv, ground_metric_object,
                                                      [activations_0, activations_1])

            print("# of ground metric features in 0 is  ", (activations_0.view(activations_0.shape[0], -1)).shape[1])
            print("# of ground metric features in 1 is  ", (activations_1.view(activations_1.shape[0], -1)).shape[1])

        if args.debug and not args.gromov:
            # bug from before (didn't change the activation part)
            M_old = ground_metric_object.process(
                aligned_wt.contiguous().view(aligned_wt.shape[0], -1),
                fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
            )
            print("Frobenius norm of old (i.e. bug involving wts) and new are ",
                  torch.norm(M_old, 'fro'), torch.norm(M0, 'fro'))
            print("Frobenius norm of difference between ground metric wrt old ",
                  torch.norm(M0 - M_old, 'fro') / torch.norm(M_old, 'fro'))

            print("ground metric old (i.e. bug involving wts) is ", M_old)
            print("ground metric new is ", M0)

        ####################

        if args.same_model != -1:
            print("Checking ground metric matrix in case of same models")
            if not args.gromov:
                print(M0)
            else:
                print(M0, M1)

        if args.skip_last_layer and idx == (num_layers - 1):

            if args.skip_last_layer_type == 'average':
                print("Simple averaging of last layer weights. NO transport map needs to be computed")
                if args.ensemble_step != 0.5:
                    print("taking baby steps (even in skip) ! ")
                    avg_aligned_layers.append((1 - args.ensemble_step) * aligned_wt +
                                              args.ensemble_step * fc_layer1_weight)
                else:
                    avg_aligned_layers.append(((aligned_wt + fc_layer1_weight) / 2))
            elif args.skip_last_layer_type == 'second':
                print("Just giving the weights of the second model. NO transport map needs to be computed")
                avg_aligned_layers.append(fc_layer1_weight)

            return avg_aligned_layers

        print("ground metric (m0) is ", M0)

        T_var = _get_current_layer_transport_map(args, mu, nu, M0, M1, idx=idx, layer_shape=layer_shape, eps=eps,
                                                 layer_name=layer0_name)

        T_var, marginals = _compute_marginals(args, T_var, device, eps=eps)

        if args.debug:
            if idx == (num_layers - 1):
                print("there goes the last transport map: \n ", T_var)
                print("and before marginals it is ", T_var / marginals)
            else:
                print("there goes the transport map at layer {}: \n ".format(idx), T_var)

        print("Ratio of trace to the matrix sum: ", torch.trace(T_var) / torch.sum(T_var))
        print("Here, trace is {} and matrix sum is {} ".format(torch.trace(T_var), torch.sum(T_var)))
        setattr(args, 'trace_sum_ratio_{}'.format(layer0_name), (torch.trace(T_var) / torch.sum(T_var)).item())

        if args.past_correction:
            print("Shape of aligned wt is ", aligned_wt.shape)
            print("Shape of fc_layer0_weight_data is ", fc_layer0_weight_data.shape)

            t_fc0_model = torch.matmul(T_var.t(), aligned_wt.contiguous().view(aligned_wt.shape[0], -1))
        else:
            t_fc0_model = torch.matmul(T_var.t(), fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1))

        # Average the weights of aligned first layers
        if args.ensemble_step != 0.5:
            print("taking baby steps! ")
            geometric_fc = (1 - args.ensemble_step) * t_fc0_model + \
                           args.ensemble_step * fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)
        else:
            geometric_fc = (t_fc0_model + fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1)) / 2
        if is_conv and layer_shape != geometric_fc.shape:
            geometric_fc = geometric_fc.view(layer_shape)
        avg_aligned_layers.append(geometric_fc)

        incoming_layer_aligned = False
        next_aligned_wt_reshaped = None

        # remove cached variables to prevent out of memory
        activations_0 = None
        activations_1 = None
        mu = None
        nu = None
        fc_layer0_weight_data = None
        fc_layer1_weight_data = None
        M0 = None
        M1 = None
        cpuM = None

        idx += 1
    return avg_aligned_layers

#
# def get_network_from_param_list(args, param_list, test_loader):
#     # Alexanderia
#     # print("using independent method")
#     new_network = get_model_from_name(args, idx=1)
#     if args.gpu_id != -1:
#         new_network = new_network.cuda(args.gpu_id)
#
#     # check the test performance of the network before
#     log_dict = {}
#     log_dict['test_losses'] = []
#     # Alexanderia
#     # routines.test(args, new_network, test_loader, log_dict)
#
#     # set the weights of the new network
#     # print("before", new_network.state_dict())
#     # Alexanderia
#     # print("len of model parameters and avg aligned layers is ", len(list(new_network.parameters())),
#     #   len(param_list))
#     assert len(list(new_network.parameters())) == len(param_list)
#
#     layer_idx = 0
#     model_state_dict = new_network.state_dict()
#
#     # Alexanderia
#     # print("len of model_state_dict is ", len(model_state_dict.items()))
#     # Alexanderia
#     # print("len of param_list is ", len(param_list))
#
#     for key, value in model_state_dict.items():
#         model_state_dict[key] = param_list[layer_idx]
#         layer_idx += 1
#
#     new_network.load_state_dict(model_state_dict)
#
#     # check the test performance of the network after
#     log_dict = {}
#     log_dict['test_losses'] = []
#     acc = routines.test(args, new_network, test_loader, log_dict)
#
#     return acc, new_network


# def geometric_ensembling_modularized(args, networks, train_loader, test_loader, activations=None):
#     '''
#     do geometric (namely OT-based) fusion to the models contained in [networks] and return the
#         model accuracy as well as the fused model
#
#     if [geom_ensemble_type] == 'wts', then do the fusion with ground metric being the weights
#
#     else if [geom_ensemble_type] == 'acts', then do the fusion with ground metric being the
#         activations specified in [activations]
#
#     the [train_loader] and [test_loader] provides the data for training and testing, respectively
#     '''
#     if args.geom_ensemble_type == 'wts':
#         avg_aligned_layers = get_wassersteinized_layers_modularized(args, networks, activations,
#                                                                     test_loader=test_loader)
#     elif args.geom_ensemble_type == 'acts':
#         avg_aligned_layers = get_acts_wassersteinized_layers_modularized(args, networks, activations,
#                                                                          train_loader=train_loader,
#                                                                          test_loader=test_loader)
#
#     return get_network_from_param_list(args, avg_aligned_layers, test_loader)

