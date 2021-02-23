#!/bin/python
#-----------------------------------------------------------------------------
# File Name : train_lenet_decolle
# Author: Emre Neftci
#
# Creation Date : Sept 2. 2019
# Last Modified :
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#-----------------------------------------------------------------------------
from decolle.subnetmlp_decolle_bp import subnetmlpDECOLLE, DECOLLELoss, LIFLayerVariableTau, LIFLayerbp
from decolle.utils import parse_args, trainbp, testbp, accuracy, save_checkpoint, load_model_from_checkpoint, prepare_experiment, write_stats, cross_entropy_one_hot
import datetime, os, socket, tqdm
from loader_tests import create_dvsgestures_attn
import numpy as np
import torch
import importlib
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
np.set_printoptions(precision=4)
args = parse_args('parameters/params_dvsgestures_subnetmlp.yml')
device = args.device


starting_epoch = 0

params, writer, dirs = prepare_experiment(name=__file__.split('/')[-1].split('.')[0], args = args)
log_dir = dirs['log_dir']
checkpoint_dir = dirs['checkpoint_dir']

dataset = importlib.import_module(params['dataset'])
try:
    create_data = dataset.create_data
except AttributeError:
    create_data = dataset.create_dataloader

verbose = args.verbose

## Load Data
gen_train, gen_test = create_data(chunk_size_train=params['chunk_size_train'],
                                  chunk_size_test=params['chunk_size_test'],
                                  batch_size=params['batch_size'],
                                  dt=params['deltat'],
                                  num_workers=params['num_dl_workers'])

data_batch, target_batch = next(iter(gen_train))
data_batch = torch.Tensor(data_batch).to(device)
target_batch = torch.Tensor(target_batch).to(device)

#d, t = next(iter(gen_train))
input_shape = data_batch.shape[-3:]

#Backward compatibility
if 'dropout' not in params.keys():
    params['dropout'] = [.5]

## Create Model, Optimizer and Loss
net = subnetmlpDECOLLE( out_channels=params['out_channels'],
                    Nhid=params['Nhid'],
                    Mhid=params['Mhid'],
                    kernel_size=params['kernel_size'],
                    pool_size=params['pool_size'],
                    input_shape=params['input_shape'],
                    alpha=params['alpha'],
                    alpharp=params['alpharp'],
                    dropout=params['dropout'],
                    beta=params['beta'],
                    num_conv_layers=params['num_conv_layers'],
                    num_mlp_layers=params['num_mlp_layers'],
                    lc_ampl=params['lc_ampl'],
                    lif_layer_type = LIFLayerbp,
                    method=params['learning_method'],
                    ).to(device)

print('layers',[[k, v.shape] for k, v in net.name_param()])
print('trainable', [k for k in net.get_trainable_named_parameters()])

opt = torch.optim.Adamax(net.get_trainable_parameters(), lr=params['learning_rate'], betas=params['betas'])


loss = torch.nn.SmoothL1Loss()

##Initialize
#net.init_parameters(data_batch)

##Resume if necessary
if args.resume_from is not None:
    print("Checkpoint directory " + checkpoint_dir)
    if not os.path.exists(checkpoint_dir) and not args.no_save:
        os.makedirs(checkpoint_dir)
    starting_epoch = load_model_from_checkpoint(checkpoint_dir, net, opt)
    print('Learning rate = {}. Resumed from checkpoint'.format(opt.param_groups[-1]['lr']))

# Printing parameters
if args.verbose:
    print('Using the following parameters:')
    m = max(len(x) for x in params)
    for k, v in zip(params.keys(), params.values()):
        print('{}{} : {}'.format(k, ' ' * (m - len(k)), v))

print('\n------Starting training with {} DECOLLE layers-------'.format(len(net)))

# --------TRAINING LOOP----------
if not args.no_train:
    test_acc_hist = []
    for e in range(starting_epoch , params['num_epochs'] ):
        interval = e // params['lr_drop_interval']
        lr = opt.param_groups[-1]['lr']
        if interval > 0:
            print('Changing learning rate from {} to {}'.format(lr, opt.param_groups[-1]['lr']))
            opt.param_groups[-1]['lr'] = np.array(params['learning_rate']) / (interval * params['lr_drop_factor'])
        else:
            print('Changing learning rate from {} to {}'.format(lr, opt.param_groups[-1]['lr']))
            opt.param_groups[-1]['lr'] = np.array(params['learning_rate'])

        if (e % params['test_interval']) == 0 and e != 0:
            print('---------------Epoch {}-------------'.format(e))
            if not args.no_save:
                print('---------Saving checkpoint---------')
                save_checkpoint(e, checkpoint_dir, net, opt)

            test_loss, test_acc, spikes = testbp(gen_test, loss, net, print_error = True)
            test_acc_hist.append(test_acc)

            if not args.no_save:
                write_stats(e, test_acc, test_loss, spikes, writer)
                np.save(log_dir+'/test_acc.npy', np.array(test_acc_hist),)

        # total_loss, act_rate = trainbp(gen_train, loss, net, opt, e,  online_update= False)     #has bug in online update
        '''
            Trains a DECOLLE network

            Arguments:
            gen_train: a dataloader
            decolle_loss: a DECOLLE loss function, as defined in base_model
            net: DECOLLE network
            opt: optimizer
            epoch: epoch number, for printing purposes only
            burnin: time during which the dynamics will be run, but no updates are made
            online_update: whether updates should be made at every timestep or at the end of the sequence.
            '''
        device = net.get_input_layer_device()
        iter_gen_train = iter(gen_train)

        act_rate = 0
        total_loss = np.zeros(1)
        s_total = 0
        loss_tv = torch.tensor(0.).to(device)
        net.train()
        if hasattr(net.LIF_layers[0], 'base_layer'):
            dtype = net.LIF_layers[0].base_layer.weight.dtype
        else:
            dtype = net.LIF_layers[0].weight.dtype
        batch_iter = 0
        with torch.autograd.set_detect_anomaly(True):
            for data_batch, target_batch in tqdm.tqdm(iter_gen_train, desc='Epoch {}'.format(epoch)):

                data_batch = torch.Tensor(data_batch).type(dtype).to(device)
                target_batch = torch.Tensor(target_batch).type(dtype).to(device)
                if len(target_batch.shape) == 2:
                    # print('replicate targets for all timesteps')
                    target_batch = target_batch.unsqueeze(1)
                    shape_with_time = np.array(target_batch.shape)
                    shape_with_time[1] = data_batch.shape[1]
                    target_batch = target_batch.expand(*shape_with_time)

                loss_mask = (target_batch.sum(2) > 0).unsqueeze(2).float()
                # print('loss mask',loss_mask.shape, loss_mask)
                # loss_mask = (data_batch.reshape(data_batch.shape[0],data_batch.shape[1],-1).mean(2)>0.01).unsqueeze(2).float()
                # net.init(data_batch, burnin)
                t_sample = data_batch.shape[1]
                for k in (range(t_sample)):
                    # print('datautil',data_batch.shape)
                    s, u = net.forward(data_batch[:, k, :, :])
                    # print('s, tar',s.shape, target_batch[:, k, :].shape,s[0],target_batch[:, k, :][0])
                    loss_ = loss(s, target=target_batch[:, k, :])
                    # print('loss_', loss_,)
                    loss_tv = loss_.clone() + loss_tv
                    ss = s.clone()
                    s_total += sum(ss.detach().cpu().numpy())

                    total_loss += loss_.clone().detach().cpu().numpy()

                    if online_update:  # cannot be used in bp
                        opt.zero_grad()
                        loss_.backward(retain_graph=True)
                        opt.step()

                        act_rate += s_total / t_sample
                    # act_rate += tonp(s_total.mean().data) / t_sample

                if not online_update:
                    opt.zero_grad()
                    loss_tv.backward(retain_graph=True)
                    opt.step()

                    act_rate += s_total / t_sample

                batch_iter += 1
                if batches_per_epoch > 0:
                    if batch_iter >= batches_per_epoch: break

            total_loss /= t_sample
            print('Loss {0}'.format(total_loss))
            print('Activity Rate {0}'.format(act_rate))
        if not args.no_save:
            for i in range(len(net)):
                writer.add_scalar('/act_rate/{0}'.format(i), act_rate[i], e)
                writer.add_scalar('/learning_rate/{0}'.format(i), opt.param_groups[-1]['lr'], e)
