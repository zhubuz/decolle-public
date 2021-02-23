from decolle.lenet_delle import *
from decolle.utils import parse_args, train, test, save_checkpoint, load_model_from_checkpoint, prepare_experiment, write_stats, cross_entropy_one_hot
import os
import numpy as np
import torch
import importlib

np.set_printoptions(precision=4)
args = parse_args('parameters/params.yml')
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

input_shape = data_batch.shape[-3:]

net = LenetDELLE(out_channels=params['out_channels'],
           Nhid=params['Nhid'],
           Mhid=params['Mhid'],
           kernel_size=params['kernel_size'],
           pool_size=params['pool_size'],
           input_shape=params['input_shape'],
           num_conv_layers=params['num_conv_layers'],
           num_mlp_layers=params['num_mlp_layers'],
           lc_ampl=params['lc_ampl'],
           method=params['learning_method']).to(device)

if hasattr(params['learning_rate'], '__len__'):
    from decolle.utils import MultiOpt
    opts = []
    for i in range(len(params['learning_rate'])):
        opts.append(torch.optim.Adamax(net.get_trainable_parameters(i), lr=params['learning_rate'][i], betas=params['betas']))
    opt = MultiOpt(*opts)
else:
    opt = torch.optim.Adamax(net.get_trainable_parameters(), lr=params['learning_rate'], betas=params['betas'])
print('opt',opt)
reg_l = params['reg_l'] if 'reg_l' in params else None

# if 'loss_scope' in params and params['loss_scope']=='crbp':
#     from decolle.lenet_decolle_model import CRBPLoss
#     loss = torch.nn.SmoothL1Loss(reduction='none')
#     decolle_loss = CRBPLoss(net = net, loss_fn = loss, reg_l=reg_l)
# else:
#     loss = [torch.nn.SmoothL1Loss() for i in range(len(net))]
#     loss[-1] = cross_entropy_one_hot
#     decolle_loss = DECOLLELoss(net = net, loss_fn = loss, reg_l=reg_l)


loss = [torch.nn.MSELoss() for i in range(len(net))]
decolle_loss = DECOLLELoss(net = net, loss_fn = loss, reg_l=None)

if args.resume_from is not None:
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    starting_epoch = load_model_from_checkpoint(checkpoint_dir, net, opt)
    print('Learning rate = {}. Resumed from checkpoint'.format(opt.param_groups[-1]['lr']))

print('\n------Starting training with {} DELLE layers-------'.format(len(net)))

# --------TRAINING LOOP----------
test_acc_hist = []
for e in range(starting_epoch , params['num_epochs'] ):
    interval = e // params['lr_drop_interval']
    lr = opt.param_groups[-1]['lr']
    if interval > 0:
        opt.param_groups[-1]['lr'] = params['learning_rate'] / (interval * params['lr_drop_factor'])
    else:
        opt.param_groups[-1]['lr'] = params['learning_rate']
    if lr != opt.param_groups[-1]['lr']:
        print('Changing learning rate from {} to {}'.format(lr, opt.param_groups[-1]['lr']))

    if (e % params['test_interval']) == 0 and e!=0:
        print('---------------Epoch {}-------------'.format(e))
        if not args.no_save:
            print('---------Saving checkpoint---------')
            save_checkpoint(e, checkpoint_dir, net, opt)

        test_loss, test_acc, spike= test(gen_test, decolle_loss, net, params['burnin_steps'], print_error = True)

        test_acc_hist.append(test_acc)

        if not args.no_save:
            write_stats(e, test_acc, test_loss,spike, writer)

        print("test_acc:")
        print(test_acc)
        print("\n\n")
        np.save(log_dir+'/test_acc.npy', np.array(test_acc_hist),)
        
    total_loss, act_rate = train(gen_train, decolle_loss, net, opt, e, params['burnin_steps'], online_update = params['online_update'])
    for i in range(len(net)):
        writer.add_scalar('/act_rate/{0}'.format(i), act_rate[i], e)
