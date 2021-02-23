#!/bin/python
#-----------------------------------------------------------------------------
# File Name : allconv_decolle.py
# Author: Emre Neftci
#
# Creation Date : Wed 07 Aug 2019 07:00:31 AM PDT
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
#----------------------------------------------------------------------------- 
from .base_model import *

class subnetmlpDECOLLE(DECOLLEBase):

    def __init__(self,
                 input_shape,
                 Nhid=[1],
                 Mhid=[128],
                 out_channels=1,
                 kernel_size=[7],
                 stride=[1],
                 pool_size=[2],
                 alpha=[.9],
                 beta=[.85],
                 alpharp=[.65],
                 dropout=[0.5],
                 num_conv_layers=2,
                 num_mlp_layers=1,
                 deltat=1000,
                 lc_ampl=.5,
                 lif_layer_type = LIFLayerbp,
                 method='rtrl',
                ):

        num_layers = 6
        # If only one value provided, then it is duplicated for each layer
        if len(kernel_size) == 1:   kernel_size = kernel_size * num_conv_layers
        if stride is None: stride=[1]
        if len(stride) == 1:        stride = stride * num_conv_layers
        if pool_size is None: pool_size = [1]
        if len(pool_size) == 1: pool_size = pool_size * num_conv_layers
        if len(alpha) == 1:         alpha = alpha * num_layers
        if len(alpharp) == 1:       alpharp = alpharp * num_layers
        if len(beta) == 1:          beta = beta * num_layers
        if not hasattr(dropout, '__len__'): dropout = [dropout]
        if len(dropout) == 1:       self.dropout = dropout = dropout * num_layers
        if Nhid is None:          self.Nhid = Nhid = []
        if Mhid is None:          self.Mhid = Mhid = []


        super(subnetmlpDECOLLE, self).__init__()

        base_layer1_1 = nn.Linear(16*16, 64)
        l1_1 = lif_layer_type(base_layer1_1,
                               alpha=alpha[0],
                               beta=beta[0],
                               alpharp=alpharp[0],
                               deltat=deltat,
                               do_detach= False)



        base_layer1_2 = nn.Linear(16 * 16, 64)
        l1_2 = lif_layer_type(base_layer1_2,
                                  alpha=alpha[0],
                                  beta=beta[0],
                                  alpharp=alpharp[0],
                                  deltat=deltat,
                                  do_detach=False)


        base_layer1_3 = nn.Linear(16 * 16, 64)
        l1_3 = lif_layer_type(base_layer1_3,
                                  alpha=alpha[0],
                                  beta=beta[0],
                                  alpharp=alpharp[0],
                                  deltat=deltat,
                                  do_detach= False)



        base_layer1_4 = nn.Linear(16 * 16, 64)
        l1_4 = lif_layer_type(base_layer1_4,
                                  alpha=alpha[0],
                                  beta=beta[0],
                                  alpharp=alpharp[0],
                                  deltat=deltat,
                                  do_detach= False)


        base_layer2 = nn.Linear(256, 256)
        l2 = lif_layer_type(base_layer2,
                                  alpha=alpha[0],
                                  beta=beta[0],
                                  alpharp=alpharp[0],
                                  deltat=deltat,
                                  do_detach= False)

        base_layer3 = nn.Linear(256, 11)
        l3 = lif_layer_type(base_layer3,
                                  alpha=alpha[0],
                                  beta=beta[0],
                                  alpharp=alpharp[0],
                                  deltat=deltat,
                                  do_detach= False)
        self.LIF_layers.append(l1_1)
        self.LIF_layers.append(l1_2)
        self.LIF_layers.append(l1_3)
        self.LIF_layers.append(l1_4)
        self.LIF_layers.append(l2)
        self.LIF_layers.append(l3)
    def forward(self, input):
        input = input[:, 0, :, :]  # remove polarity

        input0 = input[:, 0::2, 0::2].reshape((input.shape[0], -1))
        input1 = input[:, 0::2, 1::2].reshape((input.shape[0], -1))
        input2 = input[:, 1::2, 0::2].reshape((input.shape[0], -1))
        input3 = input[:, 1::2, 1::2].reshape((input.shape[0], -1))

        s1_1, u1_1 = self.LIF_layers[0](input0)
        s1_2, u1_2 = self.LIF_layers[1](input1)
        s1_3, u1_3 = self.LIF_layers[2](input2)
        s1_4, u1_4 = self.LIF_layers[3](input3)

        s_cat = torch.cat((s1_1, s1_2, s1_3, s1_4), dim=1)
        #u_cat = torch.cat((u1_1, u1_2, u1_3, u1_4), dim=1)
        s2, u2 =  self.LIF_layers[4](s_cat)
        #s2 = self.LIF_layers[4].sg_function(u2)
        s3, u3= self.LIF_layers[5](s2)
        # print('s3,u3', s3.shape, u3.shape)

        return s3, u3

class TimeWrappedLenetDECOLLE(subnetmlpDECOLLE):
    def forward(self, Sin):
        t_sample = Sin.shape[1]
        out = []
        for t in (range(0,t_sample)):
            Sin_t = Sin[:,t]
            out.append(super().forward(Sin_t))
        return out

    def init(self, data_batch, burnin):
        '''
        Necessary to reset the state of the network whenever a new batch is presented
        '''
        if self.requires_init is False:
            return
        for l in self.LIF_layers:
            l.state = None
        with torch.no_grad():
            self.forward(data_batch[:, burnin:])

    def init_parameters(self, data_batch):
        Sin = data_batch[:, :, :, :]
        s_out = self.forward(Sin)[0][0]
        ins = [self.LIF_layers[0].state.Q]+s_out
        for i,l in enumerate(self.LIF_layers):
            l.init_parameters(ins[i])
    
    
    
if __name__ == "__main__":
    #Test building network
    net = subnetmlpDECOLLE(Nhid=[1,8],Mhid=[32,64],out_channels=10, input_shape=[1,28,28])
    d = torch.zeros([1,1,28,28])
    net(d)
