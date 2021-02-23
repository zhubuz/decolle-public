#!/bin/python
# -----------------------------------------------------------------------------
# File Name : allconv_decolle.py
# Author: Emre Neftci
#
# Creation Date : Wed 07 Aug 2019 07:00:31 AM PDT
# Last Modified : 
#
# Copyright : (c) UC Regents, Emre Neftci
# Licence : GPLv2
# -----------------------------------------------------------------------------
from .base_model import *


class mlpDECOLLE(DECOLLEBase):

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
                 lif_layer_type=LIFLayer,
                 method='rtrl',
                 with_output_layer=False):

        self.with_output_layer = with_output_layer
        if with_output_layer:
            Mhid += [out_channels]
            num_mlp_layers += 1
        self.num_layers = num_layers = num_conv_layers + num_mlp_layers
        # If only one value provided, then it is duplicated for each layer
        if len(kernel_size) == 1:   kernel_size = kernel_size * num_conv_layers
        if stride is None: stride = [1]
        if len(stride) == 1:        stride = stride * num_conv_layers
        if pool_size is None: pool_size = [1]
        if len(pool_size) == 1: pool_size = pool_size * num_conv_layers
        if len(alpha) == 1:         alpha = alpha * num_layers
        if len(alpharp) == 1:       alpharp = alpharp * num_layers
        if len(beta) == 1:          beta = beta * num_layers
        if not hasattr(dropout, '__len__'): dropout = [dropout]
        if len(dropout) == 1:       self.dropout = dropout * num_layers
        if Nhid is None:          self.Nhid = Nhid = []
        if Mhid is None:          self.Mhid = Mhid = []

        super(mlpDECOLLE, self).__init__()

        l1, ro1 = self.decolle_linear(inF=16 * 16, outF=256, out_channels=11,
                                          alpha=alpha[0],
                                          beta=beta[0],
                                          alpharp=alpharp[0],
                                          deltat=deltat,
                                          lc_ampl=lc_ampl,
                                          lif_layer_type=LIFLayer,
                                          weightScale=150,
                                          base = 'dense',
                                          do_detach=True if method == 'rtrl' else False)

        l2, ro2 = self.decolle_linear(inF=256, outF=256, out_channels=11,
                                      alpha=alpha[0],
                                      beta=beta[0],
                                      alpharp=alpharp[0],
                                      deltat=deltat,
                                      lc_ampl=lc_ampl,
                                      lif_layer_type=LIFLayer,
                                      weightScale=150,
                                      base='dense',
                                      do_detach=True if method == 'rtrl' else False)

        # l3, ro3 = self.decolle_linear(inF=256, outF=11, out_channels=11,
        #                                         alpha=alpha[0],
        #                                         beta=beta[0],
        #                                         alpharp=alpharp[0],
        #                                         deltat=deltat,
        #                                         lc_ampl=lc_ampl,
        #                                         lif_layer_type=LIFLayer,
        #                                         do_detach=True if method == 'rtrl' else False,
        #                                         is_output_layer=True)
        self.LIF_layers.append(l1)
        self.LIF_layers.append(l2)
        # self.LIF_layers.append(l3)
        self.readout_layers.append(ro1)
        self.readout_layers.append(ro2)
        # self.readout_layers.append(ro3)

    def forward(self, input):
        s_out = []
        u_out = []
        r_out = []

        input = input[:, 0, :, :]  # remove polarity
        #print('input', input.shape)
        input = input.reshape((input.shape[0], -1))
        s1, u1 = self.LIF_layers[0](input)
        s1 = self.LIF_layers[0].sg_function(u1)
        r1 = self.readout_layers[0](s1)
        # print('ro1_1', r1_1)
        s_out.append(s1)
        u_out.append(u1)
        r_out.append(r1)

        s2, u2 = self.LIF_layers[1](s1)
        s2 = self.LIF_layers[1].sg_function(u2)
        # s2 = sigmoid(u2)
        r2 = self.readout_layers[1](s2)
        s_out.append(s2)
        u_out.append(u2)
        r_out.append(r2)

        # s3, u3, = self.LIF_layers[5](s2)
        # s3 = self.LIF_layers[3].sg_function(u3)
        # # s3 = sigmoid(u3)
        # r3 = self.readout_layers[5](s3)
        # s_out.append(s3)
        # u_out.append(u3)
        # r_out.append(r3)

        return s_out, r_out, u_out


class TimeWrappedLenetDECOLLE(mlpDECOLLE):
    def forward(self, Sin):
        t_sample = Sin.shape[1]
        out = []
        for t in (range(0, t_sample)):
            Sin_t = Sin[:, t]
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
        ins = [self.LIF_layers[0].state.Q] + s_out
        for i, l in enumerate(self.LIF_layers):
            l.init_parameters(ins[i])


if __name__ == "__main__":
    # Test building network
    net = mlpDECOLLE(Nhid=[1, 8], Mhid=[32, 64], out_channels=10, input_shape=[1, 28, 28])
    d = torch.zeros([1, 1, 28, 28])
    net(d)
