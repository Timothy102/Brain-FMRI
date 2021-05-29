# -*- coding: utf-8 -*-


import torch
import torch.nn as nn

from Code.backbones import res2net, vggnet
import torch
import torch.nn as nn
import torch.nn.functional as F


from torchsummary import summary


class _GridAttentionBlockND(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3, mode='concatenation',
                 sub_sample_factor=(2,2,2)):
        super(_GridAttentionBlockND, self).__init__()

        assert dimension in [2, 3]
        assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple): self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list): self.sub_sample_factor = tuple(sub_sample_factor)
        else: self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplemented

        # Output transform
        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0, bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        # Initialise weights
        for m in self.children():
            init_weights(m, init_type='kaiming')

        # Define the operation
        if mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'concatenation_debug':
            self.operation_function = self._concatenation_debug
        elif mode == 'concatenation_residual':
            self.operation_function = self._concatenation_residual
        else:
            raise NotImplementedError('Unknown operation function.')


    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''

        output = self.operation_function(x, g)
        return output

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(4, 16, kernel_size=2, stride = 1),
            nn.ELU(True),
            nn.MaxUnpool2d(2),
            nn.BatchNorm2d(32),
            ## dodaj attention gate

            nn.ConvTranspose2d(16, 3, kernel_size=2, stride = 1),
            nn.ELU(True),
            nn.MaxUnpool2d(2),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(3, 64, kernel_size=2, stride = 1),
            nn.ELU(True),
            nn.MaxUnpool2d(2),
            nn.BatchNorm2d(128),
        )
    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential (
        # First Convolutional Block
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),  # 1/2
        nn.BatchNorm2d(512),

        # Second Convolutional Block
        nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),  # 1/4
        nn.BatchNorm2d(64),

        # Third Convolutional Block
        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),  # 1/8
        nn.BatchNorm2d(16),

        # Fourth Convolutional Block
        nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),  #1/16
        nn.BatchNorm2d(2)
    )
    def forward(self, x):
        return self.layers(x)



class Att_Autoencoder(nn.Module):
    ''' This is the ultimate network '''
    def __init__(self, gated_threshold = 0.1):
        super(Att_Autoencoder, self).__init__()
        self.gated_threshold = gated_threshold

        self.attention_gate = _GridAttentionBlockND()
        self.decoder = Decoder()
        self.encoder = Encoder()

    def forward(self, x):
        x1 = self.attention_gate(x)
        x1 = self.encoder(x1)
        output, _ = self.decoder(x1)
        return output
