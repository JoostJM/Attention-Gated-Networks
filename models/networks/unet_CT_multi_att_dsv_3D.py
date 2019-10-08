import torch.nn as nn
import torch
from .utils import UnetConv3, UnetUp3_CT, UnetGridGatingSignal3, UnetDsv3
import torch.nn.functional as F
from models.networks_other import init_weights
from models.layers.grid_attention_layer import GridAttentionBlock3D


class unet_CT_multi_att_dsv_3D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3,
                 nonlocal_mode='concatenation', attention_dsample=(2,2,2), is_batchnorm=True):
        super(unet_CT_multi_att_dsv_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        self.block_gpu_ids = {}

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.gating = UnetGridGatingSignal3(filters[4], filters[4], kernel_size=(1, 1, 1), is_batchnorm=self.is_batchnorm)

        # attention blocks
        self.attentionblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock4 = MultiAttentionBlock(in_size=filters[3], gate_size=filters[4], inter_size=filters[3],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes, scale_factor=8)
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes, scale_factor=4)
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes, scale_factor=2)
        self.dsv1 = nn.Conv3d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)

        # final conv (without any concat)
        self.final = nn.Conv3d(n_classes*4, n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        if 1 in self.block_gpu_ids:  # Move from 1 to 2
            maxpool1.cuda(self.block_gpu_ids[1][1])

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        if 2 in self.block_gpu_ids:  # Move from 2 to 3
            maxpool2 = maxpool2.cuda(self.block_gpu_ids[2][1])

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        if 3 in self.block_gpu_ids:  # Move from 3 to 4
            maxpool3 = maxpool3.cuda(self.block_gpu_ids[3][1])

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        gating = self.gating(center)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4 = self.up_concat4(g_conv4, center)

        if 3 in self.block_gpu_ids:  # Move from 4 to 3
            up4 = up4.cuda(self.block_gpu_ids[3][0])

        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3 = self.up_concat3(g_conv3, up4)

        if 2 in self.block_gpu_ids:  # Move from 3 to 2
            up3 = up3.cuda(self.block_gpu_ids[2][0])

        g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2 = self.up_concat2(g_conv2, up3)

        if 1 in self.block_gpu_ids:  # Move from 2 to 1
            up2 = up2.cuda(self.block_gpu_ids[1][0])

        up1 = self.up_concat1(conv1, up2)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)

        if 4 in self.block_gpu_ids:  # Move to block 5 (final block)
            dsv4 = dsv4.cuda(self.block_gpu_ids[4])
            dsv3 = dsv3.cuda(self.block_gpu_ids[4])
            dsv2 = dsv2.cuda(self.block_gpu_ids[4])
            dsv1 = dsv1.cuda(self.block_gpu_ids[4])

        final = self.final(torch.cat([dsv1,dsv2,dsv3,dsv4], dim=1))

        return final

    def cuda(self, device=None):
        net = super(unet_CT_multi_att_dsv_3D, self).cuda(device)
        self.block_gpu_ids = {}
        return net

    def split_net(self, devices):
        if len(devices) == 1:
            self.cuda(devices[0])
            return
        elif len(devices) == 2:  # Split Block #2 and #3
            block_devices = {
                1: devices[0],
                2: devices[0],
                3: devices[1],
                4: devices[1],
                5: devices[1]
            }
            self.block_gpu_ids = {
                2: (devices[0], devices[1]),  # Block #2 <-> #3
                4: devices[1]  # Move to Block #4 (final block)
            }
        elif len(devices) == 3:  # Split Block #1 and #2
            block_devices = {
                1: devices[0],
                2: devices[1],
                3: devices[2],
                4: devices[2],
                5: devices[2]
            }
            self.block_gpu_ids = {
                1: (devices[0], devices[1]),  # Block #1 <-> #2
                2: (devices[1], devices[2]),  # Block #2 <-> #3
                4: devices[2]  # Move to Block #4 (final block)
            }
        elif len(devices) == 4:
            block_devices = {
                1: devices[0],
                2: devices[1],
                3: devices[2],
                4: devices[3],
                5: devices[3]
            }
            self.block_gpu_ids = {
                1: (devices[0], devices[1]),  # Block #1 <-> #2
                2: (devices[1], devices[2]),  # Block #2 <-> #3
                3: (devices[2], devices[3]),  # Block #3 <-> #4
                4: devices[3]  # Move to Block #4 (final block)
            }
        else:
            raise ValueError('Can only split model across a 1-4 devices')

        # Block #1
        self.conv1.cuda(block_devices[1])
        self.maxpool1.cuda(block_devices[1])

        # Block #2
        self.conv2.cuda(block_devices[2])
        self.maxpool2.cuda(block_devices[2])

        # Block #3
        self.conv3.cuda(block_devices[3])
        self.maxpool3.cuda(block_devices[3])

        # Block #4
        self.conv4.cuda(block_devices[4])
        self.maxpool4.cuda(block_devices[4])
        self.center.cuda(block_devices[4])
        self.gating.cuda(block_devices[4])

        self.attentionblock4.cuda(block_devices[4])
        self.up_concat4.cuda(block_devices[4])

        # Block #3
        self.attentionblock3.cuda(block_devices[3])
        self.up_concat3.cuda(block_devices[3])
        self.dsv4.cuda(block_devices[3])

        # Block #2
        self.attentionblock2.cuda(block_devices[2])
        self.up_concat2.cuda(block_devices[2])
        self.dsv3.cuda(block_devices[2])

        # Block #1
        self.up_concat1.cuda(block_devices[1])
        self.dsv2.cuda(block_devices[1])
        self.dsv1.cuda(block_devices[1])

        # Block #5
        self.final.cuda(block_devices[5])

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block_1 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor= sub_sample_factor)
        self.gate_block_2 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor=sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv3d(in_size*2, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm3d(in_size),
                                           nn.ReLU(inplace=True)
                                           )

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock3D') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)
        gate_2, attention_2 = self.gate_block_2(input, gating_signal)

        return self.combine_gates(torch.cat([gate_1, gate_2], 1)), torch.cat([attention_1, attention_2], 1)


