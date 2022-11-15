import importlib

from itertools import repeat
import collections.abc

from torch.nn import functional as F
import torch
import torch.nn as nn
from .buildingblocks import create_encoders, res_decoders, ExtResNetBlock


class DeepLabHead(nn.Module):
    def __init__(self, in_channels, aspp_channel, num_classes, ratio, aspp_dilate=[4, 8, 16]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_channel, aspp_dilate),
            nn.ConvTranspose3d(in_channels=aspp_channel, out_channels=int(aspp_channel / 8), kernel_size=ratio,
                               stride=ratio),
            nn.GroupNorm(num_groups=8, num_channels=int(aspp_channel / 8)),
            nn.ReLU(inplace=True),
            nn.Conv3d(int(aspp_channel / 8), int(aspp_channel / 8), 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=int(aspp_channel / 8)),
            nn.ReLU(inplace=True),
            nn.Conv3d(int(aspp_channel / 8), num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier(feature)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias, groups=in_channels),
            # PointWise Conv
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv3d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-3:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='trilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, aspp_channel, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = aspp_channel
        modules = []
        modules.append(nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv3d(5 * out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                                module.out_channels,
                                                module.kernel_size,
                                                module.stride,
                                                module.padding,
                                                module.dilation,
                                                module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


class FINNEAS_OnlySegV1(nn.Module):
    """ Masked Autoencoder with ResNet encoder + DeepLab segmentation header
    """

    def __init__(self, opt, embed_dim=1024, depth=6, decoder_embed_dim=512, norm_pix_loss=False, patch_size=5):
        super().__init__()

        # --------------------------------------------------------------------------
        self.patch_size = patch_size
        # ResNet encoder specifics
        self.opt = opt
        to_tuple = _ntuple(depth)
        # encoder
        self.local_encoder = create_encoders(in_channels=1, f_maps=to_tuple(embed_dim), basic_module=ExtResNetBlock,
                                             conv_kernel_size=4, conv_stride_size=4, conv_padding=0, layer_order='gcr',
                                             num_groups=32)

        self.global_encoder = create_encoders(in_channels=1, f_maps=to_tuple(embed_dim), basic_module=ExtResNetBlock,
                                              conv_kernel_size=4, conv_stride_size=4, conv_padding=0, layer_order='gcr',
                                              num_groups=32)

        self.seg_decoder = DeepLabHead(in_channels=embed_dim * 2, aspp_channel=embed_dim, num_classes=opt.cls_num,
                                       ratio=4)

    def patchify(self, imgs, p):
        """

        imgs: (N, 1, H, W, D)
        x: (N, H*W*D/P***3, patch_size**3)
        """
        # p = self.patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0 and imgs.shape[4] % p == 0
        h, w, d = [i // p for i in self.opt.patch_size]
        # h = w =  imgs.shape[2] // p
        # d = imgs.shape[4] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p, d, p))
        x = torch.einsum('nchpwqdr->nhwdpqrc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, p ** 3))
        return x

    def unpatchify(self, x, p):
        """

        x: (N, H*W*D/P***3, patch_size**3)
        imgs: (N, 1, H, W, D)
        """
        # p = self.patch_size
        h, w, d = [i // p for i in self.opt.patch_size]

        # h = w = d = int(round(x.shape[1] ** (1 / 3)))
        assert h * w * d == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, d, p, p, p))
        x = torch.einsum('nhwdpqr->nhpwqdr', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, w * p, d * p))
        return imgs

    def random_masking(self, x, mask_ratio, p):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        x = self.patchify(x, p)

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask_ = torch.zeros_like(x_masked)
        # generate the binary mask: 0 is keep, 1 is remove

        x_empty = torch.zeros((N, L - len_keep, D)).cuda()
        mask = torch.ones_like(x_empty)
        x_ = torch.cat([x_masked, x_empty], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        mask_ = torch.cat([mask_, mask], dim=1)
        mask_ = torch.gather(mask_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        x_masked = self.unpatchify(x_, p)

        mask = self.unpatchify(mask_, p)

        return x_masked, mask

    def forward_encoder(self, x, mask_ratio, p):

        # masking: length -> length * mask_ratio
        x, mask = self.random_masking(x, mask_ratio, p)

        # apply Transformer blocks
        for blk in self.local_encoder:
            x = blk(x)

        return x, mask

    def forward_global_encoder(self, x, mask_ratio, p):

        # masking: length -> length * mask_ratio
        x, mask = self.random_masking(x, mask_ratio, p)

        # apply Transformer blocks
        for blk in self.global_encoder:
            x = blk(x)

        return x, mask

    def forward(self, local_patch, local_label, global_img, global_label, location_target, orientation_tgt,
                mask_ratio=0,pseudo=False):
            if pseudo:
                local_latent_1, _ = self.forward_encoder(local_patch, mask_ratio=0, p=8)
                global_latent_1, _ = self.forward_global_encoder(global_img, mask_ratio=0, p=4)

                pred_1 = self.seg_decoder(torch.concat([local_latent_1, global_latent_1], dim=1))
                return pred_1


def only_seg_V1(opt, **kwargs):
    model = FINNEAS_OnlySegV1(opt=opt,
                                                embed_dim=512, depth=8, decoder_embed_dim=32, norm_pix_loss=False,
                                                patch_size=8, **kwargs)
    return model