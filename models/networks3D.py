import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import monai
import torch.nn.functional as F
# from options.train_options import TrainOptions
# opt = TrainOptions().parse()
import torchvision
from torchsummary import summary
###############################################################################
# Helper Functions
scaler = torch.cuda.amp.GradScaler()
###############################################################################

class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = Parkinson_ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_custom':
        net = UnetGenerator(input_nc, output_nc, 5, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'Dynet':
        net = Dynet()
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    # summary(netG, input_size=(1, 116, 140, 116))
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'n_layers':
        net = NLayerDiscriminator_multi(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss() #use
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


'''
define the correlation coefficient loss
'''
def Cor_CoeLoss(y_pred, y_target):
    x = y_pred
    y = y_target
    x_var = x - torch.mean(x)
    y_var = y - torch.mean(y)
    r_num = torch.sum(x_var * y_var)
    r_den = torch.sqrt(torch.sum(x_var ** 2)) * torch.sqrt(torch.sum(y_var ** 2))
    r = r_num / r_den

    # return 1 - r  # best are 0
    return 1 - r**2 # abslute constrain


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/

class AdaIN3d(nn.Module):
    def __init__(self, style_dim, num_features):
        super(AdaIN3d, self).__init__()
        self.instance_norm = nn.InstanceNorm3d(num_features, affine=False)
        self.style_scale_transform = nn.Linear(style_dim, num_features)
        self.style_shift_transform = nn.Linear(style_dim, num_features)

    def forward(self, x, style):
        # Perform instance normalization on input
        x = self.instance_norm(x)

        # Compute style scale and shift
        style_scale = self.style_scale_transform(style).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        style_shift = self.style_shift_transform(style).unsqueeze(2).unsqueeze(3).unsqueeze(4)

        # Apply style to input
        x = style_scale * x + style_shift
        return x

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode)
        return x

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 5)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_scale, style_bias):
    size = content_feat.size()
    style_scale = style_scale[:, :, None, None, None].expand(size)
    style_bias = style_bias[:, :, None, None, None].expand(size)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)

    return normalized_feat * style_scale + style_bias

def AdaIN(x, w):
    dim = x.size()[1]
    linear_scale = nn.Linear(512, dim).cuda()
    linear_bias = nn.Linear(512, dim).cuda()
    style_scale = linear_scale(w)
    style_bias = linear_bias(w)
    x = adaptive_instance_normalization(x.clone(), style_scale, style_bias)
    return x

#


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc

        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

# ------------------------------------------------ mapping network
        mapping = []
        z_dim = 512
        for i in range(3):
            mapping += [nn.Linear(z_dim, z_dim)]
            mapping += [nn.LeakyReLU()]

        model = [nn.ReplicationPad3d(3),
                 nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.LeakyReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.LeakyReLU(True)]

        mult = 2**n_downsampling
        subpixel_channel = ngf * mult
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        # ---------------------------------------------- decoder
        decoder, decoder1, decoder2 = [], [], []
        self.AdaIN_1 = AdaIN3d(z_dim, subpixel_channel)

        decoder += [nn.Conv3d(subpixel_channel,
                              int(subpixel_channel*8),
                              kernel_size=1,
                              stride=1, padding=0, bias=use_bias),
                    norm_layer(int(subpixel_channel*8)),
                    nn.LeakyReLU()]
        decoder += [PixelShuffle3d(2)]
        decoder += [nn.PReLU()]

        decoder += [nn.Conv3d(subpixel_channel,
                              subpixel_channel,
                              kernel_size=3,
                              stride=1, padding=1, bias=use_bias),
                    norm_layer(subpixel_channel),
                    nn.LeakyReLU()]
        channel = int(subpixel_channel/2)
        decoder += [nn.Conv3d(subpixel_channel,
                              channel,
                              kernel_size=1,
                              stride=1, padding=0, bias=use_bias),
                    norm_layer(channel),
                    nn.LeakyReLU()]

        self.AdaIN_2 = AdaIN3d(z_dim, channel)

        subpixel_channel = channel
        # ---------------------------------------------------------
        decoder1 += [nn.Conv3d(subpixel_channel,
                              int(subpixel_channel * 8),
                              kernel_size=1,
                              stride=1, padding=0, bias=use_bias),
                    norm_layer(int(subpixel_channel * 8)),
                    nn.LeakyReLU()]

        decoder1 += [PixelShuffle3d(2)]
        decoder1 += [nn.PReLU()]
        decoder1 += [nn.Conv3d(subpixel_channel,
                              subpixel_channel,
                              kernel_size=3,
                              stride=1, padding=1, bias=use_bias),
                    norm_layer(subpixel_channel),
                    nn.LeakyReLU()]
        channel = int(subpixel_channel / 2)
        decoder1 += [nn.Conv3d(subpixel_channel,
                               channel,
                               kernel_size=1,
                               stride=1, padding=0, bias=use_bias),
                     norm_layer(channel),
                     nn.LeakyReLU()]

        self.AdaIN_3 = AdaIN3d(z_dim, channel)
        decoder2 += [nn.ReplicationPad3d(3)]
        decoder2 += [nn.Conv3d(channel, output_nc, kernel_size=7, padding=0)]
        decoder2 += [nn.Tanh()]
        # ---------------------------------------------------------

        self.z_dim = z_dim
        self.mapping = nn.Sequential(*mapping)  # mapping network
        self.model = nn.Sequential(*model)      # Generator(input -> Resnet)
        self.decoder = nn.Sequential(*decoder)  # Generator(Resnet -> Output)
        self.decoder1 = nn.Sequential(*decoder1)
        self.decoder2 = nn.Sequential(*decoder2)

    def forward(self, input):
        with torch.cuda.amp.autocast():
            z_sample = torch.randn(1, self.z_dim).cuda()
            z_sample = nn.functional.normalize(z_sample)
            w = self.mapping(z_sample)
            x = self.model(input)
            x = self.AdaIN_1(x, w)
            x = self.decoder(x)
            x = self.AdaIN_2(x, w)
            x = self.decoder1(x)
            x = self.AdaIN_3(x, w)
            x = self.decoder2(x)
        return x



class Parkinson_ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Parkinson_ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.Parkinson_classification_module = nn.ModuleList()

        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

# ------------------------------------------------ mapping network
        mapping = []
        z_dim = 512
        mapping += [nn.Conv3d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(ngf * 8),
                    nn.LeakyReLU(True)]

        mapping += [nn.Flatten()]
        mapping += [nn.Linear(ngf * 8 * 8 * 12 * 4, z_dim)]

        for i in range(9):
            mapping += [nn.Linear(z_dim, z_dim)]
            mapping += [nn.LeakyReLU()]
        # ------------------------------------------------ Model(Common Encoding module)
        model = [nn.ReplicationPad3d(3),
                 nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.LeakyReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.LeakyReLU(True)]
        self.model = nn.Sequential(*model)

        mult = 2**n_downsampling
        subpixel_channel = ngf * mult
        # ---------------------------------------------- AFE_Module & AdaIN_list
        self.AFE_Module = nn.ModuleList()
        self.AdaIN_list = nn.ModuleList()
        for i in range(n_blocks):
            model = [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                 use_bias=use_bias)]
            model = nn.Sequential(*model)

            adain = AdaIN3d(512, ngf * mult)

            self.AFE_Module.append(model)
            self.AdaIN_list.append(adain)

        # ---------------------------------------------- Decoder(Multiscale SPECT Reconstruction Subnetwork)
        decoder, decoder1, decoder2 = [], [], []
        decoder += [nn.Conv3d(subpixel_channel,
                              int(subpixel_channel*8),
                              kernel_size=1,
                              stride=1, padding=0, bias=use_bias),
                    norm_layer(int(subpixel_channel*8)),
                    nn.LeakyReLU()]
        decoder += [PixelShuffle3d(2)]
        decoder += [nn.PReLU()]

        decoder += [nn.Conv3d(subpixel_channel,
                              subpixel_channel,
                              kernel_size=3,
                              stride=1, padding=1, bias=use_bias),
                    norm_layer(subpixel_channel),
                    nn.LeakyReLU()]
        channel = int(subpixel_channel/2)
        decoder += [nn.Conv3d(subpixel_channel,
                              channel,
                              kernel_size=1,
                              stride=1, padding=0, bias=use_bias),
                    norm_layer(channel),
                    nn.LeakyReLU()]
        subpixel_channel = channel
        # ---------------------------------------------------------
        decoder1 += [nn.Conv3d(subpixel_channel,
                              int(subpixel_channel * 8),
                              kernel_size=1,
                              stride=1, padding=0, bias=use_bias),
                    norm_layer(int(subpixel_channel * 8)),
                    nn.LeakyReLU()]

        decoder1 += [PixelShuffle3d(2)]
        decoder1 += [nn.PReLU()]
        decoder1 += [nn.Conv3d(subpixel_channel,
                              subpixel_channel,
                              kernel_size=3,
                              stride=1, padding=1, bias=use_bias),
                    norm_layer(subpixel_channel),
                    nn.LeakyReLU()]
        channel = int(subpixel_channel / 2)
        decoder1 += [nn.Conv3d(subpixel_channel,
                               channel,
                               kernel_size=1,
                               stride=1, padding=0, bias=use_bias),
                     norm_layer(channel),
                     nn.LeakyReLU()]
        decoder2 += [nn.ReplicationPad3d(3)]
        decoder2 += [nn.Conv3d(channel, output_nc, kernel_size=7, padding=0)]
        decoder2 += [nn.Tanh()]

        # --------------------------------------------------------- parkinson_classification_module

        parkinson_classification_module = []
        parkinson_classification_module += [nn.Conv3d(1, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(ngf), nn.LeakyReLU(True)]
        parkinson_classification_module = nn.Sequential(*parkinson_classification_module)
        self.Parkinson_classification_module.append(parkinson_classification_module)

        for i in range(3):
            mult = 2 ** i
            parkinson_classification_module = [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                                                norm_layer(ngf * mult * 2), nn.LeakyReLU(True)]
            model = nn.Sequential(*parkinson_classification_module)
            self.Parkinson_classification_module.append(model)

        parkinson_classification_module = [nn.Conv3d(ngf * mult * 2, 1, kernel_size=3, stride=1, padding=1, bias=use_bias), norm_layer(1), nn.LeakyReLU(True)]
        model = nn.Sequential(*parkinson_classification_module)
        self.Parkinson_classification_module.append(model)


        self.z_dim = z_dim
        self.mapping = nn.Sequential(*mapping)  # mapping network

        self.decoder = nn.Sequential(*decoder)  # Generator(Resnet -> Output)
        self.decoder1 = nn.Sequential(*decoder1)
        self.decoder2 = nn.Sequential(*decoder2)


    def forward(self, input):
        with torch.cuda.amp.autocast():
            cl = x = input
            x_common = self.model(x)

            for i in range(0, 5, 1):
                cl = self.Parkinson_classification_module[i](cl)
                if i == 2:
                    class_feature = cl

            w = self.mapping(x_common)

            x = x_common
            for j in range(9):
                x = self.AFE_Module[j](x)
                x = self.AdaIN_list[j](x, w)
            x_quarter = x
            x_half = self.decoder(x_quarter)
            x = self.decoder1(x_half)
            x = self.decoder2(x)

        return x, x_half, x_quarter, cl, class_feature, x_common




class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.LeakyReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm3d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


def Dynet():

    sizes, spacings = [128, 128, 64], (1.5,1.5,1.5)

    strides, kernels = [], []

    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])

    net = monai.networks.nets.DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        res_block=True,
    )

    net.add_module("activation", torch.nn.Tanh())

    return net


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        # if use_sigmoid:
        #     sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)
        # self.model_half_feature =/

    def forward(self, input):
        with torch.cuda.amp.autocast():
            score = self.model(input)
        return score

class NLayerDiscriminator_multi(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super(NLayerDiscriminator_multi, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        self.dis_1 = nn.ModuleList()
        self.dis_2 = nn.ModuleList()
        self.dis_3 = nn.ModuleList()
        kw = 3
        padw = 1


        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        seq_temp = nn.Sequential(*sequence)
        self.dis_1.append(seq_temp)

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence = [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            seq_temp = nn.Sequential(*sequence)
            if n < 2:
                self.dis_1.append(seq_temp)
            elif n >= 2:
                self.dis_1.append(seq_temp)
                self.dis_2.append(seq_temp)

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence = [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        seq_temp = nn.Sequential(*sequence)
        self.dis_1.append(seq_temp)
        self.dis_2.append(seq_temp)
        self.dis_3.append(seq_temp)

        sequence = [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        seq_temp = nn.Sequential(*sequence)
        self.dis_1.append(seq_temp)
        self.dis_2.append(seq_temp)
        self.dis_3.append(seq_temp)


    def forward(self, input, input_half, input_quarter):
        with torch.cuda.amp.autocast():
            x = input
            for i in range(0, 5):
                x = self.dis_1[i](x)
                if i == 1:
                    feature_half_temp = x
            score = x
            # score = self.model(input)
            # ====================================dis_1
            if input_half == None:
                x = feature_half_temp
            else:
                x = input_half
            for i in range(0, 3):
                x = self.dis_2[i](x)
                if i == 0:
                    feature_quarter_temp = x
            score_half = x
            # ====================================dis_2
            if input_quarter == None:
                x = feature_quarter_temp
            else:
                x = input_quarter

            for i in range(0, 2):
                x = self.dis_3[i](x)
            score_quarter = x
            # ====================================dis_3
        return (score + score_half + score_quarter) / 3


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        self.net = [
            nn.Conv3d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)