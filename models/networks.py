import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torch.nn.utils import spectral_norm
from torchvision import models
import numpy as np

###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            lr_l = max(lr_l, 0.1)
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


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nz, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, num_classes=10, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nz (int) -- the length of input noise
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'cDCGAN':
        net = cDCGANGenerator(input_nz, output_nc, ngf, norm_layer=nn.BatchNorm2d, num_classes=num_classes)
    elif netG == 'cDCGANResnet':
        net = cDCGANResnetGenerator(input_nz, output_nc, ngf, nf_max=256, img_size=32, num_classes=num_classes)
    elif netG == 'DCGANResnet':
        net = DCGANResnetGenerator(input_nz, output_nc, ngf, nf_max=256, img_size=32)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, num_classes=10, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'cDCGAN':
        net = cDCGANDiscriminator(input_nc, ndf, norm_layer=nn.BatchNorm2d, num_classes=num_classes)
    elif netD == 'cDCGANResnet':
        net = cDCGANResnetDiscriminator(input_nc, ndf, nf_max=256, img_size=32, num_classes=num_classes)
    elif netD == 'DCGANResnet':
        net = DCGANResnetDiscriminator(input_nc, ndf, nf_max=256, img_size=32)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'bce':
            self.loss = nn.BCELoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla', 'bce']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            if prediction.dtype == torch.float64:
                target_tensor = target_tensor.double()
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class SoftBCELoss(nn.Module):
    def __init__(self):
        super(SoftBCELoss, self).__init__()
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """
        loss = - target * inputs.log() - (1-target) * (1-inputs).log()
        loss = loss.mean()

        return loss


# Utility functions
def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg


def update_average(model_tgt, model_src, beta):
    toggle_grad(model_src, False)
    toggle_grad(model_tgt, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)

    toggle_grad(model_src, True)


class cDCGANGenerator(nn.Module):
    """Create a conditional DCGAN generator"""

    def __init__(self, input_nz, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, num_classes=10):
        """Construct a Unet generator
        Parameters:
            input_nz (int)  -- the length of input noise vector
            output_nc (int) -- the number of channels in output images
            num_classes (int) -- the number of classes in the dataset
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        """
        super(cDCGANGenerator, self).__init__()

        self.deconv1_1 = nn.ConvTranspose2d(input_nz, ngf * 2, kernel_size=4, stride=1, padding=0, bias=False)
        self.deconv1_1_bn = norm_layer(ngf * 2)
        # class one-hot vector input
        self.deconv1_2 = nn.ConvTranspose2d(num_classes, ngf * 2, kernel_size=4, stride=1, padding=0, bias=False)
        self.deconv1_2_bn = norm_layer(ngf * 2)
        self.deconv2 = nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv2_bn = norm_layer(ngf * 2)
        self.deconv3 = nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False)
        self.deconv3_bn = norm_layer(ngf)
        self.deconv4 = nn.ConvTranspose2d(ngf, output_nc, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, input, label):
        """Standard forward"""
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], dim=1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))
        return x


class cDCGANDiscriminator(nn.Module):
    """spectral normalization"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, num_classes=10):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(cDCGANDiscriminator, self).__init__()
        ndf = ndf + 1 if ndf % 2 == 1 else ndf
        self.conv1_1 = spectral_norm(nn.Conv2d(input_nc, ndf // 2, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv1_2 = spectral_norm(nn.Conv2d(num_classes, ndf // 2, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv2 = spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv3 = spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False))
        self.conv4 = spectral_norm(nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=0, bias=False))

    def forward(self, input, label):
        """Standard forward."""
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = torch.sigmoid(self.conv4(x))
        return x


# ResNet-like block
class ResNetBlock(nn.Module):
    def __init__(self, input_nc, output_nc, hidden_nc=None, bn=True, res_ratio=0.1):
        super().__init__()
        # Attributes
        self.bn = bn
        self.is_bias = not bn
        self.learned_shortcut = (input_nc != output_nc)
        self.input_nc = input_nc
        self.output_nc = output_nc
        if hidden_nc is None:
            self.hidden_nc = min(input_nc, output_nc)
        else:
            self.hidden_nc = hidden_nc
        self.res_ratio = res_ratio

        # Submodules
        self.conv_0 = nn.Conv2d(self.input_nc, self.hidden_nc, 3, stride=1, padding=1, bias=self.is_bias)
        if self.bn:
            self.bn2d_0 = nn.BatchNorm2d(self.hidden_nc)
        self.conv_1 = nn.Conv2d(self.hidden_nc, self.output_nc, 3, stride=1, padding=1, bias=self.is_bias)
        if self.bn:
            self.bn2d_1 = nn.BatchNorm2d(self.output_nc)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.input_nc, self.output_nc, 1, stride=1, padding=0, bias=False)
            if self.bn:
                self.bn2d_s = nn.BatchNorm2d(self.output_nc)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(x)
        if self.bn:
            dx = self.bn2d_0(dx)
        dx = self.relu(dx)
        dx = self.conv_1(dx)
        if self.bn:
            dx = self.bn2d_1(dx)
        out = self.relu(x_s + self.res_ratio * dx)
        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
            if self.bn:
                x_s = self.bn2d_s(x_s)
        else:
            x_s = x
        return x_s


class cDCGANResnetGenerator(nn.Module):
    '''The generator model'''

    def __init__(self, input_nz, output_nc, nf=64, nf_max=512, img_size=32, num_classes=10, bn=True, res_ratio=0.1, **kwargs):
        super().__init__()
        self.nf = nf
        self.nf_max = nf_max
        s0 = self.s0 = 4
        self.bn = bn
        self.input_nz = input_nz

        nlayers = int(np.log2(img_size / s0))
        self.nf0 = min(nf_max, nf * 2**(nlayers+1))

        self.fc = nn.Linear(input_nz, self.nf0 * s0 * s0 // 2)
        self.fc_label = nn.Linear(num_classes, self.nf0 * s0 * s0 // 2)
        if self.bn:
            self.bn1d = nn.BatchNorm1d(self.nf0 * s0 * s0 // 2)
            self.bn1d_label = nn.BatchNorm1d(self.nf0 * s0 * s0 // 2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        blocks = []
        for i in range(nlayers, 0, -1):
            nf0 = min(nf * 2 ** (i + 1), nf_max)
            nf1 = min(nf * 2 ** i, nf_max)
            blocks += [
                ResNetBlock(nf0, nf1, bn=self.bn, res_ratio=res_ratio),
                nn.Upsample(scale_factor=2)
            ]

        nf0 = min(nf * 2, nf_max)
        nf1 = min(nf, nf_max)
        blocks += [
            ResNetBlock(nf0, nf1, bn=self.bn, res_ratio=res_ratio),
            ResNetBlock(nf1, nf1, bn=self.bn, res_ratio=res_ratio)
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, output_nc, 3, padding=1)

    def forward(self, *inputs):
        z = inputs[0]   # noise
        y = inputs[1]   # label
        batch_size = z.size(0)

        # for noise
        z = z.view(batch_size, -1)
        out_z = self.fc(z)
        if self.bn:
            out_z = self.bn1d(out_z)
        out_z = self.relu(out_z)
        out_z = out_z.view(batch_size, self.nf0 // 2, self.s0, self.s0)

        # for label
        y = y.view(batch_size, -1)
        out_y = self.fc_label(y)
        if self.bn:
            out_y = self.bn1d_label(out_y)
        out_y = self.relu(out_y)
        out_y = out_y.view(batch_size, self.nf0 // 2, self.s0, self.s0)

        out = torch.cat([out_z, out_y], dim=1)

        out = self.resnet(out)
        out = self.conv_img(out)
        out = torch.tanh(out)

        return out


class cDCGANResnetDiscriminator(nn.Module):
    """ resnet, label as input """

    def __init__(self, input_nc, nf=64, nf_max=512, img_size=32, num_classes=10, res_ratio=0.1):
        super().__init__()
        s0 = self.s0 = 4
        self.nf = nf
        self.nf_max = nf_max

        # Submodules
        nlayers = int(np.log2(img_size / s0))
        self.nf0 = min(nf_max, nf * 2 ** nlayers)

        nf0 = min(nf, nf_max)
        nf1 = min(nf * 2, nf_max)
        blocks = [
            ResNetBlock(nf0, nf0, bn=False, res_ratio=res_ratio),
            ResNetBlock(nf0, nf1, bn=False, res_ratio=res_ratio)
        ]

        for i in range(1, nlayers + 1):
            nf0 = min(nf * 2 ** i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResNetBlock(nf0, nf1, bn=False, res_ratio=res_ratio),
            ]

        self.conv_img = nn.Conv2d(input_nc, 1 * nf // 2, 3, padding=1)
        self.conv_label = nn.Conv2d(num_classes, 1 * nf // 2, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0 * s0 * s0, 1)

    def forward(self, *inputs):
        x, y = inputs[0], inputs[1]
        assert (x.size(0) == y.size(0))
        batch_size = x.size(0)

        out_img = self.relu(self.conv_img(x))
        out_label = self.relu(self.conv_label(y))
        out = torch.cat([out_img, out_label], dim=1)
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0 * self.s0 * self.s0)
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out


class DCGANResnetGenerator(nn.Module):
    '''The generator model'''

    def __init__(self, input_nz, output_nc, nf=64, nf_max=512, img_size=32, bn=True, res_ratio=0.1, **kwargs):
        super().__init__()
        self.nf = nf
        self.nf_max = nf_max
        s0 = self.s0 = 4
        self.bn = bn
        self.input_nz = input_nz

        nlayers = int(np.log2(img_size / s0))
        self.nf0 = min(nf_max, nf * 2**(nlayers+1))

        self.fc = nn.Linear(input_nz, self.nf0 * s0 * s0)
        if self.bn:
            self.bn1d = nn.BatchNorm1d(self.nf0 * s0 * s0)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        blocks = []
        for i in range(nlayers, 0, -1):
            nf0 = min(nf * 2 ** (i + 1), nf_max)
            nf1 = min(nf * 2 ** i, nf_max)
            blocks += [
                ResNetBlock(nf0, nf1, bn=self.bn, res_ratio=res_ratio),
                nn.Upsample(scale_factor=2)
            ]

        nf0 = min(nf * 2, nf_max)
        nf1 = min(nf, nf_max)
        blocks += [
            ResNetBlock(nf0, nf1, bn=self.bn, res_ratio=res_ratio),
            ResNetBlock(nf1, nf1, bn=self.bn, res_ratio=res_ratio)
        ]

        self.resnet = nn.Sequential(*blocks)
        self.conv_img = nn.Conv2d(nf, output_nc, 3, padding=1)

    def forward(self, z):
        batch_size = z.size(0)

        # for noise
        z = z.view(batch_size, -1)
        out_z = self.fc(z)
        if self.bn:
            out_z = self.bn1d(out_z)
        out_z = self.relu(out_z)
        out_z = out_z.view(batch_size, self.nf0, self.s0, self.s0)
        out = self.resnet(out_z)
        out = self.conv_img(out)
        out = torch.tanh(out)

        return out


class DCGANResnetDiscriminator(nn.Module):
    """ resnet, label as input """

    def __init__(self, input_nc, nf=64, nf_max=512, img_size=32, num_classes=10, res_ratio=0.1):
        super().__init__()
        s0 = self.s0 = 4
        self.nf = nf
        self.nf_max = nf_max

        # Submodules
        nlayers = int(np.log2(img_size / s0))
        self.nf0 = min(nf_max, nf * 2 ** nlayers)

        nf0 = min(nf, nf_max)
        nf1 = min(nf * 2, nf_max)
        blocks = [
            ResNetBlock(nf0, nf0, bn=False, res_ratio=res_ratio),
            ResNetBlock(nf0, nf1, bn=False, res_ratio=res_ratio)
        ]

        for i in range(1, nlayers + 1):
            nf0 = min(nf * 2 ** i, nf_max)
            nf1 = min(nf * 2 ** (i + 1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResNetBlock(nf0, nf1, bn=False, res_ratio=res_ratio),
            ]

        self.conv_img = nn.Conv2d(input_nc, 1 * nf, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0 * s0 * s0, 1)

    def forward(self, x):
        batch_size = x.size(0)

        out = self.relu(self.conv_img(x))
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0 * self.s0 * self.s0)
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out
