import torch
from torch import nn
import copy

from .base_model import BaseModel
from . import networks
from parse_config import ConfigParser
import parse_config


class DCGANNoCondModel(BaseModel):
    """ This class implements the DCGAN model,

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        parser.set_defaults(norm='batch', netG='cDCGANResnet', netD='cDCGANResnet', load_size=32, crop_size=32)
        parser.add_argument('--nz', type=int, default=128, help='length of noise vector')
        parser.add_argument('--n_class', type=int, default=10, help='number of classes')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='bce')
            parser.add_argument('--lambda_G', type=float, default=1, help='weight for dadgan G ')
            parser.add_argument('--lambda_D', type=float, default=1, help='weight for dadgan D')
            parser.add_argument('--lambda_reg', type=float, default=10, help='weight for gradient penalty')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if self.isTrain:
            self.visual_names = ['fake_B_0', 'real_B_0', 'fake_B_1', 'real_B_1', 'fake_B_2', 'real_B_2', 'fake_B_3', 'real_B_3']
        else:
            self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.nz, opt.output_nc, opt.ngf, opt.netG, norm=opt.norm, use_dropout=not opt.no_dropout,
                                      init_type=opt.init_type, init_gain=opt.init_gain, num_classes=opt.n_class,
                                      gpu_ids=self.gpu_ids)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, n_layers_D=opt.n_layers_D,
                                               norm=opt.norm, init_type=opt.init_type, init_gain=opt.init_gain,
                                               num_classes=opt.n_class, gpu_ids=self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        print(self.netG)
        if self.isTrain:
            print(self.netD)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        self.real_B_0 = self.real_B[0:1]
        self.real_B_1 = self.real_B[1:2]
        self.real_B_2 = self.real_B[2:3]
        self.real_B_3 = self.real_B[3:4]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        noise = torch.randn(self.real_A.size(0), self.opt.nz, 1, 1).to(self.device)
        self.fake_B = self.netG(noise)  # label as onehot encoding in G

        self.fake_B_0 = self.fake_B[0:1]
        self.fake_B_1 = self.fake_B[1:2]
        self.fake_B_2 = self.fake_B[2:3]
        self.fake_B_3 = self.fake_B[3:4]

    def test(self):
        with torch.no_grad():
            noise = torch.randn(self.real_A.size(0), self.opt.nz, 1, 1).to(self.device)
            self.fake_B = self.netG(noise)  # label as onehot encoding in G
            self.compute_visuals()

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # --- label as conditional image
        pred_fake = self.netD(self.fake_B.detach())  # we use label_img as condition
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        self.real_B.requires_grad_()
        pred_real = self.netD(self.real_B)  # label as conditional image
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # zero-centered penalty
        reg = networks.compute_grad2(pred_real, self.real_B).mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real)*self.opt.lambda_D + reg * self.opt.lambda_reg
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN loss for the generator"""
        # First, G(A) should fake the discriminator
        pred_fake = self.netD(self.fake_B)     # label as onehot encoding in G
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G = self.loss_G_GAN * self.opt.lambda_G
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights


