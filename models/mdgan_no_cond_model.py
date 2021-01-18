import torch
from torch import nn
from torch.autograd import grad as torch_grad
import copy

from .base_model import BaseModel
from . import networks
from parse_config import ConfigParser
import parse_config


class MDGANNoCondModel(BaseModel):
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
            parser.add_argument('--lambda_D', type=float, default=0.1, help='weight for dadgan D')
            parser.add_argument('--lambda_reg', type=float, default=10, help='weight for gradient penalty')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN_all', 'D_real_all', 'D_fake_all']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if self.isTrain:
            self.visual_names = ['fake_B_0', 'real_B_0', 'fake_B_1', 'real_B_1', 'fake_B_2', 'real_B_2', 'fake_B_3',
                                 'real_B_3']
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
            self.netD = []
            for i in range(10):
                self.netD.append(networks.define_D(opt.output_nc, opt.ndf, opt.netD, n_layers_D=opt.n_layers_D,
                                                   norm=opt.norm, init_type=opt.init_type, init_gain=opt.init_gain,
                                                   num_classes=opt.n_class, gpu_ids=self.gpu_ids))
            self.shuffle_index = torch.range(0, len(self.netD) - 1)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr_G, betas=(opt.beta1, 0.999))
            self.optimizer_D = []
            for i in self.netD:
                opt_D = torch.optim.Adam(i.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999))
                self.optimizer_D.append(opt_D)
                self.optimizers.append(opt_D)
            self.optimizers.append(self.optimizer_G)

        self.weights = []

        print(self.netG)
        if self.isTrain:
            print(self.netD[0])

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        if self.opt.isTrain:
            self.real_A = []  # labels
            self.real_B = []  # images
            self.image_paths = []
            for i in range(10):
                self.real_A.append(input['A_' + str(i)].to(self.device))
                self.real_B.append(input['B_' + str(i)].to(self.device))
                self.image_paths.append(input['A_paths_' + str(i)])

            self.real_B_0 = self.real_B[0]
            self.real_B_1 = self.real_B[1]
            self.real_B_2 = self.real_B[2]
            self.real_B_3 = self.real_B[3]

            if len(self.weights) == 0:
                self.weights = input['weights'][0].to(self.device).float()
                print(self.weights)
        else:
            self.real_A = input['A'].to(self.device)
            self.real_B = input['B'].to(self.device)
            self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # --- generate one fake image for all discriminators --- #
        noise = torch.randn(self.real_A[0].size(0), self.opt.nz, 1, 1).to(self.device)
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

    def backward_D(self,shuffle=False, this_iter_val=99999):
        """Calculate GAN loss for the discriminator"""
        # ----- one fake image is passed to all discriminators ----- #
        self.loss_D_fake = []
        self.loss_D_real = []
        self.loss_D_reg = []

        if shuffle == True and this_iter_val % 10 == 0:
            self.shuffle_index = torch.randperm(len(self.real_A))

        for i in range(len(self.real_A)):
            this_index=int(self.shuffle_index[i].item())
            # Fake; stop backprop to the generator by detaching fake_B
            pred_fake = self.netD[this_index](self.fake_B.detach())
            self.loss_D_fake.append(self.criterionGAN(pred_fake, False))

            # Real
            self.real_B[i].requires_grad_()
            pred_real = self.netD[this_index](self.real_B[i])
            self.loss_D_real.append(self.criterionGAN(pred_real, True))

            # zero-centered penalty
            reg = networks.compute_grad2(pred_real, self.real_B[i]).mean()
            self.loss_D_reg.append(reg)

        self.loss_D_fake_all = None
        self.loss_D_real_all = None
        self.loss_D_reg_all = None
        for i in range(len(self.loss_D_real)):
            if self.loss_D_real_all is None:
                self.loss_D_fake_all = self.loss_D_fake[i]
                self.loss_D_real_all = self.loss_D_real[i]
                self.loss_D_reg_all = self.loss_D_reg[i]
            else:
                self.loss_D_fake_all += self.loss_D_fake[i]
                self.loss_D_real_all += self.loss_D_real[i]
                self.loss_D_reg_all += self.loss_D_reg[i]

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake_all + self.loss_D_real_all) * self.opt.lambda_D + self.loss_D_reg_all * self.opt.lambda_reg
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN loss for the generator"""
        # ----- one fake image is passed to all discriminators ----- #
        self.loss_G_GAN_all = 0
        for i in range(len(self.real_A)):
            # Fake; stop backprop to the generator by detaching fake_B
            pred_fake = self.netD[i](self.fake_B)
            ##Using Average Loss
            self.loss_G_GAN_all += self.criterionGAN(pred_fake, True) / len(self.real_A)

        self.loss_G = self.loss_G_GAN_all * self.opt.lambda_G  # 0.1
        self.loss_G.backward()

    def optimize_parameters(self, shuffle=False, this_iter_val=99999):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        for opt in self.optimizer_D:
            opt.zero_grad()

        self.backward_D(shuffle=shuffle, this_iter_val=this_iter_val)
        for opt in self.optimizer_D:
            opt.step()
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
