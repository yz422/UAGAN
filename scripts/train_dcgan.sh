

## ----- MNIST + font, non-identical case, 10 discriminators -----
python train.py --dataroot ./data/MNIST_font_new --name mnist_Font_GAN --model dcgan --netG cDCGANResnet --netD cDCGANResnet \
  --direction AtoB --dataset_mode mnist_font --pool_size 0 --gpu_ids 1 --niter 100 --niter_decay 100 --batch_size 256 \
  --output_nc 1 --num_threads 0 --n_class 10 --ngf 64 --ndf 64 --nz 128

###### 3 datasets mix #####
#python train.py --dataroot ./data/MNIST_fashionMNIST_font --name mnist_fashionMNIST_Font_GAN --model dcgan --netG cDCGANResnet --netD cDCGANResnet \
#  --direction AtoB --dataset_mode mnist_fashionmnist_font --pool_size 0 --gpu_ids 4 --niter 100 --niter_decay 100 --batch_size 256 \
#  --output_nc 1 --num_threads 0 --n_class 10 --ngf 64 --ndf 64 --nz 128 --save_epoch_freq 25
#