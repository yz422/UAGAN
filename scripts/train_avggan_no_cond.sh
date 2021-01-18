

# ----- MNIST + fashionMNIST, non-identical case, 10 discriminators -----
python train.py --dataroot ./data/MNIST_fashionMNIST --name mnistUniqueFashionUniform_AvgGAN_NoCond_10D --model avggan_no_cond --netG DCGANResnet --netD DCGANResnet \
  --direction AtoB --dataset_mode mnist_fashionmnist_split --pool_size 0 --gpu_ids 0 --niter 200 --niter_decay 200 --batch_size 256 \
  --output_nc 1 --num_threads 0 --n_class 10 --ngf 64 --ndf 64 --nz 128


# ----- MNIST + font, non-identical case, 10 discriminators -----
#python train.py --dataroot ./data/MNIST_font_new --name mnistUniqueFontUniform_AvgGAN_NoCond_10D --model avggan_no_cond --netG DCGANResnet --netD DCGANResnet \
#  --direction AtoB --dataset_mode mnist_font_split --pool_size 0 --gpu_ids 6 --niter 200 --niter_decay 200 --batch_size 256 \
#  --output_nc 1 --num_threads 0 --n_class 10 --ngf 64 --ndf 64 --nz 128


## ----- MNIST + font, identical case, 10 discriminators -----
#python train.py --dataroot ./data/MNIST_font_new --name mnistAndFontUniform_AvgGAN_NoCond_10D --model avggan_no_cond --netG DCGANResnet --netD DCGANResnet \
#  --direction AtoB --dataset_mode mnist_font_split --pool_size 0 --gpu_ids 6 --niter 200 --niter_decay 200 --batch_size 256 \
#  --output_nc 1 --num_threads 0 --n_class 10 --ngf 64 --ndf 64 --nz 128


###### three datasets mix #####
#python train.py --dataroot ./data/MNIST_fashionMNIST_font --name mnistUniqueFashionAndFontUniform_AvgGAN_NoCond_10D --model avggan_no_cond --netG DCGANResnet --netD DCGANResnet \
#  --direction AtoB --dataset_mode mnist_fashionmnist_split --pool_size 0 --gpu_ids 1 --niter 200 --niter_decay 200 --batch_size 256 \
#  --output_nc 1 --num_threads 0 --n_class 10 --ngf 64 --ndf 64 --nz 128