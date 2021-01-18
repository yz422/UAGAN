

# ----- MNIST + fashionMNIST, non-identical case, 10 discriminators -----
python train_mdgan.py --dataroot ./data/MNIST_fashionMNIST --name mnistUniqueFashionUniform_MDGAN_10D --model mdgan --netG cDCGANResnet --netD cDCGANResnet \
  --direction AtoB --dataset_mode mnist_fashionmnist_split --pool_size 0 --gpu_ids 4 --niter 200 --niter_decay 200 --batch_size 256 \
  --output_nc 1 --num_threads 0 --n_class 10 --ngf 64 --ndf 64 --nz 128


# ----- MNIST + font, non-identical case, 10 discriminators -----
#python train_mdgan.py --dataroot ./data/MNIST_font_new --name mnistUniqueFontUniform_MDGAN_10D --model mdgan --netG cDCGANResnet --netD cDCGANResnet \
#  --direction AtoB --dataset_mode mnist_font_split --pool_size 0 --gpu_ids 6,7 --niter 200 --niter_decay 200 --batch_size 256 \
#  --output_nc 1 --num_threads 0 --n_class 10 --ngf 64 --ndf 64 --nz 128


# ----- MNIST + fashionMNIST, identical case, 10 discriminators -----
#python train_mdgan.py --dataroot ./data/MNIST_fashionMNIST --name mnistAndFashionUniform_MDGAN_10D --model mdgan --netG cDCGANResnet --netD cDCGANResnet \
#  --direction AtoB --dataset_mode mnist_fashionmnist_split --pool_size 0 --gpu_ids 4,5 --niter 200 --niter_decay 200 --batch_size 256 \
#  --output_nc 1 --num_threads 0 --n_class 10 --ngf 64 --ndf 64 --nz 128


## ----- MNIST + font, identical case, 10 discriminators -----
#python train_mdgan.py --dataroot ./data/MNIST_font_new --name mnistAndFontUniform_MDGAN_10D --model mdgan --netG cDCGANResnet --netD cDCGANResnet \
#  --direction AtoB --dataset_mode mnist_font_split --pool_size 0 --gpu_ids 6,7 --niter 200 --niter_decay 200 --batch_size 256 \
#  --output_nc 1 --num_threads 0 --n_class 10 --ngf 64 --ndf 64 --nz 128



##### three datasets mix #####
#python train_mdgan.py --dataroot ./data/MNIST_fashionMNIST_font --name mnistUniqueFashionAndFontUniform_MDGAN_10D_iter10 --model mdgan --netG cDCGANResnet --netD cDCGANResnet \
#  --direction AtoB --dataset_mode mnist_fashionmnist_font_split --pool_size 0 --gpu_ids 4 --niter 200 --niter_decay 200 --batch_size 256 \
#  --output_nc 1 --num_threads 0 --n_class 10 --ngf 64 --ndf 64 --nz 128

#python train_mdgan.py --dataroot ./data/MNIST_fashionMNIST_font --name mnistAndFashionAndFontUniform_MDGAN_10D_iter10 --model mdgan --netG cDCGANResnet --netD cDCGANResnet \
#  --direction AtoB --dataset_mode mnist_fashionmnist_split --pool_size 0 --gpu_ids 4,5 --niter 200 --niter_decay 200 --batch_size 256 \
#  --output_nc 1 --num_threads 0 --n_class 10 --ngf 64 --ndf 64 --nz 128

