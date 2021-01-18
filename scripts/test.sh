


#python test.py --dataroot ./data/MNIST_fashionMNIST --name mnistUniqueFashionUniform_MDGAN_10D --model mdgan \
#  --netG cDCGANResnet --direction AtoB --dataset_mode mnist_fashionmnist --gpu_ids 0 --batch_size 1024 --epoch 400 --results_dir results \
#  --output_nc 1 --n_class 10 --ngf 64 --nz 128
#
#python test.py --dataroot ./data/MNIST_font --name mnistUniqueFontUniform_MDGAN_10D --model mdgan \
#  --netG cDCGANResnet --direction AtoB --dataset_mode mnist_font --gpu_ids 0 --batch_size 1024 --epoch 400 --results_dir results \
#  --output_nc 1 --n_class 10 --ngf 64 --nz 128


#python test.py --dataroot ./data/MNIST_fashionMNIST --name mnistAndFashionUniform_MDGAN_10D --model mdgan \
#  --netG cDCGANResnet --direction AtoB --dataset_mode mnist_fashionmnist --gpu_ids 0 --batch_size 1024 --epoch 400 --results_dir results \
#  --output_nc 1 --n_class 10 --ngf 64 --nz 128


#python test.py --dataroot ./data/MNIST_font_new --name mnistAndFontUniform_UAGAN_10 --model uagan \
#  --netG cDCGANResnet --direction AtoB --dataset_mode mnist_font --gpu_ids 0 --batch_size 1024 --epoch 400 --results_dir results \
#  --output_nc 1 --n_class 10 --ngf 64 --nz 128
##
#python test.py --dataroot ./data/MNIST_font_new --name mnistAndFontUniform_MDGAN_10D --model mdgan \
#  --netG cDCGANResnet --direction AtoB --dataset_mode mnist_font --gpu_ids 0 --batch_size 1024 --epoch 400 --results_dir results \
#  --output_nc 1 --n_class 10 --ngf 64 --nz 128
#
#python test.py --dataroot ./data/MNIST_font_new --name mnistAndFontUniform_AvgGAN_10D --model avggan \
#  --netG cDCGANResnet --direction AtoB --dataset_mode mnist_font --gpu_ids 0 --batch_size 1024 --epoch 400 --results_dir results \
#  --output_nc 1 --n_class 10 --ngf 64 --nz 128

#python test.py --dataroot ./data/MNIST_font_new --name mnist_Font_GAN --model dcgan \
#  --netG cDCGANResnet --direction AtoB --dataset_mode mnist_font --gpu_ids 0 --batch_size 1024 --epoch 200 --results_dir results \
#  --output_nc 1 --n_class 10 --ngf 64 --nz 128
