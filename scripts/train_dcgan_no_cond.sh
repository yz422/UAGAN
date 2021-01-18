

# ----- MNIST + font, non-identical case, 10 discriminators -----
python train.py --dataroot ./data/MNIST_font_new --name mnist_Font_GAN_NoCond --model dcgan_no_cond --netG DCGANResnet --netD DCGANResnet \
  --direction AtoB --dataset_mode mnist_font --pool_size 0 --gpu_ids 3 --niter 100 --niter_decay 100 --batch_size 256 \
  --output_nc 1 --num_threads 0 --n_class 10 --ngf 64 --ndf 64 --nz 128
