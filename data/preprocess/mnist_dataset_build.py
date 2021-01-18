
import os, json, glob
from skimage import io
from tqdm import tqdm
import numpy as np
import random
from torchvision import datasets
from PIL import Image
import h5py


def build_MNIST_h5(data_dir, save_dir):
    dataset = datasets.MNIST(root=data_dir, download=True, transform=None)
    test_dataset = datasets.MNIST(root=data_dir, download=True, train=False, transform=None)
    train_all_file = h5py.File(os.path.join(save_dir, 'train_MNIST.h5'), 'w')
    test_file = h5py.File(os.path.join(save_dir, 'test_MNIST.h5'), 'w')

    print('Processing training files')
    for i in range(len(dataset)):
        image, label = dataset[i]
        train_all_file.create_dataset('images/{:d}'.format(i), data=image)
        train_all_file.create_dataset('labels/{:d}'.format(i), data=label)

    print('Processing test files')
    for i in range(len(test_dataset)):
        image, label = test_dataset[i]
        test_file.create_dataset('images/{:d}'.format(i), data=image)
        test_file.create_dataset('labels/{:d}'.format(i), data=label)

    print('all: {:d}'.format(len(train_all_file['images'])))
    print('all: {:d}'.format(len(test_file['images'])))
    train_all_file.close()
    test_file.close()


def build_fashionMNIST_h5(data_dir, save_dir):
    dataset = datasets.FashionMNIST(root=data_dir, download=True, transform=None)
    test_dataset = datasets.FashionMNIST(root=data_dir, download=True, train=False, transform=None)
    train_all_file = h5py.File(os.path.join(save_dir, 'train_fashionMNIST.h5'), 'w')
    test_file = h5py.File(os.path.join(save_dir, 'test_fashionMNIST.h5'), 'w')

    print('Processing training files')
    for i in range(len(dataset)):
        image, label = dataset[i]
        train_all_file.create_dataset('images/{:d}'.format(i), data=image)
        train_all_file.create_dataset('labels/{:d}'.format(i), data=label)

    print('Processing test files')
    for i in range(len(test_dataset)):
        image, label = test_dataset[i]
        test_file.create_dataset('images/{:d}'.format(i), data=image)
        test_file.create_dataset('labels/{:d}'.format(i), data=label)

    print('all: {:d}'.format(len(train_all_file['images'])))
    print('all: {:d}'.format(len(test_file['images'])))
    train_all_file.close()
    test_file.close()


def build_h5_MNIST_unique_fashionMNIST_uniform(mnist_data_dir, fashion_mnist_data_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    MNIST_dataset = datasets.MNIST(root=mnist_data_dir, download=True, transform=None)
    fashionMNIST_dataset = datasets.FashionMNIST(root=fashion_mnist_data_dir, download=True, transform=None)

    train_files = []
    for i in range(10):
        train_files.append(h5py.File(os.path.join(save_dir, 'train_MNIST_unique_fashionMNIST_uniform_{:d}.h5'.format(i)), 'w'))

    print('Processing MNIST files')
    for idx in range(len(MNIST_dataset)):
        image, label = MNIST_dataset[idx]
        train_files[label].create_dataset('images/MNIST_{:d}'.format(idx), data=image)
        train_files[label].create_dataset('labels/MNIST_{:d}'.format(idx), data=label)

    print('Processing fashionMNIST files')
    indices = list(range(len(fashionMNIST_dataset)))
    random.shuffle(indices)

    N = len(fashionMNIST_dataset) // 10
    for i in range(10):
        idx_list = indices[N*i:N*(i+1)]
        for idx in idx_list:
            image, label = fashionMNIST_dataset[idx]
            train_files[i].create_dataset('images/fashionMNIST_{:d}'.format(idx), data=image)
            train_files[i].create_dataset('labels/fashionMNIST_{:d}'.format(idx), data=label)

    for i in range(10):
        print('{:d}: {:d}'.format(i, len(train_files[i]['images'])))
        train_files[i].close()


def build_h5_MNIST_unique_font_uniform(mnist_data_dir, font_data_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    MNIST_dataset = h5py.File('{:s}/train_MNIST.h5'.format(mnist_data_dir), 'r')
    font_dataset = h5py.File('{:s}/font_train_db_new.h5'.format(font_data_dir), 'r')

    train_file_all = h5py.File(os.path.join(save_dir, 'train_MNIST_font.h5'), 'w')
    train_files = []
    for i in range(10):
        train_files.append(h5py.File(os.path.join(save_dir, 'train_MNIST_unique_font_uniform_{:d}.h5'.format(i)), 'w'))

    print('Processing MNIST files')
    for key in MNIST_dataset['images'].keys():
        image, label = MNIST_dataset['images/{:s}'.format(key)][()], MNIST_dataset['labels/{:s}'.format(key)][()]
        train_file_all.create_dataset('images/MNIST_{:s}'.format(key), data=image)
        train_file_all.create_dataset('labels/MNIST_{:s}'.format(key), data=label)
        train_files[label].create_dataset('images/MNIST_{:s}'.format(key), data=image)
        train_files[label].create_dataset('labels/MNIST_{:s}'.format(key), data=label)

    print('Processing font files')
    keys = list(font_dataset['images'].keys())
    random.shuffle(keys)

    N = len(keys) // 10
    for i in range(10):
        key_list = keys[N*i:N*(i+1)]
        for key in key_list:
            image, label = font_dataset['images/{:s}'.format(key)][()], font_dataset['labels/{:s}'.format(key)][()]
            train_file_all.create_dataset('images/font_{:s}'.format(key), data=image)
            train_file_all.create_dataset('labels/font_{:s}'.format(key), data=label)
            train_files[i].create_dataset('images/font_{:s}'.format(key), data=image)
            train_files[i].create_dataset('labels/font_{:s}'.format(key), data=label)

    print('All: {:d}'.format(len(train_file_all['images'])))
    train_file_all.close()
    for i in range(10):
        print('{:d}: {:d}'.format(i, len(train_files[i]['images'])))
        train_files[i].close()


def build_h5_MNIST_unique_fashionMNIST_and_font_uniform(mnist_data_dir, fashion_mnist_data_dir, font_data_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    MNIST_dataset = datasets.MNIST(root=mnist_data_dir, download=True, transform=None)
    fashionMNIST_dataset = datasets.FashionMNIST(root=fashion_mnist_data_dir, download=True, transform=None)
    font_dataset = h5py.File('{:s}/font_train_db_new.h5'.format(font_data_dir), 'r')

    train_files = []
    for i in range(10):
        train_files.append(h5py.File(os.path.join(save_dir, 'train_MNIST_unique_fashionMNIST_and_font_uniform_{:d}.h5'.format(i)), 'w'))

    print('Processing MNIST files')
    for idx in range(len(MNIST_dataset)):
        image, label = MNIST_dataset[idx]
        train_files[label].create_dataset('images/MNIST_{:d}'.format(idx), data=image)
        train_files[label].create_dataset('labels/MNIST_{:d}'.format(idx), data=label)

    print('Processing fashionMNIST files')
    indices = list(range(len(fashionMNIST_dataset)))
    random.shuffle(indices)
    N = len(fashionMNIST_dataset) // 10
    for i in range(10):
        idx_list = indices[N*i:N*(i+1)]
        for idx in idx_list:
            image, label = fashionMNIST_dataset[idx]
            train_files[i].create_dataset('images/fashionMNIST_{:d}'.format(idx), data=image)
            train_files[i].create_dataset('labels/fashionMNIST_{:d}'.format(idx), data=label)

    print('Processing font files')
    keys = list(font_dataset['images'].keys())
    random.shuffle(keys)
    N = len(keys) // 10
    for i in range(10):
        key_list = keys[N * i:N * (i + 1)]
        for key in key_list:
            image, label = font_dataset['images/{:s}'.format(key)][()], font_dataset['labels/{:s}'.format(key)][()]
            train_files[i].create_dataset('images/font_{:s}'.format(key), data=image)
            train_files[i].create_dataset('labels/font_{:s}'.format(key), data=label)

    for i in range(10):
        print('{:d}: {:d}'.format(i, len(train_files[i]['images'])))
        train_files[i].close()


def build_h5_MNIST_and_fashionMNIST_uniform(mnist_data_dir, fashion_mnist_data_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    MNIST_dataset = datasets.MNIST(root=mnist_data_dir, download=True, transform=None)
    fashionMNIST_dataset = datasets.FashionMNIST(root=fashion_mnist_data_dir, download=True, transform=None)

    train_files = []
    for i in range(10):
        train_files.append(h5py.File(os.path.join(save_dir, 'train_MNIST_and_fashionMNIST_uniform_{:d}.h5'.format(i)), 'w'))

    print('Processing MNIST files')
    indices = list(range(len(MNIST_dataset)))
    random.shuffle(indices)
    N = len(MNIST_dataset) // 10
    for i in range(10):
        idx_list = indices[N * i:N * (i + 1)]
        for idx in idx_list:
            image, label = MNIST_dataset[idx]
            train_files[i].create_dataset('images/MNIST_{:d}'.format(idx), data=image)
            train_files[i].create_dataset('labels/MNIST_{:d}'.format(idx), data=label)

    print('Processing fashionMNIST files')
    indices = list(range(len(fashionMNIST_dataset)))
    random.shuffle(indices)
    N = len(fashionMNIST_dataset) // 10
    for i in range(10):
        idx_list = indices[N * i:N * (i + 1)]
        for idx in idx_list:
            image, label = fashionMNIST_dataset[idx]
            train_files[i].create_dataset('images/fashionMNIST_{:d}'.format(idx), data=image)
            train_files[i].create_dataset('labels/fashionMNIST_{:d}'.format(idx), data=label)

    for i in range(10):
        print('{:d}: {:d}'.format(i, len(train_files[i]['images'])))
        train_files[i].close()


def build_h5_MNIST_and_font_uniform(mnist_data_dir, font_data_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    MNIST_dataset = h5py.File('{:s}/train_MNIST.h5'.format(mnist_data_dir), 'r')
    font_dataset = h5py.File('{:s}/font_train_db_new.h5'.format(font_data_dir), 'r')

    train_files = []
    for i in range(10):
        train_files.append(h5py.File(os.path.join(save_dir, 'train_MNIST_and_font_uniform_{:d}.h5'.format(i)), 'w'))

    print('Processing MNIST files')
    keys = list(MNIST_dataset['images'].keys())
    random.shuffle(keys)
    N = len(keys) // 10
    for i in range(10):
        key_list = keys[N*i:N*(i+1)]
        for key in key_list:
            image, label = MNIST_dataset['images/{:s}'.format(key)][()], MNIST_dataset['labels/{:s}'.format(key)][()]
            train_files[i].create_dataset('images/MNIST_{:s}'.format(key), data=image)
            train_files[i].create_dataset('labels/MNIST_{:s}'.format(key), data=label)

    print('Processing font files')
    keys = list(font_dataset['images'].keys())
    random.shuffle(keys)
    N = len(keys) // 10
    for i in range(10):
        key_list = keys[N*i:N*(i+1)]
        for key in key_list:
            image, label = font_dataset['images/{:s}'.format(key)][()], font_dataset['labels/{:s}'.format(key)][()]
            train_files[i].create_dataset('images/font_{:s}'.format(key), data=image)
            train_files[i].create_dataset('labels/font_{:s}'.format(key), data=label)

    for i in range(10):
        print('{:d}: {:d}'.format(i, len(train_files[i]['images'])))
        train_files[i].close()


def build_h5_MNIST_and_fashionMNIST_and_font_uniform(mnist_data_dir, fashion_mnist_data_dir, font_data_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    MNIST_dataset = datasets.MNIST(root=mnist_data_dir, download=True, transform=None)
    fashionMNIST_dataset = datasets.FashionMNIST(root=fashion_mnist_data_dir, download=True, transform=None)
    font_dataset = h5py.File('{:s}/font_train_db_new.h5'.format(font_data_dir), 'r')

    train_files = []
    for i in range(10):
        train_files.append(h5py.File(os.path.join(save_dir, 'train_MNIST_and_fashionMNIST_and_font_uniform_{:d}.h5'.format(i)), 'w'))

    print('Processing MNIST files')
    indices = list(range(len(MNIST_dataset)))
    random.shuffle(indices)
    N = len(MNIST_dataset) // 10
    for i in range(10):
        idx_list = indices[N * i:N * (i + 1)]
        for idx in idx_list:
            image, label = MNIST_dataset[idx]
            train_files[i].create_dataset('images/MNIST_{:d}'.format(idx), data=image)
            train_files[i].create_dataset('labels/MNIST_{:d}'.format(idx), data=label)

    print('Processing fashionMNIST files')
    indices = list(range(len(fashionMNIST_dataset)))
    random.shuffle(indices)
    N = len(fashionMNIST_dataset) // 10
    for i in range(10):
        idx_list = indices[N * i:N * (i + 1)]
        for idx in idx_list:
            image, label = fashionMNIST_dataset[idx]
            train_files[i].create_dataset('images/fashionMNIST_{:d}'.format(idx), data=image)
            train_files[i].create_dataset('labels/fashionMNIST_{:d}'.format(idx), data=label)

    print('Processing font files')
    keys = list(font_dataset['images'].keys())
    random.shuffle(keys)
    N = len(keys) // 10
    for i in range(10):
        key_list = keys[N*i:N*(i+1)]
        for key in key_list:
            image, label = font_dataset['images/{:s}'.format(key)][()], font_dataset['labels/{:s}'.format(key)][()]
            train_files[i].create_dataset('images/font_{:s}'.format(key), data=image)
            train_files[i].create_dataset('labels/font_{:s}'.format(key), data=label)

    for i in range(10):
        print('{:d}: {:d}'.format(i, len(train_files[i]['images'])))
        train_files[i].close()


def build_MNIST_font_h5(mnist_data_dir, font_data_dir, save_dir, mode):
    os.makedirs(save_dir, exist_ok=True)
    MNIST_dataset = h5py.File('{:s}/{:s}_MNIST.h5'.format(mnist_data_dir, mode), 'r')
    font_dataset = h5py.File('{:s}/font_{:s}_db_new.h5'.format(font_data_dir, mode), 'r')

    test_file = h5py.File(os.path.join(save_dir, '{:s}_MNIST_font.h5'.format(mode)), 'w')

    print('Processing MNIST files')
    for key in MNIST_dataset['images'].keys():
        image, label = MNIST_dataset['images/{:s}'.format(key)][()], MNIST_dataset['labels/{:s}'.format(key)][()]
        test_file.create_dataset('images/MNIST_{:s}'.format(key), data=image)
        test_file.create_dataset('labels/MNIST_{:s}'.format(key), data=label)

    print('Processing font files')
    for key in font_dataset['images'].keys():
        image, label = font_dataset['images/{:s}'.format(key)][()], font_dataset['labels/{:s}'.format(key)][()]
        test_file.create_dataset('images/font_{:s}'.format(key), data=image)
        test_file.create_dataset('labels/font_{:s}'.format(key), data=label)

    print('All: {:d}'.format(len(test_file['images'])))
    test_file.close()


def build_MNIST_fashionMNIST_h5(mnist_data_dir, fashion_mnist_data_dir, save_dir, mode):
    os.makedirs(save_dir, exist_ok=True)
    train_flag = True if mode == 'train' else False
    MNIST_dataset = datasets.MNIST(root=mnist_data_dir, train=train_flag, download=True, transform=None)
    fashionMNIST_dataset = datasets.FashionMNIST(root=fashion_mnist_data_dir, train=train_flag, download=True, transform=None)

    test_file = h5py.File(os.path.join(save_dir, '{:s}_MNIST_fashionMNIST.h5'.format(mode)), 'w')

    print('Processing MNIST files')
    for idx in range(len(MNIST_dataset)):
        image, label = MNIST_dataset[idx]
        test_file.create_dataset('images/MNIST_{:d}'.format(idx), data=image)
        test_file.create_dataset('labels/MNIST_{:d}'.format(idx), data=label)

    print('Processing fashionMNIST files')
    for idx in range(len(fashionMNIST_dataset)):
        image, label = fashionMNIST_dataset[idx]
        test_file.create_dataset('images/fashionMNIST_{:d}'.format(idx), data=image)
        test_file.create_dataset('labels/fashionMNIST_{:d}'.format(idx), data=label)

    print('All: {:d}'.format(len(test_file['images'])))
    test_file.close()


def build_MNIST_fashionMNIST_font_h5(mnist_data_dir, fashion_mnist_data_dir, font_data_dir, save_dir, mode):
    os.makedirs(save_dir, exist_ok=True)
    train_flag = True if mode == 'train' else False
    MNIST_dataset = datasets.MNIST(root=mnist_data_dir, train=train_flag, download=True, transform=None)
    fashionMNIST_dataset = datasets.FashionMNIST(root=fashion_mnist_data_dir, train=train_flag, download=True, transform=None)
    font_dataset = h5py.File('{:s}/font_{:s}_db_new.h5'.format(font_data_dir, mode), 'r')

    test_file = h5py.File(os.path.join(save_dir, '{:s}_MNIST_fashionMNIST_font.h5'.format(mode)), 'w')

    print('Processing MNIST files')
    for idx in range(len(MNIST_dataset)):
        image, label = MNIST_dataset[idx]
        test_file.create_dataset('images/MNIST_{:d}'.format(idx), data=image)
        test_file.create_dataset('labels/MNIST_{:d}'.format(idx), data=label)

    print('Processing fashionMNIST files')
    for idx in range(len(fashionMNIST_dataset)):
        image, label = fashionMNIST_dataset[idx]
        test_file.create_dataset('images/fashionMNIST_{:d}'.format(idx), data=image)
        test_file.create_dataset('labels/fashionMNIST_{:d}'.format(idx), data=label)

    print('Processing font files')
    for key in font_dataset['images'].keys():
        image, label = font_dataset['images/{:s}'.format(key)][()], font_dataset['labels/{:s}'.format(key)][()]
        test_file.create_dataset('images/font_{:s}'.format(key), data=image)
        test_file.create_dataset('labels/font_{:s}'.format(key), data=label)

    print('All: {:d}'.format(len(test_file['images'])))
    test_file.close()


# ICLR rebuttal: imbalanced data

def build_h5_MNIST_unique_fashionMNIST_uniform_imbalanced(mnist_fashionmnist_data_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(5):
        original_file = h5py.File('{:s}/train_MNIST_unique_fashionMNIST_uniform_{:d}.h5'.format(mnist_fashionmnist_data_dir, i), 'r')
        train_file = h5py.File('{:s}/train_MNIST_unique_fashionMNIST_uniform_{:d}.h5'.format(save_dir, i), 'w')
        keys = list(original_file['images'].keys())
        random.shuffle(keys)
        for key in keys[:len(keys)//2]:
            image = original_file['images/{:s}'.format(key)][()]
            label = original_file['labels/{:s}'.format(key)][()]
            train_file.create_dataset('images/{:s}'.format(key), data=image)
            train_file.create_dataset('labels/{:s}'.format(key), data=label)

        print('{:d}: {:d}'.format(i, len(train_file['images'])))
        train_file.close()
        original_file.close()


def build_h5_MNIST_fashionMNIST_imbalanced(mnist_fashionmnist_data_dir):
    train_file = h5py.File('{:s}/train_MNIST_fashionMNIST.h5'.format(mnist_fashionmnist_data_dir), 'w')
    for i in range(10):
        original_file = h5py.File('{:s}/train_MNIST_unique_fashionMNIST_uniform_{:d}.h5'.format(mnist_fashionmnist_data_dir, i), 'r')
        for key in original_file['images'].keys():
            image = original_file['images/{:s}'.format(key)][()]
            label = original_file['labels/{:s}'.format(key)][()]
            train_file.create_dataset('images/{:s}'.format(key), data=image)
            train_file.create_dataset('labels/{:s}'.format(key), data=label)
        original_file.close()
    train_file.close()


def extract_imgs(h5_file_path, save_path, max_num=1000):
    h5_file = h5py.File(h5_file_path, 'r')
    os.makedirs('{:s}/images'.format(save_path), exist_ok=True)
    keys = list(h5_file['images'].keys())
    for file_key in tqdm(keys[:max_num]):
        # print(file_key)
        img = h5_file['images/{:s}'.format(file_key)][()]
        if img.shape[0] != 28:
            img = np.array(Image.fromarray(img).resize((28, 28)))
        label = h5_file['labels/{:s}'.format(file_key)][()]
        io.imsave('{:s}/images/{:s}_{:d}.png'.format(save_path, file_key, label), img)


def print_labels():
    # h5_file = h5py.File('../MNIST_fashionMNIST_imbalanced/train_MNIST_unique_fashionMNIST_uniform_9.h5', 'r')
    # h5_file = h5py.File('../MNIST_fashionMNIST_imbalanced/train_MNIST_fashionMNIST.h5', 'r')
    # h5_file = h5py.File('../MNIST_font/train_MNIST_and_font_uniform_0.h5', 'r')
    mnist_nums = np.zeros((10, 1))
    fashionmnist_nums = np.zeros((10, 1))
    for key in h5_file['labels'].keys():
        label = h5_file['labels/{:s}'.format(key)]
        if key.startswith('MNIST'):
            mnist_nums[label] += 1
        elif key.startswith('fashion'):
            fashionmnist_nums[label] += 1

    # print(mnist_nums)
    # print(fashionmnist_nums)

    print(sum(mnist_nums))
    print(sum(fashionmnist_nums))
    print(sum(mnist_nums)+sum(fashionmnist_nums))

    # print(mnist_nums / np.sum(mnist_nums))
    # print(fashionmnist_nums / np.sum(fashionmnist_nums))


# build_MNIST_h5('../MNIST', '../MNSIT')
# build_MNIST_fashionMNIST_h5('../MNIST', '../fashionMNIST', '../MNIST_fashionMNIST', mode='train')
# build_MNIST_fashionMNIST_h5('../MNIST', '../fashionMNIST', '../MNIST_fashionMNIST', mode='test')
# build_h5_MNIST_unique_fashionMNIST_uniform('../MNIST', '../fashionMNIST', '../MNIST_fashionMNIST')
# build_h5_MNIST_and_fashionMNIST_uniform('../MNIST', '../fashionMNIST', '../MNIST_fashionMNIST')

# build_h5_MNIST_unique_fashionMNIST_and_font_uniform('../MNIST', '../fashionMNIST', '../Font', '../MNIST_fashionMNIST_font')
# build_h5_MNIST_and_fashionMNIST_and_font_uniform('../MNIST', '../fashionMNIST', '../Font', '../MNIST_fashionMNIST_font')
# build_MNIST_fashionMNIST_font_h5('../MNIST', '../fashionMNIST', '../Font', '../MNIST_fashionMNIST_font', mode='train')
# build_MNIST_fashionMNIST_font_h5('../MNIST', '../fashionMNIST', '../Font', '../MNIST_fashionMNIST_font', mode='test')

# build_h5_MNIST_unique_fashionMNIST_uniform_imbalanced('../MNIST_fashionMNIST', '../MNIST_fashionMNIST_imbalanced')
# build_h5_MNIST_fashionMNIST_imbalanced('../MNIST_fashionMNIST_imbalanced')

# build_MNIST_font_h5('../MNIST', '../Font', 'MNIST_font_new', 'train')
# build_MNIST_font_h5('../MNIST', '../Font', 'MNIST_font_new', 'test')
# build_h5_MNIST_unique_font_uniform('../MNIST', '../Font', '../MNIST_font_new')
# build_h5_MNIST_and_font_uniform('../MNIST', '../Font', '../MNIST_font_new')
print_labels()


# extract_imgs('../MNIST_fashionMNIST/train_MNIST_fashionMNIST.h5', '../MNIST_fashionMNIST/train_images', 500000)
# extract_imgs('../Font/font_train_db_new.h5', '../Font/train_images', 500)


# extract_imgs('../../results/mnistUniqueFashionUniform_MDGAN_10D/test_400/mnistUniqueFashionUniform_MDGAN_10D_cDCGANResnet_epoch400_x1.h5',
#              '../../results/mnistUniqueFashionUniform_MDGAN_10D/test_400/images', 500000)
# extract_imgs('../../results/mnistUniqueFontUniform_MDGAN_10D/test_400/mnistUniqueFontUniform_MDGAN_10D_cDCGANResnet_epoch400_x1.h5',
#              '../../results/mnistUniqueFontUniform_MDGAN_10D/test_400/images', 500000)

# extract_imgs('../../results/mnistAndFashionUniform_MDGAN_10D/test_400/mnistAndFashionUniform_MDGAN_10D_cDCGANResnet_epoch400_x1.h5',
#              '../../results/mnistAndFashionUniform_MDGAN_10D/test_400/images', 500000)
# extract_imgs('../../results/mnistAndFontUniform_MDGAN_10D/test_400/mnistAndFontUniform_MDGAN_10D_cDCGANResnet_epoch400_x1.h5',
#              '../../results/mnistAndFontUniform_MDGAN_10D/test_400/images', 500000)


# extract_imgs('../MNIST_font_new/train_MNIST_font.h5', '../MNIST_font_new/train_images', 500000)

# names = [
#     # 'mnist_Fashion_Uniform_daddcgan_resnetGandD_10D_avg_GP',
#     # 'mnist_Fashion_Uniform_daddcgan_resnetGandD_10D_newEqn_GP',
#     # 'mnist_Fashion_Uniform_mddcgan_resnetGandD_10D_GP',
#     # 'mnistUniqueFashionUniform_daddcgan_resnetGandD_10D_avg_GP',
#     # 'mnistUniqueFashionUniform_daddcgan_resnetGandD_10D_newEqn_GP',
#     # 'mnistUniqueFashionUniform_mddcgan_resnetGandD_10D_GP',
#     'mnistUniqueFontUniform_UAGAN_10D_new',
#     'mnistUniqueFontUniform_MDGAN_10D_new'
# ]
#
# for name in names:
#     print(name)
#     h5_filepath = '/research/cbim/medical/hq43/run_projects/UAGAN/results/{:s}/test_400/{:s}_cDCGANResnet_epoch400_x1.h5'.format(name, name)
#     save_dir = '/research/cbim/medical/hq43/run_projects/UAGAN/results/{:s}/test_400/images'.format(name)
#     extract_imgs(h5_filepath, save_dir, 500000)