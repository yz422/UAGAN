import os.path

import h5py
import numpy as np
from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform


class MNISTFashionMNISTFontDataset(BaseDataset):
    """A dataset class for label-image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt, idx=None):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        if idx is None:
            h5_name = "train_MNIST_fashionMNIST_font.h5"
        else:
            h5_name = 'train_MNIST_unique_fashionMNIST_and_font_uniform_{:d}.h5'.format(idx)
            # h5_name = 'train_MNIST_and_fashionMNIST_and_font_uniform_{:d}.h5'.format(idx)

        print(f"Load: {h5_name}")
        self.is_test = True
        self.extend_len = 0
        BaseDataset.__init__(self, opt)
        self.file = h5py.File(os.path.join(opt.dataroot, h5_name), 'r')

        if 'train' in self.file:
            train_db = self.file['train']
        else:
            train_db = self.file
        self.image, self.label = self.build_pairs(train_db)
        # self.image, self.label, self.fine_label = self.build_pairs(train_db)

        # self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        # self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def build_pairs(self, dataset):
        image_arr = []
        label_arr = []
        # fine_label_arr = []

        images = dataset['images']
        labels = dataset['labels']
        # fine_labels = dataset['fine_labels']

        keys = images.keys()
        # selected_keys = np.random.choice(list(keys), int(len(keys)*ratio), replace=False)
        # keys = list(keys)[:2]
        for key in keys:
            img = images[key][()]
            label = labels[key][()]
            # fine_label = fine_labels[key][()]
            image_arr.append(img)
            label_arr.append(label)
            # fine_label_arr.append(fine_label)

        return image_arr, label_arr #, fine_label_arr

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        A = self.label[index]
        B = self.image[index]
        if self.output_nc == 3 and len(B.shape) == 2:
            B = B[:, :, np.newaxis].repeat(3, axis=2)
        B = Image.fromarray(B)

        # fine_label = self.fine_label[index]

        # apply the same transform to both A and B
        # transform_params = get_params(self.opt, A.size)
        # if self.opt.phase.lower() != 'train':
        transform_params = {}
        transform_params['crop_pos'] = (0, 0)
        transform_params['vflip'] = False
        transform_params['hflip'] = False

        self.opt.load_size = 32
        self.opt.preprocess = 'resize'
        # A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        # A = A_transform(A)
        B = B_transform(B)

        # seg[seg < 0] = 0

        return {'A': A, 'B': B, 'A_paths': str(index), 'B_paths': str(index)} #, 'fine_label': fine_label}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image)
