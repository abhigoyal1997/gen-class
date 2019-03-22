import numpy as np
import torch
import h5py
import cv2 as cv

from torch.utils.data import Dataset, Sampler


class SBatchSampler(Sampler):
    def __init__(self, masks_idx, batch_size, shuffle=False):
        self.batch_size = batch_size

        self.indices_with_masks = []
        self.indices_without_masks = []
        for i,x in enumerate(masks_idx):
            if x:
                self.indices_with_masks.append(i)
            else:
                self.indices_without_masks.append(i)

        num_masks = len(self.indices_with_masks)

        self.len = len(masks_idx) // (batch_size)
        masks_per_batch = num_masks // self.len
        mask_surplus_batches = len(self.indices_with_masks) - masks_per_batch*self.len
        self.masks_per_batch = np.asarray([masks_per_batch]*self.len)
        self.masks_per_batch[:mask_surplus_batches] += 1

        self.shuffle = shuffle

    def __iter__(self):
        i = 0
        if self.shuffle:
            np.random.shuffle(self.indices_with_masks)
            np.random.shuffle(self.indices_without_masks)
        with_masks_used = 0
        without_masks_used = 0
        while i < self.len:
            batch = self.indices_with_masks[with_masks_used:with_masks_used+self.masks_per_batch[i]]
            batch += self.indices_without_masks[without_masks_used:without_masks_used+self.batch_size-self.masks_per_batch[i]]
            yield batch
            with_masks_used += self.masks_per_batch[i]
            without_masks_used += self.batch_size - self.masks_per_batch[i]
            i += 1

    def __len__(self):
        return self.len


class DDSMDataset(Dataset):
    def __init__(self, h5_file, batch_size, image_size=None, augment=False, masks=True, size=None, num_masks=None, random_seed=1234, mask_only=False):
        super(Dataset, self).__init__()
        self.is_init = False
        self.augment = augment
        self.image_size = image_size
        self.masks = masks
        self.mask_only = mask_only
        self.file = h5_file
        self.batch_size = batch_size
        with h5py.File(h5_file, 'r') as hf:
            if size is None:
                self.size = hf['y'].shape[0]
            else:
                self.size = min(size, hf['y'].shape[0])

        self.batch_size = min(self.batch_size, self.size)
        if num_masks is None:
            self.num_masks = self.size
        else:
            self.num_masks = min(self.size, num_masks)

        if masks:
            self.num_masks = min(self.size, self.num_masks)
            self.masks_idx = [1]*self.num_masks + [0]*(self.size - self.num_masks)
        print('Loaded {} instances with {} masks!'.format(self.size, self.num_masks if self.num_masks is not None else 0))

    def init(self):
        data = h5py.File(self.file, 'r')

        self.x = data['x']
        self.y = data['y']
        self.z = data['z']

        if self.image_size is None:
            self.image_size = tuple(self.x[0].shape)
        else:
            self.image_size = tuple(self.image_size)

        self.is_init = True

    def transformation(self, imgs):
        if self.augment:
            img_sz = (int(1.1*self.image_size[0]), int(1.1*self.image_size[1]))
            for i in range(len(imgs)):
                imgs[i] = cv.resize(imgs[i], dsize=img_sz[::-1], interpolation=cv.INTER_LINEAR)

            if self.train:  # FIXME: self.train doesn't exist
                px = np.random.randint(0,img_sz[1]-self.image_size[1])
                py = np.random.randint(0,img_sz[0]-self.image_size[0])
            else:
                px = img_sz[1]//2 - self.image_size[1]//2
                py = img_sz[0]//2 - self.image_size[0]//2

            for i in range(len(imgs)):
                imgs[i] = imgs[i][py:py+self.image_size[0],px:px+self.image_size[1]]

                if np.random.rand() > 0.5:
                    for i in range(len(imgs)):
                        imgs[i] = np.fliplr(imgs[i])

        for i in range(len(imgs)):
            imgs[i] = torch.Tensor(imgs[i].astype(np.float)).unsqueeze(0)
        imgs[0] /= 65535
        if len(imgs) == 2:
            imgs[1] = (imgs[1]>0).float()

        return imgs

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if not self.is_init:
            self.init()

        x = self.x[index].T
        y = self.y[index]
        if self.masks:
            if index < self.num_masks:
                z = self.z[index].T
                x,z = self.transformation([x,z])
                m = 1
            else:
                z = torch.ones(1,*self.image_size)
                x = self.transformation([x])[0]
                m = 0
            return [x,y,z,m]
        else:
            return [x,y]
