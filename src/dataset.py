import os
import cv2 as cv
import numpy as np
import torch
import pandas as pd
import pydicom

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
    def __init__(self, data_root, meta_file, image_size, augment=False, masks=True, size=None, num_masks=None, random_seed=1234, mask_only=False):
        super(Dataset, self).__init__()
        self.augment = augment
        self.image_size = image_size
        self.masks = masks
        self.mask_only = mask_only
        self.data_root = data_root

        self.df = pd.read_csv(meta_file)

        self.df = self.df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        self.num_masks = len(self.df)

        if size is not None:
            self.df = self.df.iloc[:size]

        if masks:
            self.num_masks = min(len(self.df), self.num_masks)
            self.masks_idx = [1]*self.num_masks + [0]*(len(self.df) - self.num_masks)

        print('Loaded {} instances with {} masks!'.format(len(self.df), self.num_masks if self.num_masks is not None else 0))

    def transformation(self, imgs):
        if self.augment:
            img_sz = (int(1.1*self.image_size[0]), int(1.1*self.image_size[1]))
            for i in range(len(imgs)):
                imgs[i] = cv.resize(imgs[i], dsize=img_sz[::-1], interpolation=cv.INTER_LINEAR)

            if self.train:
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
        else:
            for i in range(len(imgs)):
                imgs[i] = cv.resize(imgs[i], dsize=self.image_size[::-1], interpolation=cv.INTER_LINEAR)

        for i in range(len(imgs)):
            imgs[i] = torch.Tensor(imgs[i].astype(np.float))
        imgs[0] /= 65535
        if len(imgs) == 1:
            imgs[1] = (imgs[1]>0)

        return imgs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        d = self.df.iloc[[index]].get_values()[0]
        x = pydicom.read_file(os.path.join(self.data_root, 'images', d[0])).pixel_array
        y = d[-1]
        if self.masks:
            if index < self.num_masks:
                z = pydicom.read_file(os.path.join(self.data_root, 'masks', d[1])).pixel_array
                x,z = self.transformation([x,z])
                m = 1
            else:
                x = self.transformation([x])[0]
                z = torch.ones(1,*self.image_size)
                m = 0
            return [x,y,z,m]
        else:
            x = self.transformation([x])[0]
            return [x,y]
