import numpy as np
import torch
import os

from PIL import Image
from torchvision import transforms
from torchvision.datasets import DatasetFolder


class BloodCellsDataset(DatasetFolder):
    def __init__(self, root, augment=False, image_size=None, masks=True):
        super(BloodCellsDataset, self).__init__(os.path.join(root, 'images'), self.loader, ['.jpeg'])
        self.augment = augment
        self.image_size = image_size
        self.masks = masks
        self.post_transform = transforms.ToTensor()

    def transformation(self, imgs):
        for i in range(len(imgs)):
            imgs[i] = transforms.functional.resize(imgs[i], self.image_size)

        if self.augment:
            img_sz = (int(1.1*self.image_size[0]), int(1.1*self.image_size[1]))
            for i in range(len(imgs)):
                imgs[i] = transforms.functional.resize(imgs[i], img_sz)

            if self.train:
                params = transforms.RandomCrop.get_params(imgs[0], output_size=self.image_size)
                for i in range(len(imgs)):
                    imgs[i] = transforms.functional.crop(imgs[i], *params)
                params = transforms.RandomRotation.get_params((-30,30))
                for i in range(len(imgs)):
                    imgs[i] = transforms.functional.rotate(imgs[i], params)
            else:
                for i in range(len(imgs)):
                    imgs[i] = transforms.functional.center_crop(imgs[i], self.image_size)

            if self.train:
                if np.random.rand() > 0.5:
                    for i in range(len(imgs)):
                        imgs[i] = transforms.functional.hflip(imgs[i])

        if self.post_transform is not None:
            for i in range(len(imgs)):
                imgs[i] = self.post_transform(imgs[i])
        return imgs

    def loader(self, image_path):
        x = Image.open(image_path)
        if self.masks:
            mask_path = image_path.replace('images', 'masks')
            if os.path.exists(mask_path):
                z = Image.open(mask_path)
            else:
                z = None
            return (x,z)
        else:
            return x

    def __getitem__(self, index):
        if self.masks:
            (x,z), y = super(BloodCellsDataset, self).__getitem__(index)
            if z is None:
                x = self.transformation([x])[0]
                return x,torch.ones_like(x),y
            else:
                imgs = self.transformation([x,z])
                x = imgs[0]
                z = imgs[1]
                return [x,z,y]
        else:
            x, y = super(BloodCellsDataset, self).__getitem__(index)
            x = self.transformation([x])[0]
            return [x,y]
