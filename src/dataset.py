import numpy as np
import torch
import pandas as pd
import xml.etree.ElementTree as ET

from PIL import Image
from torchvision import transforms
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


class FacesDataset(Dataset):
	def __init__(self, meta_file, image_size, augment=False, masks=True, size=None, num_masks=None, random_seed=1234, mask_only=False):
		super(Dataset, self).__init__()
		self.augment = augment
		self.image_size = (image_size, image_size)
		self.masks = masks
		self.post_transform = transforms.ToTensor()
		self.mask_only = mask_only

		self.df = pd.read_csv(meta_file)

		if mask_only:
			self.df = self.df[self.df.mask_path != 'none'].sample(frac=1, random_state=random_seed).reset_index(drop=True)
			self.num_masks = len(self.df)
		elif masks:
			self.df = pd.concat([
				self.df[self.df.mask_path != 'none'].sample(frac=1, random_state=random_seed),
				self.df[self.df.mask_path == 'none'].sample(frac=1, random_state=random_seed),
			]).reset_index(drop=True)

			self.num_masks = (self.df.mask_path != 'none').sum()
			if num_masks is not None:
				self.num_masks = min(num_masks, self.num_masks)
		else:
			self.num_masks = None

		if size is not None:
			self.df = self.df.iloc[:size]

		if masks:
			self.num_masks = min(len(self.df), self.num_masks)
			self.masks_idx = [1]*self.num_masks + [0]*(len(self.df) - self.num_masks)

		print('Loaded {} instances with {} masks!'.format(len(self.df), self.num_masks if self.num_masks is not None else 0))

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

	def get_label_id(self, label):
		LABEL_IDS = {
			'm.010p3': 0,
			'm.010ngb': 1,
			'm.010g87': 2,
			'm.010bk0': 3,
			'm.010lz5': 4,
			'm.010hn': 5
		}

		return LABEL_IDS[label]

	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):
		d = self.df.iloc[[index]].get_values()[0]
		x = Image.open(d[0])
		y = self.get_label_id(d[-1])
		if self.masks:
			if index < self.num_masks:
				root = ET.parse(d[1]).getroot()
				p = [int(i.text) for i in root[-1][-1].getchildren()]
				z = np.zeros(x.size[::-1])
				z[p[1]:p[3],p[0]:p[2]] = 1
				z = Image.fromarray(z)
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
