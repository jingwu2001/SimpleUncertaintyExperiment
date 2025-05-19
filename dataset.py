#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:05:05 2021

@author: laurent
"""




import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision

from soft_dataset import SoftDataset, CIFAR10HCombinedLabels


def getSets(name='mnist', filteredClass=None, removeFiltered=True):
	if name == 'mnist':
		return getSetsMNIST(filteredClass=filteredClass, removeFiltered=removeFiltered)
	elif name == "cifar10":
		return getSetsCIFAR(filteredClass=filteredClass, removeFiltered=removeFiltered)

def getSetsMNIST(filteredClass = None, removeFiltered = True) :
	"""
	Return a torch dataset
	"""
	
	train = torchvision.datasets.MNIST('./data/', train=True, download=True,
								transform=torchvision.transforms.Compose([
										torchvision.transforms.ToTensor(),
										torchvision.transforms.Normalize((0.1307,), (0.3081,))
								]))

	test = torchvision.datasets.MNIST('./data/', train=False, download=True,
								transform=torchvision.transforms.Compose([
										torchvision.transforms.ToTensor(),
										torchvision.transforms.Normalize((0.1307,), (0.3081,))
								]))

	if filteredClass is not None :
		
		train_loader = torch.utils.data.DataLoader(train, batch_size=len(train), num_workers=20, pin_memory=True)
	
		train_labels = next(iter(train_loader))[1].squeeze()
		
		test_loader = torch.utils.data.DataLoader(test, batch_size=len(test), num_workers=20, pin_memory=True)
	
		test_labels = next(iter(test_loader))[1].squeeze()
		
		if removeFiltered : 
			trainIndices = torch.nonzero(train_labels != filteredClass).squeeze()
			testIndices = torch.nonzero(test_labels != filteredClass).squeeze()
		else :
			trainIndices = torch.nonzero(train_labels == filteredClass).squeeze()
			testIndices = torch.nonzero(test_labels == filteredClass).squeeze()
		
		train = torch.utils.data.Subset(train, trainIndices)
		test = torch.utils.data.Subset(test, testIndices)
	
	return train, test


def getSetsCIFAR(filteredClass = None, removeFiltered = True) :
	"""
	Return a torch dataset (without OOD transformations)
	"""
	mean = 0.4914,0.4822,0.4465
	std = 0.2023,0.1994,0.2010 # the values are from the commands from wandb, used by untangle authors
	train_tf = torchvision.transforms.Compose([
		torchvision.transforms.RandomCrop(32, padding=4),
		torchvision.transforms.RandomHorizontalFlip(0.5),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize(mean, std),
	])
	test_tf = torchvision.transforms.Compose([
		# CenterCrop(32) is a no-op, but the helper still calls it
		torchvision.transforms.CenterCrop(32),  
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize(mean, std),
	])



	train = torchvision.datasets.CIFAR10('./data/', train=True, download=True,
								transform=train_tf)
	
	# two test datasets: one with hard label, one with soft labels
	soft_ds = SoftDataset("CIFAR10H", split="all", transform=test_tf) # uses the SoftDataset class from untangle
	hard_ds  = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=test_tf)
	

	def name_to_test_id(name):
		"""Converts the filename for an image to the corresponding testID. """
		return int(name[-8:-4])

	# Make sure the soft and hard datasets contain the exact same set of images
	c = 0
	for imgpath in soft_ds.file_path_to_img_id:
		test_id = name_to_test_id(imgpath)
		test_img = hard_ds[test_id][0]
		soft_id = soft_ds.file_path_to_img_id[imgpath]
		soft_img = soft_ds[soft_id][0]
		majority_vote  = soft_ds[soft_id][1][-1]
		test_label = hard_ds[test_id][1]
		# assert majority_vote == test_label, f"majority vote: {majority_vote}; test_label: {test_label}, {k}"
		assert (np.array(test_img) == np.array(soft_img)).all()
		c += 1
	# print(c)
	# Assertions are true. So the images are exactly the same. Now we only have to put together the soft labels and the hard labels

	soft_loader = torch.utils.data.DataLoader(soft_ds, batch_size=len(soft_ds), shuffle=False)
	hard_loader = torch.utils.data.DataLoader(hard_ds, batch_size=len(hard_ds), shuffle=False)
	soft_labels = next(iter(soft_loader))[1]
	hard_labels = next(iter(hard_loader))[1]
	file_paths = list(soft_ds.file_path_to_img_id.keys())
	test_ids = np.array([name_to_test_id(fp) for fp in file_paths])

	# compute the permutation that would sort soft_labels into test‐order
	perm = np.argsort(test_ids)    # now test_ids[perm] = [0,1,2,...,N-1]

	# reorder soft_labels once
	soft_in_test_order = soft_labels[perm]  # (N,11)

	# 5) make sure hard_labels is (N,1)
	if hard_labels.ndim == 1:
		hard_labels = hard_labels.unsqueeze(1)

	# 6) concat along the last dimension → (N,12)
	combined_labels = torch.cat([soft_in_test_order, hard_labels], dim=1)
	test = CIFAR10HCombinedLabels(hard_ds, combined_labels)
	
	# print(combined_labels.shape)  # should be (10000, 12)
	
	if filteredClass is not None :
		
		train_loader = torch.utils.data.DataLoader(train, batch_size=len(train), num_workers=20, pin_memory=True)
	
		train_labels = next(iter(train_loader))[1].squeeze()
		
		test_loader = torch.utils.data.DataLoader(test, batch_size=len(test), num_workers=20, pin_memory=True)
	
		test_labels = next(iter(test_loader))[1] # Should be (10000, 12)
		
		if removeFiltered : 
			trainIndices = torch.nonzero(train_labels != filteredClass).squeeze()
			testIndices = torch.nonzero(test_labels[:, -1] != filteredClass).squeeze()
		else :
			trainIndices = torch.nonzero(train_labels == filteredClass).squeeze()
			testIndices = torch.nonzero(test_labels[:, -1] == filteredClass).squeeze()
		
		train = torch.utils.data.Subset(train, trainIndices)
		test = torch.utils.data.Subset(test, testIndices)
	
	return train, test


if __name__ == "__main__" :
	
	#test getSets function
	train, test = getSets(filteredClass = 3, removeFiltered = False)
	
	test_loader = torch.utils.data.DataLoader(test, batch_size=len(test), num_workers=20, pin_memory=True)
	
	images, labels = next(iter(test_loader))
	
	print(images.shape)
	print(torch.unique(labels.squeeze()))