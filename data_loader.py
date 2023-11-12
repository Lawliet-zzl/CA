import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from IPython.display import HTML
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from sklearn.metrics import roc_auc_score, average_precision_score

class LoadDataset(Dataset):
	"""docstring for LoadDataset"""
	def __init__(self,  root, list_file='train', transform=None, target_transform=None, full_dir=True):
		super(LoadDataset, self).__init__()
		self.root = root
		self.list_file = list_file
		self.transform = transform
		self.target_transform = target_transform
		self.full_dir = full_dir
		self._parse_list()

	def _load_image(self, directory):
		if self.full_dir:
			return Image.open(directory).convert('RGB')
		else:
			return Image.open(os.path.join(self.root, 'data', directory)).convert('RGB')

	def _parse_list(self):
		self.data_list = [LoadRecord(x.strip().split(' ')) for x in open(os.path.join(self.root, self.list_file))]

	def __getitem__(self, index):
		record = self.data_list[index]

		return self.get(record)

	def get(self, record, indices=None):
		img = self._load_image(record.path)

		process_data = self.transform(img)
		if not self.target_transform == None:
			process_label = self.target_transform(record.label)
		else:
			process_label = record.label

		return process_data, process_label

	def __len__(self):
		return len(self.data_list)

class LoadRecord(object):
	"""docstring for LoadRecord"""
	def __init__(self, data):
		super(LoadRecord, self).__init__()
		self._data = data

	@property
	def path(self):
		return self._data[0]

	@property
	def label(self):
		return int(self._data[1])

def tpr95(soft_IN, soft_OOD, precision):
	#calculate the falsepositive error when tpr is 95%

	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision # precision:200000

	total = 0.0
	fpr = 0.0
	for delta in np.arange(start, end, gap):
		tpr = np.sum(np.sum(X1 >= delta)) / float(len(X1))
		error2 = np.sum(np.sum(Y1 > delta)) / float(len(Y1))
		if tpr <= 0.9505 and tpr >= 0.9495:
			fpr += error2
			total += 1
	if total == 0:
		# print('corner case')
		fprBase = 1
	else:
		fprBase = fpr/total
	return fprBase

def auroc(soft_IN, soft_OOD, precision):
	#calculate the AUROC
	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision
	aurocBase = 0.0
	fprTemp = 1.0
	for delta in np.arange(start, end, gap):
		tpr = np.sum(np.sum(X1 >= delta)) / float(len(X1))
		fpr = np.sum(np.sum(Y1 > delta)) / float(len(Y1))
		aurocBase += (-fpr+fprTemp)*tpr
		fprTemp = fpr
	aurocBase += fpr * tpr
	#improve
	return aurocBase

def auprIn(soft_IN, soft_OOD, precision):
	#calculate the AUPR

	precisionVec = []
	recallVec = []
	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision

	auprBase = 0.0
	recallTemp = 1.0
	for delta in np.arange(start, end, gap):
		tp = np.sum(np.sum(X1 >= delta)) / float(len(X1))
		fp = np.sum(np.sum(Y1 >= delta)) / float(len(Y1))
		if tp + fp == 0: continue
		precision = tp / (tp + fp)
		recall = tp
		precisionVec.append(precision)
		recallVec.append(recall)
		auprBase += (recallTemp-recall)*precision
		recallTemp = recall
	auprBase += recall * precision
	return auprBase

def auprOut(soft_IN, soft_OOD, precision):
	#calculate the AUPR
	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision

	auprBase = 0.0
	recallTemp = 1.0
	for delta in np.arange(end, start, -gap):
		fp = np.sum(np.sum(X1 < delta)) / float(len(X1))
		tp = np.sum(np.sum(Y1 < delta)) / float(len(Y1))
		if tp + fp == 0: break
		precision = tp / (tp + fp)
		recall = tp
		auprBase += (recallTemp-recall)*precision
		recallTemp = recall
	auprBase += recall * precision

	return auprBase

def detection(soft_IN, soft_OOD, precision):
	#calculate the minimum detection error
	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision

	errorBase = 1.0
	for delta in np.arange(start, end, gap):
		tpr = np.sum(np.sum(X1 < delta)) / float(len(X1))
		error2 = np.sum(np.sum(Y1 > delta)) / float(len(Y1))
		errorBase = np.minimum(errorBase, (tpr+error2)/2.0)

	return errorBase

def detect_OOD(soft_ID, soft_OOD, precision=200000):
	detection_results = np.array([0.0,0.0,0.0,0.0,0.0])
	detection_results[0] = auroc(soft_ID, soft_OOD, precision)*100
	detection_results[1] = auprIn(soft_ID, soft_OOD, precision)*100
	detection_results[2] = auprOut(soft_ID, soft_OOD, precision)*100
	detection_results[3] = tpr95(soft_ID, soft_OOD, precision)*100
	detection_results[4] = detection(soft_ID, soft_OOD, precision)*100
	return detection_results

def detect_OOD2(ID_confidence, OOD_confidence):
	ID_labels = np.ones(len(ID_confidence))
	OOD_labels = np.zeros(len(OOD_confidence))
	detection_results = np.array([0.0,0.0,0.0,0.0,0.0])
	detection_results[0] = 100.0*roc_auc_score(np.concatenate((ID_labels, OOD_labels)), np.concatenate((ID_confidence, OOD_confidence)))
	# aupr = average_precision_score(np.concatenate((ID_labels, OOD_labels)), np.concatenate((ID_confidence, OOD_confidence)))
	# fpr, tpr, _ = roc_curve(np.concatenate((ID_labels, OOD_labels)), np.concatenate((ID_confidence, OOD_confidence)))
	# precision, recall, _ = precision_recall_curve(np.concatenate((ID_labels, OOD_labels)), np.concatenate((ID_confidence, OOD_confidence)))
	return detection_results

def generate_miniimagenet_list(root='../../data224/miniimagenet224/', classnum = 10):
	path = root + 'data'
	classname = os.listdir(path)
	#random.shuffle(classname)
	for i in range(classnum):
		# print(i, classname[i])
		images = os.listdir(os.path.join(path, classname[i]))
		m = 'w' if i == 0 else 'a'
		with open(os.path.join(root, 'train_list.txt'), m) as f:
			for j in range(500):
				f.write(classname[i] + '/')
				f.write(images[j] + ' ' + str(i))
				f.write('\n')
		with open(os.path.join(root, 'test_list.txt'), m) as f:
			for j in range(500, len(images)):
				f.write(classname[i] + '/')
				f.write(images[j] + ' ' + str(i))
				f.write('\n')

def generate_synthesized_list(dataset, num_seeds, T, gap, root='./dataset'):
	path = os.path.join(root, dataset)
	classname = os.listdir(path)
	num_samples = 0
	if gap > 0:
		for i in range(len(classname)):
			t = int(classname[i])
			if t % gap == 0 and t <= T:
				images = os.listdir(os.path.join(path, classname[i]))
				m = 'w' if num_samples == 0 else 'a'
				num_subset_samples = len(images) if len(images) < num_seeds else num_seeds
				num_samples += num_subset_samples
				with open(os.path.join(root, dataset + '_list.txt'), m) as f:
					for j in range(num_subset_samples):
						f.write(path+ '/' + str(t) + '/' + images[j] + ' ' + str(t) + '\n')
	else:
		images = os.listdir(os.path.join(path, str(T)))
		num_samples = len(images) if len(images) < num_seeds else num_seeds
		with open(os.path.join(root, dataset + '_list.txt'), 'w') as f:
			for j in range(num_samples):
				f.write(path+ '/' + str(T) + '/' + images[j] + ' ' + str(T) + '\n')

	print("The number of synthesized samples is " + str(num_samples) + ".")

def generate_all_list(root='../../data224/CUB_200_2011/'):
	path = root + 'data'
	classname = os.listdir(path)
	classnum = len(classname)
	#random.shuffle(classname)
	for i in range(classnum):
		images = os.listdir(os.path.join(path, classname[i]))
		m = 'w' if i == 0 else 'a'
		with open(os.path.join(root, 'all_list.txt'), m) as f:
			for j in range(len(images)):
				f.write(classname[i] + '/')
				f.write(images[j] + ' ' + str(i))
				f.write('\n')

def get_OOD_list():
	OOD_list = ['SVHN', 'CIFAR10','CIFAR100', 'imagenet', 
	'TinyImageNet(r)', 'TinyImageNet(c)', 'LSUN(r)','LSUN(c)','iSUN','gaussian','uniform',
	'CUB200', 'StanfordDogs120', 'OxfordPets37', 'Oxfordflowers102', 'Caltech256', 'DTD47', 'COCO']
	return OOD_list

def save_OOD_result(ood_name, detection_results):
	OOD_list = get_OOD_list()
	with open(ood_name, 'a') as logfile:
		logwriter = csv.writer(logfile, delimiter=',')
		for i in range(np.size(detection_results,0)):
			logwriter.writerow([OOD_list[i],
				detection_results[i][0],
				detection_results[i][1],
				detection_results[i][2],
				detection_results[i][3],
				detection_results[i][4]])

def save_table_result(table_name, info, test_acc, ece, ys, detection_results):
	with open(table_name, 'a') as logfile:
		logwriter = csv.writer(logfile, delimiter=',')
		info.append(test_acc)
		info.append(ece)
		info.append(ys)
		info.extend([i[0] for i in detection_results])
		logwriter.writerow(info)

def get_ID_dataset(dataset, batch_size):
	transform_train, transform_test = get_transform(dataset)
	if dataset == 'CIFAR10':
		trainloader, testloader = load_CIFAR10(batch_size, transform_train, transform_test)
	elif dataset == 'CIFAR100':
		trainloader, testloader = load_CIFAR100(batch_size, transform_train, transform_test)
	elif dataset == 'SVHN':
		trainloader, testloader = load_SVHN(batch_size, transform_train, transform_test)
	elif dataset == 'imagenet':
		trainloader, testloader = load_miniimagenet(batch_size, transform_train, transform_test)
	return trainloader, testloader

def get_OOD_dataset(dataset, OOD):
	_, transform_test = get_transform(dataset)

	if OOD == 'CIFAR10':
		_, dataloader = load_CIFAR10(100, transform_test, transform_test)
	elif OOD == 'CIFAR100':
		_, dataloader = load_CIFAR100(100, transform_test, transform_test)
	elif OOD == 'SVHN':
		_, dataloader = load_SVHN(100, transform_test, transform_test)
	elif OOD == 'imagenet':
		_, dataloader = load_miniimagenet(100, transform_test, transform_test)
	else:
		dataloader = load_OOD(OOD, 100, transform_test)
	return dataloader

def get_transform(dataset):
	if dataset == 'CIFAR10':
		mean = (0.4914, 0.4822, 0.4465)
		std = (0.2470, 0.2435, 0.2616)
		transform_train = transforms.Compose([
			transforms.Resize(32),
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
			])
		transform_test = transforms.Compose([
			transforms.Resize((32,32)),
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
			])
	elif dataset == 'CIFAR100':
		mean = (0.5071, 0.4865, 0.4409)
		std = (0.2673, 0.2564, 0.2762)
		transform_train = transforms.Compose([
			transforms.Resize(32),
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
			])
		transform_test = transforms.Compose([
			transforms.Resize((32,32)),
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
			])
	elif dataset == 'SVHN':
		mean = (0.4377, 0.4438, 0.4728)
		std = (0.1980, 0.2010, 0.1970)
		transform_train = transforms.Compose([
			transforms.Resize(32),
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
			])
		transform_test = transforms.Compose([
			transforms.Resize((32,32)),
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
			])
	elif dataset == 'imagenet':
		mean = (.485, .456, .406)
		std=(.229, .224, .225)
		transform = transforms.Compose([
			transforms.Resize(256),
			transforms.RandomCrop((224,224)),
			transforms.ToTensor(),
			transforms.Normalize(mean,std)
			])
		transform_train = transform
		transform_test = transform
	return transform_train, transform_test

def load_CIFAR10(batch_size, transform_train, transform_test):
	trainset = datasets.CIFAR10(root='../../data', train=True, download=True, transform=transform_train)
	testset = datasets.CIFAR10(root='../../data', train=False, download=True, transform=transform_test)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
	return trainloader, testloader

def load_SVHN(batch_size, transform_train, transform_test):
	trainset = datasets.SVHN(root='../../data', split='train', download=False, transform=transform_train)
	testset = datasets.SVHN(root='../../data', split='test', download=False, transform=transform_test)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
	return trainloader, testloader

def load_CIFAR100(batch_size, transform_train, transform_test):
	trainset = datasets.CIFAR100(root='../../data/CIFAR100', train=True, download=True, transform=transform_train)
	testset = datasets.CIFAR100(root='../../data/CIFAR100', train=False, download=True, transform=transform_test)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
	return trainloader, testloader

def load_miniimagenet(batch_size, transform_train, transform_test):
	generate_miniimagenet_list(classnum = 100)
	trainset = LoadDataset(root='../../data224/miniimagenet224', list_file='train_list.txt', transform=transform_train, full_dir=False)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
	testset = LoadDataset(root='../../data224/miniimagenet224', list_file='test_list.txt', transform=transform_test, full_dir=False)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
	return trainloader, testloader

def load_synthesized_samples(batch_size, dataset, num_seeds, T, gap):
	root = './dataset'
	generate_synthesized_list(dataset, num_seeds, T, gap, root = root)
	# mean = (0.5, 0.5, 0.5)
	# std = (0.125, 0.125, 0.125)
	mean = (0.4914, 0.4822, 0.4465)
	std = (0.2470, 0.2435, 0.2616)
	transform = transforms.Compose([
		transforms.Resize(32),
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean,std)])
	dataset = LoadDataset(root = root, list_file= dataset + '_list.txt', transform=transform)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
	return dataloader

def load_OOD(dataset, batch_size, transform, glist=False):
	if dataset == 'COCO':
		dataset = datasets.ImageFolder('./data/COCO',transform=transform)
	elif dataset == 'gaussian':
		dataset = datasets.ImageFolder('./data/gaussian',transform=transform)
	elif dataset == 'uniform':
		dataset = datasets.ImageFolder('./data/uniform',transform=transform)
	elif dataset == 'TinyImageNet(c)':
		dataset = datasets.ImageFolder('./data/Imagenet_crop',transform=transform)
	elif dataset == 'TinyImageNet(r)':
		dataset = datasets.ImageFolder('./data/Imagenet_resize',transform=transform)
	elif dataset == 'LSUN(c)':
		dataset = datasets.ImageFolder('./data/LSUN_crop',transform=transform)
	elif dataset == 'LSUN(r)':
		dataset = datasets.ImageFolder('./data/LSUN_resize',transform=transform)
	elif dataset == 'iSUN':
		dataset = datasets.ImageFolder('./data/iSUN',transform=transform)
	else:
		root = './data/' + dataset + '/'
		if glist:
			generate_all_list(root=root)
		dataset = LoadDataset(root=root, list_file='all_list.txt', transform=transform, full_dir=False)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
	return dataloader

def cal_OOD_scores(dataloader, net1, net2 = None):
	net1.eval()
	if net2 != None:
		net2.eval()
	res = np.array([])
	with torch.no_grad():
		for idx, (inputs, _) in enumerate(dataloader):
			inputs= inputs.cuda()
			outputs = net1(inputs)
			probs = F.softmax(outputs.data, dim=1)
			if net2 != None:
				outputs = net2(inputs)
				probs_extra = F.softmax(outputs.data, dim=1)
				probs += probs_extra
				probs = probs / 2
			softmax_vals, predicted = torch.max(probs, dim=1)
			# print(softmax_vals[idx].item(), predicted[idx].item())
			res = np.append(res, softmax_vals.cpu().numpy())
	return res

def evaluate_detection(dataset, precision, net1, net2 = None):
	OOD_list = get_OOD_list()
	detection_results = np.zeros((len(OOD_list),5))
	trainloader, testloader = get_ID_dataset(dataset, 100)
	soft_ID = cal_OOD_scores(testloader, net1, net2)

	# print(f"Confidence on {dataset}: {np.mean(soft_ID)}")
	for i in range(len(OOD_list)):
		dataset_OOD = OOD_list[i]
		testloader_OOD = get_OOD_dataset(dataset, dataset_OOD)
		soft_OOD = cal_OOD_scores(testloader_OOD, net1, net2)
		detection_results[i,:] = detect_OOD2(soft_ID, soft_OOD)
		# print(f"AUROC and Confidence on {dataset_OOD}: {detection_results[i,0]} and {np.mean(soft_OOD)}")
	return detection_results