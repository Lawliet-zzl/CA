from __future__ import print_function

import argparse
import csv
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import time
import math
import random
import copy
from tqdm import tqdm
from models import ResNet18, VGG, LeNet, SENet18
import collections
from PIL import Image
from data_loader import *

parser = argparse.ArgumentParser(description='Synthesizing')
parser.add_argument('--name', default='test', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--num_classes', default=10, type=int, help='the number of classes (default: 10)')
parser.add_argument('--dataset', default="CIFAR10", type=str, help='dataset')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--gap', default=200, type=int, help='sampling gap')
parser.add_argument('--T', default=1000, type=int, help='iteration')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--save', default=False, action='store_true')
parser.add_argument('--cf', default=1, type=float,help=[2.5e-5])
parser.add_argument('--delta', default=0, type=float,help='Gaussian noise')
parser.add_argument('--teacher', default="resnet", type=str, help='model type (default: resnet)')
parser.add_argument('--student', default="mirror", type=str, help=['mirror','mlp','adapter','vgg','lenet','senet'])
parser.add_argument('--generator', default="DR", type=str)
args = parser.parse_args()

class DeepInversionFeatureHook():
	def __init__(self, module):
		self.hook = module.register_forward_hook(self.hook_fn)
	def hook_fn(self, module, input, output):
		nch = input[0].shape[1]
		mean = input[0].mean([0, 2, 3])
		var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
		r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
			module.running_mean.data.type(var.type()) - mean, 2)
		self.r_feature = r_feature
	def close(self):
		self.hook.remove()

def build_teacher(model, num_classes):
	if model == "resnet":
		net = ResNet18(num_classes = num_classes).cuda()
	elif model == 'vgg':
		net = VGG(num_classes = num_classes).cuda()
	elif model == 'senet':
		net = SENet18(num_classes = num_classes).cuda()
	elif model == 'lenet':
		net = LeNet(num_classes = num_classes).cuda()
	return net

def get_size(dataset):
	if dataset == 'Imagenet':
		return 224, 30
	else:
		return 32, 2

def adjust_learning_rate(optimizer, epoch, gap = 1500, rate = 0.3):
	if (epoch + 1) % gap == 0:
		for param_group in optimizer.param_groups:
			new_lr = param_group['lr'] * rate
			param_group['lr'] = new_lr

def generate_dataset(model, dataloader, num_iteration, log_name):
	if args.generator == 'DR':
		for epoch, (images, targets) in enumerate(dataloader):
			images = images.cuda()
			generate_batch_CEP(model, images, num_iteration, epoch, log_name)
			torch.cuda.empty_cache()
	elif args.generator == 'DD':
		generate_batch_dream(model, num_iteration, args.seed, log_name, 1e-2, 2)
		torch.cuda.empty_cache()
	elif args.generator == 'DI':
		generate_batch_dream(model, num_iteration, args.seed, log_name, 1e-5, 10)
		torch.cuda.empty_cache()
	elif args.generator == 'RD':
		for epoch in range(500):
			generate_batch_random(model, num_iteration, epoch, log_name)
			torch.cuda.empty_cache()

def generate_batch_random(model, num_iteration, epoch, log_name):
	model.eval()
	imgsize, lim = get_size(args.dataset)
	num_samples = args.batch_size
	samples = torch.randn((num_samples, 3, imgsize, imgsize), requires_grad=True, device='cuda')
	targets = torch.LongTensor([random.randint(0,9) for _ in range(num_samples)]).to('cuda')

	optimizer = optim.Adam([samples], lr=args.lr)
	optimizer.state = collections.defaultdict(dict)
	criterion = nn.CrossEntropyLoss()

	for t in range(num_iteration + 1):
		off1 = random.randint(-lim, lim)
		off2 = random.randint(-lim, lim)
		samples_jit = torch.roll(samples/2, shifts=(off1,off2), dims=(2,3))

		optimizer.zero_grad()
		model.zero_grad()
		outputs = model(samples_jit)
		probabilities = F.softmax(outputs, dim=1)

		loss = criterion(outputs, targets)

		if epoch == 0:
			info = [epoch, t, loss.item(),
			torch.max(probabilities, dim=1)[0].mean().item(),
			optimizer.state_dict()['param_groups'][0]['lr']]
			write_log(log_name, info)

		if t % args.gap == 0 and args.save:
			save_images(samples_jit, t, epoch)

		loss.backward()
		optimizer.step()

def generate_batch_dream(model, num_iteration, epoch, log_name, coff_l2 = 1e-2, coff_f = 2):
	model.eval()
	imgsize, lim = get_size(args.dataset)
	num_samples = args.batch_size
	samples = torch.randn((num_samples, 3, imgsize, imgsize), requires_grad=True, device='cuda')
	targets = torch.LongTensor([random.randint(0,9) for _ in range(num_samples)]).to('cuda')

	optimizer = optim.Adam([samples], lr=args.lr)
	optimizer.state = collections.defaultdict(dict)
	criterion = nn.CrossEntropyLoss()

	loss_r_feature_layers = []
	for module in model.modules():
		if isinstance(module, nn.BatchNorm2d):
			loss_r_feature_layers.append(DeepInversionFeatureHook(module))

	for t in range(num_iteration + 1):
		off1 = random.randint(-lim, lim)
		off2 = random.randint(-lim, lim)
		samples_jit = torch.roll(samples/2, shifts=(off1,off2), dims=(2,3))

		optimizer.zero_grad()
		model.zero_grad()
		outputs = model(samples_jit)
		probabilities = F.softmax(outputs, dim=1)

		loss_ce = criterion(outputs, targets)
		loss_var_l1, loss_var_l2 = regularization_TV(samples_jit)
		loss_l2 = regularization_norm(samples_jit)
		loss_f = regularization_R(loss_r_feature_layers)
		loss = loss_ce + coff_l2*loss_var_l2 + coff_f*loss_f + 3e-8*loss_l2

		if epoch == 0:
			info = [epoch, t, loss.item(), loss_ce.item(), loss_var_l2.item(), loss_l2.item(), loss_f.item(),
			torch.max(probabilities, dim=1)[0].mean().item(),
			optimizer.state_dict()['param_groups'][0]['lr']]
			write_log(log_name, info)

		if t % args.gap == 0 and args.save:
			save_images(samples_jit, t, epoch)

		loss.backward()
		optimizer.step()

def generate_batch_CEP(model, images, num_iteration, epoch, log_name):
	model.eval()
	imgsize, lim = get_size(args.dataset)
	num_samples = images.size(0)
	samples = torch.randn((num_samples, 3, imgsize, imgsize), requires_grad=True, device='cuda')
	optimizer = optim.Adam([samples], lr=args.lr)

	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_iteration)
	optimizer.state = collections.defaultdict(dict)
	criterion = nn.KLDivLoss(reduction='batchmean')

	for t in range(num_iteration + 1):
		off1 = random.randint(-lim, lim)
		off2 = random.randint(-lim, lim)
		# noise = torch.randn_like(samples)
		samples_jit = torch.roll(samples/2, shifts=(off1,off2), dims=(2,3))

		optimizer.zero_grad()
		model.zero_grad()
		outputs = model(samples_jit)
		probabilities = F.softmax(outputs, dim=1)
		log_probabilities = torch.log(probabilities)
		p = F.softmax(model(images), dim=1).detach()
		loss_kl = criterion(log_probabilities, p)

		loss_mse = F.mse_loss(samples_jit, images) 
		loss = loss_kl + args.cf*loss_mse

		if epoch == 0:
			info = [epoch, t, loss.item(), loss_kl.item(), loss_mse.item(),
			torch.max(probabilities, dim=1)[0].mean().item(),
			optimizer.state_dict()['param_groups'][0]['lr']]
			write_log(log_name, info)

		if t % args.gap == 0 and args.save:
			if t == args.T:
				save_images(images, args.T, epoch)
			else:
				save_images(samples_jit, t, epoch)

		loss.backward()
		optimizer.step()
		# adjust_learning_rate(optimizer, t, args.gap, 0.9)
		# scheduler.step()

def regularization_norm(samples):
	return torch.norm(samples, 2)

def regularization_TV(inputs_jit):
	diff1 = inputs_jit[:,:,:,:-1] - inputs_jit[:,:,:,1:]
	diff2 = inputs_jit[:,:,:-1,:] - inputs_jit[:,:,1:,:]
	diff3 = inputs_jit[:,:,1:,:-1] - inputs_jit[:,:,:-1,1:]
	diff4 = inputs_jit[:,:,:-1,:-1] - inputs_jit[:,:,1:,1:]
	loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
	loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() 
	+ (diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
	return loss_var_l1, loss_var_l2

def regularization_R(loss_r_feature_layers):
	loss_distr = sum([mod.r_feature for mod in loss_r_feature_layers])
	return loss_distr

def save_images(samples, t, epoch):
	mean = (0.4914, 0.4822, 0.4465)
	std = (0.2470, 0.2435, 0.2616)
	mean=[-m/s for m, s in zip(mean, std)]
	std=[1/s for s in std]

	data = samples.data.clone()

	idx = args.batch_size*epoch

	root = './dataset/' + args.dataset + '_' + args.teacher + '_' + args.generator + '_' + str(args.T) + '/' + str(t)
	if not os.path.isdir(root):
		os.mkdir(root)
	for i in range(data.size(0)):
		name = root + '/' + str(idx + i + 1) + '.png'
		normalize = transforms.Normalize(mean, std)
		sample = normalize(data[i])
		vutils.save_image(sample, name, normalize=False)

	root = './dataset/' + args.dataset + '_' + args.teacher + '_' + args.generator + '_' + str(args.T) + '_grid'
	if data.size(0) >= 100:
		name = root + '/' + str(epoch) + '_' + str(t) + '.png'
		vutils.save_image(data[0:100,:], name, normalize=True, scale_each=True, nrow=10)

def init_setting():

	# init seed
	if args.seed != 0:
		torch.manual_seed(args.seed)

	# Address
	if not os.path.isdir('dataset'):
		os.mkdir('dataset')

	if not os.path.isdir('results'):
		os.mkdir('results')

	if args.save:
		name = './dataset/' +args.dataset + '_' + args.teacher + '_' + args.generator + '_' + str(args.T)
		# if os.path.isdir(name):
		# 	shutil.rmtree(name)
		# if os.path.isdir(name + '_grid'):
		# 	shutil.rmtree(name + '_grid')
		if not os.path.isdir(name):
			os.mkdir(name)
		if not os.path.isdir(name + '_grid'):
			os.mkdir(name + '_grid')

	pretrained_pth = ('checkpoints/pretrained_' + args.dataset + '_' + args.teacher + '_'  + str(0) + '.pth')
	log_name = ('results/' + args.dataset + '_' + args.generator + '_' + str(args.T) + '.csv')

	with open(log_name, 'w') as logfile:
		logwriter = csv.writer(logfile, delimiter=',')
		if args.generator == 'DR':
			logwriter.writerow(['epoch', 'iteration', 'loss', 'loss_kl', 'loss_mse', 'Confidence','lr'])
		elif args.generator == 'DD':
			logwriter.writerow(['epoch', 'iteration', 'loss', 'loss_ce', 'loss_tv', 'loss_l2', 'loss_f', 'Confidence','lr'])
		elif args.generator == 'RD':
			logwriter.writerow(['epoch', 'iteration', 'loss', 'Confidence','lr'])

	return pretrained_pth, log_name

def write_log(log_name, info):
	with open(log_name, 'a') as logfile:
		logwriter = csv.writer(logfile, delimiter=',')
		logwriter.writerow(info)

def get_CIFAR10_transform():
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
	return transform_train, transform_test

def main():

	pretrained_pth, log_name = init_setting()
	net_teacher = build_teacher(args.teacher, args.num_classes)
	net_teacher.load_state_dict(torch.load(pretrained_pth))

	transform_train, transform_test = get_CIFAR10_transform()
	trainloader, testloader = load_CIFAR10(args.batch_size, transform_test, transform_test)

	print(f"Generating {args.generator} Samples with iteration {args.T} and seed {args.seed}")
	generate_dataset(net_teacher, trainloader, args.T, log_name)
	# print(f"Generating samples with confidence from {min(clist)} to {max(clist)} of seed {args.seed}")

if __name__ == '__main__':
    main()
