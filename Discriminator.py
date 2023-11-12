from __future__ import print_function

import argparse
import csv
import os

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
import time
import math
import random
import copy
from tqdm import tqdm
from models import ResNet18, VGG, ShuffleNetV2, MobileNetV2, DenseNet121, LeNet, SENet18, VIT
from data_loader import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument("--decay_epochs", nargs="+", type=int, default=[100, 150], 
	help="decay learning rate by decay_rate at these epochs")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--name', default='test', type=str, help='name of run')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--decay', default=5e-4, type=float, help='weight decay 5e-4')
parser.add_argument('--precision', default=100000, type=float)
parser.add_argument('--num_classes', default=10, type=int, help='the number of classes (default: 10)')
parser.add_argument('--optimizer', default="SGD", type=str)
parser.add_argument('--scheduler', default="manual", type=str)
parser.add_argument('--train', default=True, action='store_false')
parser.add_argument('--test', default=True, action='store_false')

parser.add_argument('--dataset', default="CIFAR10", type=str, help='dataset')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int, help='total epochs to run')
parser.add_argument('--alpha', default=1, type=float, help=[0.01,0.1,0.2,0.5,1,2,5,10,100])
parser.add_argument('--num_samples', default=50000, type=int, help='the number of samples (default: 500)')
parser.add_argument('--T', default=100, type=int, help=[20,40,60,80,100,120,140,160,180,200])
parser.add_argument('--gap', default=1, type=int, help='sampling gap')
parser.add_argument('--teacher', default="resnet", type=str, help='model type (default: resnet)')
parser.add_argument('--student', default="mirror", type=str, help=['mirror','mlp','adapter','vgg','lenet','senet'])
parser.add_argument('--generator', default="DR", type=str)
parser.add_argument('--temperature', default=1, type=int)
args = parser.parse_args()

class MLP(nn.Module):
	def __init__(self, input_size = 32 * 32 * 3, hidden_size = 512, num_classes = 10):
		super(MLP, self).__init__()
		self.input_size = input_size
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, num_classes)

	def forward(self, x):
		out = x.view(-1, self.input_size)
		out = self.fc1(out)
		out = self.relu(out)
		out = self.fc2(out)
		return out

class Adapter(nn.Module):
	def __init__(self, module, in_features = 10, out_features = 64):
		super(Adapter, self).__init__()
		self.downscale = nn.Linear(in_features, out_features)
		self.upscale = nn.Linear(out_features, in_features)
		self.GELU = nn.GELU()
		self.module = module
	def forward(self, x):
		with torch.no_grad():
			y = self.module(x)
		out = self.downscale(y)
		out = self.GELU(out)
		out = self.upscale(out)
		out = y + out
		return out

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

def build_student(model, num_classes, net_teacher):
	if model == 'mirror':
		net = ResNet18(num_classes = num_classes).cuda()
	elif model == "resnet":
		net = ResNet18(num_classes = num_classes).cuda()
	elif model == 'mlp':
		net = MLP(num_classes = num_classes).cuda()
	elif model == 'adapter':
		net = Adapter(module = net_teacher, in_features = num_classes).cuda()
	elif model == 'vgg':
		net = VGG('VGG19',num_classes=num_classes).cuda()
	elif model == 'senet':
		net = SENet18(num_classes = num_classes).cuda()
	elif model == 'lenet':
		net = LeNet(num_classes = num_classes).cuda()
	elif model == 'shufflenet':
		net = ShuffleNetV2(num_classes = num_classes).cuda()
	elif model == 'mobilenet':
		net = MobileNetV2(num_classes = num_classes).cuda()
	elif model == 'densenet':
		net = DenseNet121(num_classes = num_classes).cuda()
	elif model == 'vit':
		net = VIT(num_classes = num_classes).cuda()
	return net

def adjust_learning_rate(decay_epochs, optimizer, epoch):
	if epoch in decay_epochs:
		for param_group in optimizer.param_groups:
			new_lr = param_group['lr'] * 0.1
			param_group['lr'] = new_lr

def weight_function(t, num_iterations, a = 1):
	weight = (t / num_iterations)**a
	return weight

def select_optimizer(net):
	if args.optimizer == 'ADAM':
		optimizer = optim.Adam(net.parameters(), lr=args.lr)
	else:
		optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
	return optimizer, scheduler

def train(log_name, net_teacher, net_student, dataloader, criterion, optimizer):
	net_student.train()
	net_teacher.eval()
	train_loss = 0.0
	confidence = 0.0
	std = [0.0, 0.0, 0.0]
	total = 0

	for idx, (images, targets) in enumerate(dataloader):
		images, targets = images.cuda(), targets.cuda()

		total += images.size(0)
		p = F.softmax(net_teacher(images) / args.temperature, dim=1).detach()
		u = torch.Tensor(images.size(0), args.num_classes).fill_((1./ args.num_classes)).cuda()
		weights = weight_function(targets, args.T, args.alpha).unsqueeze(dim = 1)
		a = (1 - weights)*u + weights*p

		probabilities = F.softmax(net_student(images), dim=1)
		log_probabilities = torch.log(probabilities)
		loss = criterion(log_probabilities, a)

		softmax_vals= torch.max(probabilities.data, dim=1)[0]
		confidence += softmax_vals.mean().item()
		train_loss += loss.item()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	train_loss = train_loss / (idx + 1)
	confidence = confidence / (idx + 1)
	# stdn = [x / (idx + 1) for x in std]

	return train_loss, confidence, std

def test_acc_loss(dataloader, net):
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	criterion =  nn.CrossEntropyLoss()
	with torch.no_grad():
		for idx, (inputs, targets) in enumerate(dataloader):
			inputs, targets = inputs.cuda(), targets.cuda()
			outputs = net(inputs)
			loss = criterion(outputs, targets)
			test_loss += loss.item()
			_, predicted = torch.max(outputs.data, 1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
	test_loss = test_loss/idx
	test_acc = 100.*correct/total
	# print(f"Test data: {inputs.min().item(), inputs.max().item()}")
	return test_acc, test_loss

def compute_ece(probs, labels, n_bins=15):
	confidences, predictions = torch.max(probs, 1)
	bin_boundaries = np.linspace(0, 1, n_bins + 1)
	bin_lowers = bin_boundaries[:-1]
	bin_uppers = bin_boundaries[1:]
	ece = 0.0
	num_samples = probs.size(0)
	ys = []
	for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
		in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
		bin_size = in_bin.sum().item()
		if bin_size > 0:
			accuracy = (predictions[in_bin] == labels[in_bin]).float().mean()
			avg_confidence = confidences[in_bin].mean()
			ece += bin_size * torch.abs(avg_confidence - accuracy)
			accuracy = accuracy.item()
		else:
			accuracy = 0
		ys.append(accuracy)
	return ece / num_samples, ys
	
def calculate_scores(dataloader, net1, net2 = None):
	net1.eval()
	if net2 != None:
		net2.eval()
	res = np.array([])
	with torch.no_grad():
		for idx, (inputs, targets) in enumerate(dataloader):
			inputs= inputs.cuda()
			probs = torch.softmax(net1(inputs), dim=1)
			if net2 != None:
				probs += torch.softmax(net2(inputs), dim=1)
				probs = probs / 2
			softmax_vals, predicted = torch.max(probs, dim=1)
			res = np.append(res, softmax_vals.cpu().numpy())
	return res

def test(testloader, net1, net2 = None):
	net1.eval()
	if net2 != None:
		net2.eval()
	probs_list = []
	labels_list = []
	with torch.no_grad():
		for inputs, labels in testloader:
			inputs, labels = inputs.cuda(), labels.cuda()
			probs = torch.softmax(net1(inputs), dim=1)
			if net2 != None:
				probs += torch.softmax(net2(inputs), dim=1)
				probs = probs / 2
			probs_list.append(probs)
			labels_list.append(labels)
	probs_all = torch.cat(probs_list, dim=0)
	labels_all = torch.cat(labels_list, dim=0)
	accuracy = accuracy_score(labels_all.cpu().numpy(), torch.argmax(probs_all, 1).cpu().numpy())*100.0
	ece, ys = compute_ece(probs_all, labels_all)
	ece = ece.item()
	return accuracy, ece, ys

def init_setting():

	OOD_list = get_OOD_list()

	# init seed
	if args.seed != 0:
		torch.manual_seed(args.seed)

	# Address
	if not os.path.isdir('results'):
		os.mkdir('results')
	if not os.path.isdir('checkpoints'):
		os.mkdir('checkpoints')

	filename = (args.dataset + '_' + args.teacher + '_' + args.student + '_' + args.generator
	+ '_' + str(args.alpha) + '_' + str(args.T) + '_' + str(args.gap))

	pretrained_pth = ('checkpoints/pretrained_' + args.dataset + '_' + args.teacher + '_'  + str(args.seed) + '.pth')
	auxiliary_pth = ('checkpoints/auxiliary_' + filename + '.pth')
	log_name = ('results/' + filename + '_LOG.csv')
	ood_name = ('results/' + filename + '_OOD.csv')
	table_name = ('results/' + args.dataset + '.csv')
		
	if not os.path.exists(table_name):
		with open(table_name, 'w') as logfile:
			logwriter = csv.writer(logfile, delimiter=',')
			info = ['dataset', 'teacher', 'student',  
			'name', 'epoch', 'optimizer', 'scheduler', 'alpha', 'num_samples', 'T', 'gap', 'temperature', 'generator',
			'output','ACC', 'ECE', 'ys']
			info.extend(OOD_list)
			logwriter.writerow(info)

	with open(log_name, 'w') as logfile:
		logwriter = csv.writer(logfile, delimiter=',')
		logwriter.writerow(['epoch', 'lr', 'train loss', 'confidence', 'test acc', 'test loss', 
			'std_target', 'std_student', 'std_teacher'])

	if not os.path.exists(ood_name):
		with open(ood_name, 'w') as logfile:
			logwriter = csv.writer(logfile, delimiter=',')
			logwriter.writerow(['OOD', 'AUROC', 'AUPR(IN)', 'AUPR(OUT)', 'FPR(95)', 'Detection'])

	return pretrained_pth, auxiliary_pth, log_name, ood_name, table_name

def save_log_result(log_name, epoch, optimizer, train_loss, confidence, test_loss, test_acc, std):
	lr = optimizer.state_dict()['param_groups'][0]['lr']
	with open(log_name, 'a') as logfile:
		logwriter = csv.writer(logfile, delimiter=',')
		logwriter.writerow([epoch, lr,
			train_loss, confidence, test_loss, test_acc, std[0], std[1], std[2]])

def evaluate_performance(ood_name, table_name, dataloader, net_student, net_teacher, ttype):
	detection_results = evaluate_detection(args.dataset, args.precision, net_student, net_teacher)
	accuracy, ece, ys = test(dataloader, net_student, net_teacher)
	save_OOD_result(ood_name, detection_results)
	save_table_result(table_name, 
		[args.dataset, args.teacher, args.student, args.name, args.epoch, args.optimizer, args.scheduler,
		args.alpha, args.num_samples, args.T, args.gap, args.temperature, args.generator, ttype], 
		accuracy, ece, ys, detection_results)

def get_mean_std(dataloader):
	mean = 0.0
	std = 0.0
	total_samples = 0
	for data in dataloader:
		images, _ = data
		batch_samples = images.size(0)
		images = images.view(batch_samples, images.size(1), -1)
		mean += images.mean(2).sum(0)
		std += images.std(2).sum(0)
		total_samples += batch_samples
	mean /= total_samples
	std /= total_samples
	return mean, std

def update_lr(optimizer, scheduler, epoch):
	if args.scheduler == "manual":
		adjust_learning_rate(args.decay_epochs, optimizer, epoch)
	elif args.scheduler == "automatic":
		scheduler.step()

def get_dataloader():
	name = args.dataset + '_' + args.teacher + '_' + args.generator + '_' + str(args.T)
	dataloader = load_synthesized_samples(args.batch_size, name, args.num_samples, args.T, args.gap)
	return dataloader

def main():

	pretrained_pth, auxiliary_pth, log_name, ood_name, table_name = init_setting()
	
	trainloader, testloader = get_ID_dataset(args.dataset, args.batch_size)
	net_teacher = build_teacher(args.teacher, args.num_classes)
	net_teacher.load_state_dict(torch.load(pretrained_pth))
	net_student = build_student(args.student, args.num_classes, net_teacher)
	if args.student == args.teacher:
		net_student.load_state_dict(torch.load(pretrained_pth))
	optimizer, scheduler = select_optimizer(net_student)
	criterion = nn.KLDivLoss(reduction='batchmean')

	dataloader = get_dataloader()

	if args.train:
		print("Training: " + auxiliary_pth)
		for epoch in range(args.epoch):
			train_loss, confidence, std = train(log_name, net_teacher, net_student, dataloader, criterion, optimizer)
			update_lr(optimizer, scheduler, epoch)
			test_acc, test_loss = test_acc_loss(testloader, net_student)
			save_log_result(log_name, epoch, optimizer, train_loss, confidence, test_acc, test_loss, std)
			if (epoch + 1) % 50 == 0:
				print("Epoch: ", epoch, 
					"LR: ", round(optimizer.state_dict()['param_groups'][0]['lr'], 6),
					"Conf: ", round(confidence, 4),
					"Train Loss: ", round(train_loss, 4), 
					"Test Acc: ", test_acc,
					"Std: ", std)
		torch.save(net_student.state_dict(), auxiliary_pth)

	if args.test:
		net_student.load_state_dict(torch.load(auxiliary_pth))
		print("Testing: " + auxiliary_pth + " (Single)")
		evaluate_performance(ood_name, table_name, testloader, net_student, None, "Single")

	print("######################################################################################################")
if __name__ == '__main__':
    main()
