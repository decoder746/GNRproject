import scipy.io
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import kde
import cv2
import torch.nn.functional as F
import random
import operator
from itertools import tee
labels = np.zeros(shape=(200,256,256))
data = np.zeros(shape=(200,256,256))
for i in range(1,201):
	if (i < 10):
		s = "00" + str(i)
	elif (i < 100):
		s = "0" + str(i)
	else:
		s = str(i)
	s1 = s + "dots.png"
	s2 = s + "cell.png"	
	im = Image.open(s1) # Can be many different formats.
	pix = im.load()
	arr = np.asarray(im)
	blur = cv2.GaussianBlur(arr[:][:][:],(3,3),0)
	#print(np.shape(blur))
	#img = np.reshape(blur[:,:,0], (65536))
	labels[i-1][:][:] = blur[:,:,0]
	im = Image.open(s2) # Can be many different formats.
	pix = im.load()
	arr = np.asarray(im)
	#blur = cv2.GaussianBlur(arr[:][:],(9,9),0)
	#img = np.reshape(arr[:,:,2], (65536))
	data[i-1][:][:] = arr[:,:,2]

class imagedataset(Dataset):
	def __init__(self):
		train_data = data[0:100][:][:]
		train_labels = labels[0:100][:][:]
		train_data = train_data.astype(np.float32)
		train_labels = train_labels.astype(np.float32)
		self.len = train_data.shape[0]
		self.x_data = torch.from_numpy(train_data)
		self.y_data = torch.from_numpy(train_labels)
	def __getitem__(self,index):
		x = random.randint(1,200)
		y = random.randint(1,200)
		x1 = x+48
		y1 = y+48
		x_reduced = self.x_data[index,x:x1,y:y1]
		y_reduced = self.y_data[index,x:x1,y:y1]
		y_double_reduced = torch.randn(8,8,dtype=torch.float32)
		for i in range(0,8):
			for j in range(0,8):
				temp_arr = y_reduced[i:i+6][j:j+6]
				y_double_reduced[i][j] = torch.sum(temp_arr)
	
		return x_reduced.reshape(1,48,48), y_double_reduced.reshape(1,8,8)
	def __len__(self):
		return self.len
class imagetestset(Dataset):
	def __init__(self):
		train_data = data[100:200][:][:]
		train_labels = labels[100:200][:][:]
		train_data = train_data.astype(np.float32)
		train_labels = train_labels.astype(np.float32)
		self.len = train_data.shape[0]
		#print("Len",self.len)
		self.x_data = torch.from_numpy(train_data)
		#print("x",self.x_data.size())
		self.y_data = torch.from_numpy(train_labels)
		#print("y",self.y_data.size())
	def __getitem__(self,index):
		x = random.randint(1,200)
		y = random.randint(1,200)
		x_reduced = self.x_data[index,x:x+48,y:y+48]
		y_reduced = self.y_data[index,x:x+48,y:y+48]
		#print("xr",x_reduced.size())
		#print("yr", y_reduced.size())
		y_double_reduced = torch.randn(8,8,dtype=torch.float32)
		for i in range(0,8):
			for j in range(0,8):
				temp_arr = y_reduced[i:i+6][j:j+6]
				y_double_reduced[i][j] = torch.sum(temp_arr)
		return x_reduced.reshape(1,48,48), y_double_reduced.reshape(1,8,8)
	def __len__(self):
		return self.len				

class imagenet(nn.Module):
	def __init__(self):
		super(imagenet, self).__init__()
		self.conv1 = nn.Conv2d(1,16,kernel_size=3)
		torch.nn.init.xavier_uniform_(self.conv1.weight)
		self.mp = nn.MaxPool2d(2)
		self.drop = nn.Dropout2d(0.5)
		self.conv2 = nn.Conv2d(16,32,kernel_size=3)
		torch.nn.init.xavier_uniform_(self.conv2.weight)
		self.fc1 = nn.Linear(3200, 100)
		self.fc2 = nn.Linear(100, 64)
		#self.fc3 = nn.Linear(hidden2_size,num_classes)
		#self.softmax = nn.Softmax()

	def forward(self,x):
		in_size = x.size(0)

		x = self.drop(self.mp(self.conv1(x)))
		x = F.relu(self.drop(self.mp(self.conv2(x))))
		x = x.view(in_size,-1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		#out = self.softmax(out)
		return x.reshape(-1,1,8,8)

model1 = imagenet()
model2 = imagenet()
model3 = imagenet()
model4 = imagenet()
model5 = imagenet()
model6 = imagenet()
batch_size = 100
train_data = imagedataset()
test_data = imagetestset()
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

criterion = nn.MSELoss()
optimizer1 = torch.optim.RMSprop(model1.parameters(), lr=0.000001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
optimizer1.zero_grad()
optimizer2 = torch.optim.RMSprop(model2.parameters(), lr=0.000001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
optimizer2.zero_grad()
optimizer3 = torch.optim.RMSprop(model3.parameters(), lr=0.000001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
optimizer3.zero_grad()
optimizer4 = torch.optim.RMSprop(model4.parameters(), lr=0.000001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
optimizer4.zero_grad()
optimizer5 = torch.optim.RMSprop(model5.parameters(), lr=0.000001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
optimizer5.zero_grad()
optimizer6 = torch.optim.RMSprop(model6.parameters(), lr=0.000001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
optimizer6.zero_grad()

def train1(epoch):
	model1.train()
	for i in range(1,1600):
		alpha = enumerate(train_loader)
		alpha, p = tee(alpha)
		dic = {}
		for batch_idx, (data, target) in alpha:
			data, target = Variable(data), Variable(target)
			target = target.squeeze()
			output = model1(data)
			#print(data.shape)
			#print(output[1,0,:,:].data)
			#print(target[1,:,:])
			#out = torch.sum(output[1,:,:,:])/256
			#inp = torch.sum(target[1,:,:])/256
			#print(inp,out)
			test_loss = criterion(output, target).data
			dic[tuple((data,target))] = test_loss
		#print(dic)	
		sorted_data = sorted(dic.items(), key=operator.itemgetter(1))[0][0][0][31:97]
		sorted_target = sorted(dic.items(), key=operator.itemgetter(1))[0][0][1][31:97]		
		data = sorted_data
		target = sorted_target
		optimizer1.zero_grad()
		#print(sorted_data.shape)
		output = model1(data)
		#print(output.shape)
		#print(target.shape)
		target = target.squeeze()
		loss = criterion(output, target)
		loss.backward()
		optimizer1.step()
		if batch_idx==0:
			print('Train1 Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx*len(data), len(train_loader.dataset), 100.* batch_idx / len(train_loader), loss.data))	
			
		#print(data.shape)
			

def train2(epoch):
	model2.train()
	
	for i in range(1,1600):
		alpha = enumerate(train_loader)
		alpha, p = tee(alpha)
		dic = {}
		for batch_idx, (data, target) in alpha:
			data, target = Variable(data), Variable(target)
			target = target - model1(data)
			target = target.squeeze()
			output = model2(data)
			#print(data.shape)
			#print(output[1,0,:,:].data)
			#print(target[1,:,:])
			#out = torch.sum(output[1,:,:,:])/256
			#inp = torch.sum(target[1,:,:])/256
			#print(inp,out)
			test_loss = criterion(output, target).data
			dic[tuple((data,target))] = test_loss
		#print(dic)	
		sorted_data = sorted(dic.items(), key=operator.itemgetter(1))[0][0][0][31:97]
		sorted_target = sorted(dic.items(), key=operator.itemgetter(1))[0][0][1][31:97]		
		data = sorted_data
		target = sorted_target
		optimizer2.zero_grad()
		#print(sorted_data.shape)
		output = model2(data)
		#print(output.shape)
		#print(target.shape)
		target = target.squeeze()
		loss = criterion(output, target)
		loss.backward()
		optimizer2.step()
		if batch_idx==0:
			print('Train2 Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx*len(data), len(train_loader.dataset), 100.* batch_idx / len(train_loader), loss.data))	


def train3(epoch):
	model3.train()
	
	for i in range(1,1600):
		alpha = enumerate(train_loader)
		alpha, p = tee(alpha)
		dic = {}
		for batch_idx, (data, target) in alpha:
			data, target = Variable(data), Variable(target)
			target = target - model1(data) - model2(data)
			target = target.squeeze()
			output = model3(data)
			#print(data.shape)
			#print(output[1,0,:,:].data)
			#print(target[1,:,:])
			#out = torch.sum(output[1,:,:,:])/256
			#inp = torch.sum(target[1,:,:])/256
			#print(inp,out)
			test_loss = criterion(output, target).data
			dic[tuple((data,target))] = test_loss
		#print(dic)	
		sorted_data = sorted(dic.items(), key=operator.itemgetter(1))[0][0][0][31:97]
		sorted_target = sorted(dic.items(), key=operator.itemgetter(1))[0][0][1][31:97]		
		data = sorted_data
		target = sorted_target
		optimizer3.zero_grad()
		#print(sorted_data.shape)
		output = model3(data)
		#print(output.shape)
		#print(target.shape)
		target = target.squeeze()
		loss = criterion(output, target)
		loss.backward()
		optimizer3.step()
		if batch_idx==0:
			print('Train3 Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx*len(data), len(train_loader.dataset), 100.* batch_idx / len(train_loader), loss.data))	

def train4(epoch):
	model4.train()
	
	for i in range(1,1600):
		alpha = enumerate(train_loader)
		alpha, p = tee(alpha)
		dic = {}
		for batch_idx, (data, target) in alpha:
			data, target = Variable(data), Variable(target)
			target = target - model1(data) - model2(data) - model3(data)
			target = target.squeeze()
			output = model4(data)
			#print(data.shape)
			#print(output[1,0,:,:].data)
			#print(target[1,:,:])
			#out = torch.sum(output[1,:,:,:])/256
			#inp = torch.sum(target[1,:,:])/256
			#print(inp,out)
			test_loss = criterion(output, target).data
			dic[tuple((data,target))] = test_loss
		#print(dic)	
		sorted_data = sorted(dic.items(), key=operator.itemgetter(1))[0][0][0][31:97]
		sorted_target = sorted(dic.items(), key=operator.itemgetter(1))[0][0][1][31:97]		
		data = sorted_data
		target = sorted_target
		optimizer4.zero_grad()
		#print(sorted_data.shape)
		output = model4(data)
		#print(output.shape)
		#print(target.shape)
		target = target.squeeze()
		loss = criterion(output, target)
		loss.backward()
		optimizer4.step()
		if batch_idx==0:
			print('Train4 Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx*len(data), len(train_loader.dataset), 100.* batch_idx / len(train_loader), loss.data))	

def train5(epoch):
	model5.train()
	
	for i in range(1,1600):
		alpha = enumerate(train_loader)
		alpha, p = tee(alpha)
		dic = {}
		for batch_idx, (data, target) in alpha:
			data, target = Variable(data), Variable(target)
			target = target - model1(data) - model2(data) - model3(data) - model4(data)
			target = target.squeeze()
			output = model5(data)
			#print(data.shape)
			#print(output[1,0,:,:].data)
			#print(target[1,:,:])
			#out = torch.sum(output[1,:,:,:])/256
			#inp = torch.sum(target[1,:,:])/256
			#print(inp,out)
			test_loss = criterion(output, target).data
			dic[tuple((data,target))] = test_loss
		#print(dic)	
		sorted_data = sorted(dic.items(), key=operator.itemgetter(1))[0][0][0][31:97]
		sorted_target = sorted(dic.items(), key=operator.itemgetter(1))[0][0][1][31:97]		
		data = sorted_data
		target = sorted_target
		optimizer5.zero_grad()
		#print(sorted_data.shape)
		output = model5(data)
		#print(output.shape)
		#print(target.shape)
		target = target.squeeze()
		loss = criterion(output, target)
		loss.backward()
		optimizer5.step()
		if batch_idx==0:
			print('Train5 Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx*len(data), len(train_loader.dataset), 100.* batch_idx / len(train_loader), loss.data))

def train6(epoch):
	model6.train()
	
	for i in range(1,1600):
		alpha = enumerate(train_loader)
		alpha, p = tee(alpha)
		dic = {}
		for batch_idx, (data, target) in alpha:
			data, target = Variable(data), Variable(target)
			target = target - model1(data) - model2(data) - model3(data) - model4(data) - model5(data)
			target = target.squeeze()
			output = model6(data)
			#print(data.shape)
			#print(output[1,0,:,:].data)
			#print(target[1,:,:])
			#out = torch.sum(output[1,:,:,:])/256
			#inp = torch.sum(target[1,:,:])/256
			#print(inp,out)
			test_loss = criterion(output, target).data
			dic[tuple((data,target))] = test_loss
		#print(dic)	
		sorted_data = sorted(dic.items(), key=operator.itemgetter(1))[0][0][0][31:97]
		sorted_target = sorted(dic.items(), key=operator.itemgetter(1))[0][0][1][31:97]		
		data = sorted_data
		target = sorted_target
		optimizer6.zero_grad()
		#print(sorted_data.shape)
		output = model6(data)
		#print(output.shape)
		#print(target.shape)
		target = target.squeeze()
		loss = criterion(output, target)
		loss.backward()
		optimizer6.step()
		if batch_idx==0:
			print('Train6 Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx*len(data), len(train_loader.dataset), 100.* batch_idx / len(train_loader), loss.data))

def test():
	model3.eval()
	test_loss = 0
	test_loss1 = 0
	test_loss2 = 0
	test_loss3 = 0
	test_loss4 = 0
	test_loss5 = 0
	test_loss6 = 0
	correct = 0
	for data, target in test_loader:
		data, target = Variable(data), Variable(target)
		target = target.squeeze()
		output1 = model1(data)
		output2 = model1(data) + model2(data)
		output3 = model3(data) + model2(data) + model1(data)
		output4 = model3(data) + model2(data) + model1(data) + model4(data)
		output5 = model3(data) + model2(data) + model1(data) + model4(data) + model5(data)
		output6 = model3(data) + model2(data) + model1(data) + model4(data) + model5(data) + model6(data)
		output = model3(data) + model2(data) + model1(data) + model4(data) + model5(data) + model6(data)
		#print(output[1,0,:,:].data)
		#print(target[1,:,:])
		out1 = torch.sum(output1[1,:,:,:])/256
		out2 = torch.sum(output2[1,:,:,:])/256
		out3 = torch.sum(output3[1,:,:,:])/256
		out4 = torch.sum(output4[1,:,:,:])/256
		out5 = torch.sum(output5[1,:,:,:])/256
		out6 = torch.sum(output6[1,:,:,:])/256
		inp = torch.sum(target[1,:,:])/256
		test_loss1 += abs(out1 - inp)
		test_loss2 += abs(out2 - inp)
		test_loss3 += abs(out3 - inp)
		test_loss4 += abs(out4 - inp)
		test_loss5 += abs(out5 - inp)
		test_loss6 += abs(out6 - inp)
		test_loss += criterion(output, target).data
		#pred = torch.max(output.data,1)[1]
		#correct += pred.eq(target.data.view_as(pred)).cpu().sum()
	test_loss1 /= len(test_loader.dataset)
	print('\nTest set: Average loss without boosting: {:.4f}\n'.format(test_loss1))
	print('\nTest set: Average loss with 1 boost: {:.4f}\n'.format(test_loss2))
	print('\nTest set: Average loss with 2 boost: {:.4f}\n'.format(test_loss3))
	print('\nTest set: Average loss with 3 boost: {:.4f}\n'.format(test_loss4))
	print('\nTest set: Average loss with 4 boost: {:.4f}\n'.format(test_loss5))
	print('\nTest set: Average loss with 5 boost: {:.4f}\n'.format(test_loss6))
	print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))


for epoch in range(1,2):
	train1(epoch)
	train2(epoch)
	train3(epoch)
	train4(epoch)
	train5(epoch)
	train6(epoch)
	test()	