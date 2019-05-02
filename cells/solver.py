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
	blur = cv2.GaussianBlur(arr[:][:],(9,9),0)
	img = np.reshape(blur, (65536,1))
	labels[i][:] = img
	im = Image.open(s2) # Can be many different formats.
	pix = im.load()
	arr = np.asarray(im)
	#blur = cv2.GaussianBlur(arr[:][:],(9,9),0)
	img = np.reshape(blur, (65536,1))
	data[i][:] = img			
class imagedataset(Dataset):
	def __init__(self):
		data = scipy.io.loadmat("Indian_pines_corrected.mat")
		labels = scipy.io.loadmat("Indian_pines_gt.mat")
		complete_data =  np.array(data['indian_pines_corrected'])
		complete_labels = np.array(labels['indian_pines_gt'])
		flat_data = np.reshape(complete_data, (21025,200))
		flat_labels = np.reshape(complete_labels,(21025,1))
		train_data = flat_data[:10512][:]
		train_labels = flat_labels[:10512][:]
		train_data = train_data.astype(np.float32)
		train_labels = train_labels.astype(np.long)
		self.len = train_data.shape[0]
		self.x_data = torch.from_numpy(train_data)
		self.y_data = torch.from_numpy(train_labels)
	def __getitem__(self,index):
		return self.x_data[index], self.y_data[index]
	def __len__(self):
		return self.len
class imagetestset(Dataset):
	def __init__(self):
		data = scipy.io.loadmat("Indian_pines_corrected.mat")
		labels = scipy.io.loadmat("Indian_pines_gt.mat")
		complete_data =  np.array(data['indian_pines_corrected'])
		complete_labels = np.array(labels['indian_pines_gt'])
		flat_data = np.reshape(complete_data, (21025,200))
		flat_labels = np.reshape(complete_labels,(21025,1))
		test_data = flat_data[10512:][:]
		test_labels = flat_labels[10512:][:]
		test_data = test_data.astype(np.float32)
		test_labels = test_labels.astype(np.long)
		self.len = test_data.shape[0]
		self.x_data = torch.from_numpy(test_data)
		self.y_data = torch.from_numpy(test_labels)
	def __getitem__(self,index):
		return self.x_data[index], self.y_data[index]
	def __len__(self):
		return self.len				
class imagenet(nn.Module):
	def __init__(self,input_size,hidden1_size,hidden2_size,num_classes):
		super(imagenet, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden1_size)
		self.fc2 = nn.Linear(hidden1_size, hidden2_size)
		self.fc3 = nn.Linear(hidden2_size,num_classes)
		#self.softmax = nn.Softmax()

	def forward(self,x):
		out = self.fc1(x)
		out = self.fc2(out)
		out = self.fc3(out)
		#out = self.softmax(out)
		return out

model = imagenet(200, 200, 100, 17)
batch_size = 64
train_data = imagedataset()
test_data = imagetestset()
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.00002, momentum=0.5)

def train(epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		#print(data.shape)
		output = model(data)
		#print(output.shape)
		#print(target.shape)
		target = target.squeeze()
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % 10 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx*len(data), len(train_loader.dataset), 100.* batch_idx / len(train_loader), loss.item()))

def test():
	model.eval()
	test_loss = 0
	correct = 0
	for data, target in test_loader:
		data, target = Variable(data), Variable(target)
		target = target.squeeze()
		output = model(data)
		test_loss += criterion(output, target).data
		pred = torch.max(output.data,1)[1]
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()
	test_loss /= len(test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss,correct,len(test_loader.dataset),100.*correct/len(test_loader.dataset)))	

for epoch in range(1,10):	
	train(epoch)
	test()