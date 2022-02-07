import numpy as np
import torch
from torch.utils.data import DataLoader

class DataSet():
	def __init__(self,args):
		self.path = args.data_path
		self.dim = [args.x,args.y,args.z]
		self.c = [args.crop_x,args.crop_y,args.crop_z]
		self.total_samples = args.total_samples
		self.training_samples = args.total_samples*4//10
		self.interval = args.interval
		self.data = np.zeros((self.training_samples,1,self.dim[0],self.dim[1],self.dim[2]))
		self.croptimes = args.croptimes
		if (self.dim[0] == self.c[0]) and (self.dim[1] == self.c[1]) and (self.dim[2] == self.c[2]):
			self.croptimes = 1

	def ReadData(self):
		for i in range(1,self.training_samples+1):
			v = np.fromfile(args.data_path+'{:04d}'.format(i)+'.dat',dtype='<f')
			v = v.reshape(self.dim[2],self.dim[1],self.dim[0]).transpose()
			v = 2*(v-v.min())/(v.max()-v.min())-1
			self.data[i-1] = v

	def GetTrainingData(self):
		group = self.training_samples-self.interval-2
		s = np.zeros((self.croptimes*group,1,1,self.c[0],self.c[1],self.c[2]))
		i = np.zeros((self.croptimes*group,self.interval,1,self.c[0],self.c[1],self.c[2]))
		e = np.zeros((self.croptimes*group,1,1,self.c[0],self.c[1],self.c[2]))
		idx = 0
		for k in range(0,group):
			sc,ic,ec = self.CropData(self.data[k:k+self.interval+2])
			for j in range(0,self.croptimes):
				s[idx] = sc[j]
				i[idx] = ic[j]
				e[idx] = ec[j]
				idx += 1
		s = torch.FloatTensor(s)
		i = torch.FloatTensor(i)
		e = torch.FloatTensor(e)
		data = torch.utils.data.TensorDataset(s,i,e)
		train_loader = DataLoader(dataset=data, batch_size=1, shuffle=True)
		return train_loader

	def InferenceData(self):
		for i in range(1,self.total_samples,self.interval+1):
			v = np.fromfile(args.data_path+'{:04d}'.format(i)+'.dat',dtype='<f')
			v = v.reshape(self.dim[2],self.dim[1],self.dim[0]).transpose()
			v = 2*(v-v.min())/(v.max()-v.min())-1
			self.data[i-1] = v

	def CropData(self,data):
		s = []
		i = []
		e = []
		n = 0
		while n<self.croptimes:
			if self.c[0]==self.dim[0]:
				x = 0
			else:
				x = np.random.randint(0,self.dim[0]-self.c[0])
			if self.c[1] == self.dim[1]:
				y = 0
			else:
				y = np.random.randint(0,self.dim[1]-self.c[1])
			if self.c[2] == self.dim[2]:
				z = 0
			else:
				z = np.random.randint(0,self.dim[2]-self.c[2])
			sc = data[0:1,0:1,x:x+self.c[0],y:y+self.c[1],z:z+self.c[2]]
			ic = data[1:1+self.interval,0:1,x:x+self.c[0],y:y+self.c[1],z:z+self.c[2]]
			ec = data[1+self.interval:,0:1,x:x+self.c[0],y:y+self.c[1],z:z+self.c[2]]
			s.append(sc)
			i.append(ic)
			e.append(ec)
			n = n+1
		return s,i,e


