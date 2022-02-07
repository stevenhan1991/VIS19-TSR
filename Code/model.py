import torch.nn as nn
from torch.nn import init
from torch.nn.modules import conv, Linear
import torch.nn.functional as F
import torch
from collections import OrderedDict 
import math
import torch.nn.utils.spectral_norm as spectral_norm

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv")!=-1:
        init.kaiming_uniform_(m.weight.data)
    elif classname.find("Linear")!=-1:
        init.kaiming_uniform_(m.weight.data)
    elif classname.find("BatchNorm")!=-1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

class Block(nn.Module):
	def __init__(self,inchannels,outchannels,dropout=False,kernel=3,bias=False,depth=2,mode='down'):
		super(Block,self).__init__()
		layers = []
		for i in range(int(depth)):
			layers += [spectral_norm(nn.Conv3d(inchannels,inchannels,kernel_size=kernel,padding=kernel//2,bias=bias)),
			           nn.ReLU(inplace=True)]
			if dropout:
				layers += [nn.Dropout(0.5)]
		self.model = nn.Sequential(*layers)
		if mode == 'down':
			self.conv1 = spectral_norm(nn.Conv3d(inchannels,outchannels,4,2,1))
			self.conv2 = spectral_norm(nn.Conv3d(inchannels,outchannels,4,2,1))
		elif mode == 'up':
			self.conv1 = VoxelShuffle(inchannels,outchannels,2)
			self.conv2 = VoxelShuffle(inchannels,outchannels,2)

	def forward(self,x):
		y = self.model(x)
		y = self.conv1(y)
		x = self.conv2(x)
		return y+x

def voxel_shuffle(input, upscale_factor):
    batch_size, channels, in_height, in_width, in_depth = input.size()
    channels //= upscale_factor ** 3

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor
    out_depth = in_depth * upscale_factor

    input_view = input.reshape(batch_size, channels, upscale_factor, upscale_factor, upscale_factor, in_height, in_width, in_depth)

    return input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(batch_size, channels, out_height, out_width, out_depth)

class VoxelShuffle(nn.Module):
	def __init__(self,inchannels,outchannels,upscale_factor):
		super(VoxelShuffle,self).__init__()
		self.upscale_factor = upscale_factor
		self.conv = nn.Conv3d(inchannels,outchannels*(upscale_factor**3),3,1,1)

	def forward(self,x):
		x = voxel_shuffle(self.conv(x),self.upscale_factor)
		return x


class LSTMCell(nn.Module):
	def __init__(self,input_size,hidden_size,kernel):
		super(LSTMCell,self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		pad = kernel//2
		self.Gates = nn.Conv3d(input_size+hidden_size,4*hidden_size,kernel,padding=pad)

	def forward(self,input_,prev_hidden=None,prev_cell=None):
		batch_size = input_.data.size()[0]
		spatial_size = input_.data.size()[2:]
		if prev_hidden is None and prev_cell is None:
			state_size = [batch_size,self.hidden_size]+list(spatial_size)
			prev_hidden = torch.zeros(state_size)
			prev_cell = torch.zeros(state_size)
		prev_hidden = prev_hidden.cuda()
		prev_cell = prev_cell.cuda()
		stacked_inputs = torch.cat((input_,prev_hidden),1)
		gates = self.Gates(stacked_inputs)

		in_gate,remember_gate,out_gate,cell_gate = gates.chunk(4,1)
		in_gate = torch.sigmoid(in_gate)
		remember_gate = torch.sigmoid(remember_gate)
		out_gate = torch.sigmoid(out_gate)
		cell_gate = torch.tanh(cell_gate)

		cell = (remember_gate*prev_cell)+(in_gate*cell_gate)
		hidden = out_gate*torch.tanh(cell)
		return hidden,cell

class Encoder(nn.Module):
	def __init__(self,inc,init_channels):
		super(Encoder,self).__init__()
		self.conv1 = Block(inc,init_channels) 
		self.conv2 = Block(init_channels,2*init_channels) 
		self.conv3 = Block(2*init_channels,4*init_channels)
		self.conv4 = Block(4*init_channels,8*init_channels) 

	def forward(self,x):
		x1 = F.relu(self.conv1(x)) 
		x2 = F.relu(self.conv2(x1))
		x3 = F.relu(self.conv3(x2))
		x4 = F.relu(self.conv4(x3))
		return [x1,x2,x3,x4]

class Decoder(nn.Module):
	def __init__(self,outc,init_channels):
		super(Decoder,self).__init__()
		self.deconv41 = Block(init_channels,init_channels//2,mode='up') 
		self.conv_u41 = nn.Conv3d(init_channels,init_channels//2,3,1,1)
		self.deconv31 = Block(init_channels//2,init_channels//4,mode='up') 
		self.conv_u31 = nn.Conv3d(init_channels//2,init_channels//4,3,1,1)
		self.deconv21 = Block(init_channels//4,init_channels//8,mode='up')
		self.conv_u21 = nn.Conv3d(init_channels//4,init_channels//8,3,1,1)
		self.deconv11 = Block(init_channels//8,init_channels//16,mode='up')
		self.conv_u11 = nn.Conv3d(init_channels//16,outc,3,1,1)
		self.ac = ac

	def forward(self,features):
		u11 = F.relu(self.deconv41(features[-1]))
		u11 = F.relu(self.conv_u41(torch.cat((features[-2],u11),dim=1)))
		u21 = F.relu(self.deconv31(u11))
		u21 = F.relu(self.conv_u31(torch.cat((features[-3],u21),dim=1)))
		u31 = F.relu(self.deconv21(u21)) 
		u31 = F.relu(self.conv_u21(torch.cat((features[-4],u31),dim=1)))
		u41 = F.relu(self.deconv11(u31))
		out = self.conv_u11(u41)
		return out

class TSR(nn.Module):
	def __init__(self,inc,outc,init_channels,interval):
		super(TSR,self).__init__()
		self.encoder = Encoder(inc,init_channels)
		self.decoder = Decoder(outc,8*init_channels)
		self.interval = interval
		self.lstm = LSTMCell(init_channels*8,init_channels*8,3)

	def forward(self,x):
		x = self.encoder(x)
		h = None
		c = None
		comps = []
		for i in range(self.interval):
			h,c = self.lstm(x[-1],h,c)
			comp = self.decoder([x[0],x[1],x[2],h])
			comps.append(comp)
		comps = torch.stack(comps)
		comps = comps.permute(1,2,0,3,4,5)
		comps = torch.squeeze(comps,0)
		return comps

class Dis(nn.Module):
    def __init__(self):
        super(Dis,self).__init__(args)
        self.conv1 = spectral_norm(nn.Conv3d(interval,64,4,2,1))
        self.conv2 = spectral_norm(nn.Conv3d(64,128,4,2,1))
        self.conv3 = spectral_norm(nn.Conv3d(128,256,4,2,1))
        self.conv4 = spectral_norm(nn.Conv3d(256,512,4,2,1))
        self.conv5 = spectral_norm(nn.Conv3d(512,1,[args.crop_x//16,args.crop_y//16,args.crop_z//16]))
        self.ac = nn.LeakyReLU(0.2,inplace=True)

    def forward(self,x):
    	features = []
        x1 = self.ac(self.conv1(x))
        features.append(x1)
        x2 = self.ac(self.conv2(x1))
        features.append(x2)
        x3 = self.ac(self.conv3(x2))
        features.append(x3)
        x4 = self.ac(self.conv4(x3))
        features.append(x4)
        x5 = self.conv5(x4)
        features.append(x5.view(-1))
        return features






