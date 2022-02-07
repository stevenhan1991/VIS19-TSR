import torch.nn as nn
import torch.optim as optim
import time
import argparse
import DataPre
import torch
import numpy as np
from model import *

def trainGAN(G,D,args,dataset):
    device = torch.device("cuda:0" if args.cuda else "cpu")
    optimizer_G = optim.Adam(G.parameters(), lr=args.lr_G,betas=(0.0,0.999)) #betas=(0.5,0.999)
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr_D,betas=(0.0,0.999))
    ganloss = nn.MSELoss()
    critic = 1
    for itera in range(1,args.epochs+1):
        train_loader = dataset.GetTrainingData()
        loss_G = 0
        loss_D = 0
        mse_loss = 0
        print("========================")
        for batch_idx,(s,i,e) in enumerate(train_loader):
        	if args.cuda:
        		starting = s.cuda()
        		ending = e.cuda()
        		intermerdiate = ie.cuda()

        	batch = i.size(0)

        	for p in G.parameters():
        		p.requires_grad = False
        	for j in range(1,args.critic+1):
        		optimizer_D.zero_grad()
        		label_real = torch.Variable(torch.full((batch,),1.0,device=device))
        		output_real = D(intermerdiate)
        		real_loss = ganloss(output_real[-1],label_real)

        		fake_data = G(torch.cat((starting,ending),dim=1))
        		label_fake = torch.Variable(torch.full((batch,),0.0,device=device))
        		output_fake = D(fake_data)
        		fake_loss = ganloss(output_fake[-1],label_fake)
        		loss = 0.5*(real_loss+fake_loss)

        		loss.backward()
        		loss_D += loss.mean().item()
        		optimizer_D.step()

        	for p in G.parameters():
        		p.requires_grad = True
        	for p in D.parameters():
        		p.requires_grad = False


    		optimizer_G.zero_grad()
    		label_real = torch.Variable(torch.full((batch,),1.0,device=device))
    		fake_data = G(torch.cat((starting,ending),dim=1))
    		output_real = D(fake_data)
    		output_features = D(intermerdiate)
    		L_adv = ganloss(output_real,label_real[-1])
    		L_c = ganloss(fake_data,intermerdiate)
    		L_f = ganloss(output_real[0],output_features[0])+ganloss(output_real[1],output_features[1])+ganloss(output_real[2],output_features[2])+ganloss(output_real[3],output_features[3])
    		error = args.adversarial*L_adv+args.l2*L_c+args.percep*L_f
    		error.backward()
    		loss_G += error.item()
    		optimizer_G.step()
        	for p in D.parameters():
        		p.requires_grad = True
        y = time.time()
        if itera%args.chechpoint == 0:
        	torch.save(G,args.model_path+str(itera)+'.pth')

def inference(model,scalar,args):
	for i in range(0,len(scalar.data),args.interval+1):
		s = np.zeros((1,1,scalar.dim[0],scalar.dim[1],scalar.dim[2]))
		e = np.zeros((1,1,scalar.dim[0],scalar.dim[1],scalar.dim[2]))
		if (i+args.interval+1)<len(scalar.data):
			s[0] = scalar.data[i]
			e[0] = scalar.data[i+args.interval+1]
			s = torch.FloatTensor(s).cuda()
			e = torch.FloatTensor(e).cuda()
			with torch.no_grad():
				intermerdiate = model(torch.cat((s,e),dim=1))
				intermerdiate = intermerdiate.cpu().detach().numpy()
			for j in range(1,args.interval+1):
				data = intermerdiate[j-1]
				data = np.asarray(data,dtype='<f')
				data = data.flatten('F')
				data.tofile(args.result_path+'{:04d}'.format(i+j+1)+'.dat',format='<f')

def concatsubvolume(model,data,win_size,args):
	x,y,z = data[0].size()[2],data[0].size()[3],data[0].size()[4]
	w = np.zeros((win_size[0],win_size[1],win_size[2]))
	for i in range(win_size[0]):
		for j in range(win_size[1]):
			for k in range(win_size[2]):
				dx = min(i,win_size[0]-1-i)
				dy = min(j,win_size[1]-1-j)
				dz = min(k,win_size[2]-1-k)
				d = min(min(dx,dy),dz)+1
				w[i,j,k] = d
	w = w/np.max(w)
	avI = np.zeros((x,y,z))
	pmap= np.zeros((args.interval,1,x,y,z))
	avk = 4
	for i in range((avk*x-win_size[0])//win_size[0]+1):
		for j in range((avk*y-win_size[1])//win_size[1]+1):
			for k in range((avk*z-win_size[2])//win_size[2]+1):
				si = (i*win_size[0]//avk)
				ei = si+win_size[0]
				sj = (j*win_size[1]//avk)
				ej = sj+win_size[1]
				sk = (k*win_size[2]//avk)
				ek = sk+win_size[2]
				if ei>x:
					ei= x
					si=ei-win_size[0]
				if ej>y:
					ej = y
					sj = ej-win_size[1]
				if ek>z:
					ek = z
					sk = ek-win_size[2]
				d0 = data[0][:,:,si:ei,sj:ej,sk:ek]
				d1 = data[1][:,:,si:ei,sj:ej,sk:ek]
				with torch.no_grad():
					intermerdiate = model(torch.cat((d0,d1),dim=1))
				k = np.multiply(intermerdiate.cpu().detach().numpy(),w)
				avI[si:ei,sj:ej,sk:ek] += w
				pmap[:,:,si:ei,sj:ej,sk:ek] += k
	result = np.divide(pmap,avI)
	return result

