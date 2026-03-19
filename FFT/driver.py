import argparse
import numpy as np
import torch
from read_dataset import data_from_name
from model import *
import os
import matplotlib.pyplot as plt
import copy
import pandas as pd
#==============================================================================
# Training settingss
#==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default='4',  help='seed value')
#
parser.add_argument('--dataset', type=str, default='pendulum', metavar='N', help='dataset name')
#
parser.add_argument('--noise', type=float, default=0.0, help='noise of data')
#
parser.add_argument('--M', type=int, default=800, help='size of trainning')
#
parser.add_argument('--L', type=int, default=96, help='size of prediction')
#
parser.add_argument('--M_S', type=int, default=32, help='size of segments')
#
parser.add_argument('--batch', type=int, default=8, metavar='N', help='batch size')
#
parser.add_argument('--alpha', type=float, default=0.2, metavar='N', help='batch size')
#
parser.add_argument('--lr', type=float, default=1e-2, metavar='N', help='learning rate')
#
parser.add_argument('--lr_decay', type=float, default='0.5', help='PCL penalty lambda hyperparameter')
#
parser.add_argument('--lr_update', type=int, nargs='+', default=[50,100,150,200,250], help='decrease learning rate')
#
parser.add_argument('--wd', type=float, default=1e-5, metavar='N', help='weight_decay L2 regulization')
#
parser.add_argument('--gradclip', type=float, default=1e-7, help='gradient clipping')
#
parser.add_argument('--epochs', type=int, default=300, metavar='N', help='number of epochs to train')
#
args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def FourierFilter(x,d):
	if d == 2:
		xf = torch.fft.fft(x, dim=0)
		mask = torch.ones_like(xf)
		mask[int((1-args.alpha)*xf.shape[0]):,:] = 0
		x_l = torch.fft.ifft(xf*mask, dim=0).real
		x_g = x - x_l
	else:
		xf = torch.fft.fft(x, dim=1)
		mask = torch.ones_like(xf)
		mask[:,int((1-args.alpha)*xf.shape[1]):,:] = 0
		x_l = torch.fft.ifft(xf*mask, dim=1).real
		x_g = x - x_l	   
	return x_g, x_l

def train(model, train_loader, lr, epoch_update,learning_rate_change, weight_decay, num_epochs, gradclip, alpha):
	optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
	def lr_scheduler(optimizer, epoch, lr_decay_rate, decayEpoch=[]):
					if epoch in decayEpoch:
						for param_group in optimizer.param_groups:
							param_group['lr'] *= lr_decay_rate
						return optimizer
					else:
						return optimizer
	criterion = torch.nn.MSELoss()
	for epoch in range(num_epochs):
		for idx, (X,Y,Z) in enumerate(train_loader):
			model.train()
			loss = torch.tensor(0.0)
			##### X global and TV ground #######
			X_g, X_l = FourierFilter(X,3)
			##### Y global and TV ground #######
			Y_g, Y_l = FourierFilter(Y,3)
			##### Z global and TV ground #######
			Z_g, Z_l = FourierFilter(Z,3)

			###### train ######
			ix_g, ny_g, iy_g , nz_g , ix_l, iy_l, nz_l, n_z = model(X,Y,args.alpha)
			###### loss g_id ######
			loss_g_id = (criterion(ix_g,X_g) + criterion(iy_g,Y_g))/2.0
			###### loss g ######
			loss_g = (criterion(ny_g,Y_g) + criterion(nz_g,Z_g))/2.0
			###### loss l_id ######
			loss_l_id = (criterion(ix_l,X_l) + criterion(iy_l,Y_l))/2.0
			###### loss l ######
			loss_l = criterion(nz_l,Z_l)
			###### loss con ######
			loss_con = criterion(n_z,Z)
			loss = 0.2*loss_g_id + 0.2*loss_g + 0.2*loss_l_id + 0.3*loss_l + 0.1*loss_con
			# ===================backward====================
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), gradclip)
			optimizer.step()
		# schedule learning rate decay	
		lr_scheduler(optimizer, epoch, lr_decay_rate=learning_rate_change, decayEpoch=epoch_update)				
		if (epoch) % 50 == 0:
			print('********** Epoche %s **********' %(epoch+1))
			print("loss g_id: ", loss_g_id.item())
			print("loss g: ", loss_g.item())
			print("loss l_id: ", loss_l_id.item())
			print("loss l: ", loss_l.item())
			print("loss con: ", loss_con.item())
			print("loss : ", loss.item())
	return model
#==============================================================================
# Load data
#==============================================================================
############ scale ###########
Z = data_from_name(args.dataset,args.noise)
M,D = Z.shape[0],Z.shape[1]
Zmax, Zmin = np.max(Z), np.min(Z)
Z = (Z-Zmin)/(Zmax-Zmin)-0.5
##### split into train and test set #####
Ztrain = Z[0:args.M]
Ztest = Z[args.M:args.M+args.L]
##### split into segments #####
S = int(args.M/args.M_S)
Xtrain = []
for i in np.arange(0,S,1):
	Xtrain.append(Ztrain[i*args.M_S:(i+1)*args.M_S])
Xtest = []
for i in np.arange(0,int(args.L/args.M_S),1):
	Xtest.append(Ztest[i*args.M_S:(i+1)*args.M_S])
Xtrain = np.array(Xtrain)
Xtest = np.array(Xtest)
###### transfer to tensor #####
Xtrain = torch.from_numpy(Xtrain).float().contiguous()
###### Create Dataloader objects #####
train_data = torch.utils.data.TensorDataset(Xtrain[:S-2],Xtrain[1:S-1],Xtrain[2:S])
train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=args.batch)
#==============================================================================
# Model training
#==============================================================================
model_ = TVNN(D)
out = []
for arg_i in np.arange(0.1,0.35,0.1):
	args.alpha = arg_i
	model = copy.deepcopy(model_)
	model = train(model, train_loader,lr=args.lr, learning_rate_change=args.lr_decay, epoch_update=args.lr_update,
		weight_decay=args.wd, num_epochs = args.epochs, gradclip=args.gradclip,alpha=args.alpha)
	#******************************************************************************
	# Prediction
	#******************************************************************************
	model.eval()
	X_fir = Xtrain[S-2]
	X_sec = Xtrain[S-1]
	Xpred = []
	error = []
	for i in np.arange(0,int(args.L/args.M_S),1):
		__, X_fir_l = FourierFilter(X_fir,2)
		X_sec_g, X_sec_l = FourierFilter(X_sec,2)
		##### global pred ######
		__, nY_g = model.Global_module(X_sec_g)
		##### TV pred ######
		__, ___, nY_l = model.TV_module(X_fir_l,X_sec_l)
		##### Sum pred ######
		X_fir = X_sec
		X_sec = nY_g + nY_l
		tmp = X_sec.detach().numpy()
		Xpred.append(tmp)
		error.append(np.sum(np.sum((tmp-Xtest[i])**2,axis=1)))
	Xpred,error = np.array(Xpred),np.array(error)
	#******************************************************************************
	# Results
	#******************************************************************************
	RMSE = np.sqrt(sum(error))/args.L
	out.append(RMSE)
	#******************************************************************************
	# Save data
	#******************************************************************************
address = './results/'
if not os.path.exists(address):
	os.makedirs(address)
df = pd.DataFrame(out, index=[i for i in range(1,4)])
df.to_csv(address+str(args.seed)+".csv")



