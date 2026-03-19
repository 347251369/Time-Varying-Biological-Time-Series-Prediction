import argparse
import numpy as np
import torch
from read_dataset import data_from_name
from model import *
import os
import matplotlib.pyplot as plt
#==============================================================================
# Training settingss
#==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default='0',  help='seed value')
#
parser.add_argument('--dataset', type=str, default='pendulum', metavar='N', help='dataset name')
#
parser.add_argument('--noise', type=float, default=0.1, help='noise of data')
#
parser.add_argument('--M', type=int, default=4000, help='size of trainning')
#
parser.add_argument('--L', type=int, default=64, help='size of prediction')
#
parser.add_argument('--M_S', type=int, default=8, help='size of segments')
#
parser.add_argument('--batch', type=int, default=8, metavar='N', help='batch size')
#
parser.add_argument('--alpha', type=float, default=0.3, metavar='N', help='batch size')
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
    xf = np.fft.fft(x,axis=0)
    mask = np.ones((xf.shape[0],xf.shape[1]))
    mask[int((1-args.alpha)*xf.shape[0]):,:] = 0
    x_l = np.fft.ifft(xf*mask, axis=0).real
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
        for idx, (X_g,X_l,Y_g,Y_l,Z_g,Z_l) in enumerate(train_loader):
            model.train()
            loss = torch.tensor(0.0)
            ###### train ######
            ix_g, ny_g, iy_g , nz_g , ix_l, iy_l, nz_l, n_z = model(X_g,X_l,Y_g,Y_l)
            ###### loss g_id ######
            loss_g_id = (criterion(ix_g,X_g) + criterion(iy_g,Y_g))/2.0
            ###### loss g ######
            loss_g = (criterion(ny_g,Y_g) + criterion(nz_g,Z_g))/2.0
            ###### loss l_id ######
            loss_l_id = (criterion(ix_l,X_l) + criterion(iy_l,Y_l))/2.0
            ###### loss l ######
            loss_l = criterion(nz_l,Z_l)
            ###### loss con ######
            loss_con = criterion(n_z,Z_g+Z_l)
            loss = 0.1*loss_g_id + 0.1*loss_g + 0.3*loss_l_id + 0.4*loss_l + 0.1*loss_con
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
Z = 2*((Z-Zmin)/(Zmax-Zmin)-0.5)
Z = Z[0:args.M+args.L]
Z_g,Z_l = FourierFilter(Z,2)
##### split into train and test set #####
Ztrain_g = Z_g[0:args.M]
Ztrain_l = Z_l[0:args.M]
Ztest = Z[args.M:args.M+args.L]
##### split into segments #####
S = int(args.M/args.M_S)
Xtrain_g = []
Xtrain_l = []
for i in np.arange(0,S,1):
    Xtrain_g.append(Ztrain_g[i*args.M_S:(i+1)*args.M_S])
    Xtrain_l.append(Ztrain_l[i*args.M_S:(i+1)*args.M_S])
Xtest = []
for i in np.arange(0,int(args.L/args.M_S),1):
    Xtest.append(Ztest[i*args.M_S:(i+1)*args.M_S])
Xtrain_g = np.array(Xtrain_g)
Xtrain_l = np.array(Xtrain_l)
Xtest = np.array(Xtest)
###### transfer to tensor #####
Xtrain_g = torch.from_numpy(Xtrain_g).float().contiguous()
Xtrain_l = torch.from_numpy(Xtrain_l).float().contiguous()
###### Create Dataloader objects #####
train_data = torch.utils.data.TensorDataset(Xtrain_g[:S-2],Xtrain_l[:S-2],Xtrain_g[1:S-1],Xtrain_l[1:S-1],Xtrain_g[2:S],Xtrain_l[2:S])
train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=args.batch)
#==============================================================================
# Model training
#==============================================================================
model = TVNN(D,args.alpha)
model = train(model, train_loader,lr=args.lr, learning_rate_change=args.lr_decay, epoch_update=args.lr_update,
            weight_decay=args.wd, num_epochs = args.epochs, gradclip=args.gradclip,alpha=args.alpha)
#******************************************************************************
# Prediction
#******************************************************************************
model.eval()

X_fir_g = Xtrain_g[S-2]
X_fir_l = Xtrain_l[S-2]
X_sec_g = Xtrain_g[S-1]
X_sec_l = Xtrain_l[S-1]
Xpred = []
error = []
for i in np.arange(0,int(args.L/args.M_S),1):
    ##### global pred ######
    __, nY_g = model.Global_module(X_sec_g)
    ##### TV pred ######
    __, ___, nY_l = model.TV_module(X_fir_l,X_sec_l)
    ##### Sum pred ######
    X_fir_g = X_sec_g
    X_fir_l = X_sec_l
    X_sec_g = nY_g
    X_sec_l = nY_l
    X_sec = X_sec_g + X_sec_l
    tmp = X_sec.detach().numpy()
    Xpred.append(tmp)
    error.append(np.sum(np.sum((tmp-Xtest[i])**2,axis=1)))
Xpred,error = np.array(Xpred),np.array(error)
#******************************************************************************
# Results
#******************************************************************************
Xpred = np.reshape(Xpred, (-1,Xpred.shape[2]))
l1 = 16
l2 = 32
l3 = 48
RMSE1 = np.sqrt(np.sum(np.sum((Xpred[0:l1]-Ztest[0:l1])**2,axis=1))/l1)
RMSE2 = np.sqrt(np.sum(np.sum((Xpred[0:l2]-Ztest[0:l2])**2,axis=1))/l2)
RMSE3 = np.sqrt(np.sum(np.sum((Xpred[0:l3]-Ztest[0:l3])**2,axis=1))/l3)
print("RMSE1:"+str(RMSE1))
print("RMSE2:"+str(RMSE2))
print("RMSE3:"+str(RMSE3))
#******************************************************************************
# Save data
#******************************************************************************
address = './results/'
if not os.path.exists(address):
	os.makedirs(address)
torch.save(model, 'TVNN.pth')
#******************************************************************************
# draw pic
#******************************************************************************
#legend
plt.title("  MSE :" + str(round(RMSE3,7)))
plt.xlabel('Time')
plt.ylabel('Value')
plt.xlim(xmin=0,xmax=args.M+args.L+1)
dim = 0
plt.ylim(ymin=Z[0:args.M+args.L,dim].min()-0.3, ymax=Z[0:args.M+args.L,dim].max()+0.3)
# draw line
plt.plot(np.arange(1,args.M+1,1), Z[0:args.M,dim], color='blue', linestyle='-',marker = "o", markersize=3)
plt.plot(np.arange(args.M+1,args.M+args.L+1,1),Ztest[:,dim], label='True',color='green', linestyle='-',marker = "o", markersize=3)
plt.plot(np.arange(args.M,args.M+args.L+1,1), np.concatenate([Z[args.M-1:args.M,dim],Xpred[:,dim]]),label='Prediction',color='red', linestyle='-',marker = "o", markersize=3)
name = address +'/'+ 'TVNN.png'
plt.savefig(name, dpi=100)
plt.show()
plt.close()