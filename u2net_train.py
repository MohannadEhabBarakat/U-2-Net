import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os

from data_loader import DatasetPipline

from model import U2NET
from model import U2NET_lite
from model import U2NETP

from train_loop import train as fit
import wandb
from dice_loss import DiceLoss
# ------- 1. define loss function --------

bce_loss = DiceLoss()# nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
	
	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	# print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

	return loss0, loss
	
# ------- 2. set the directory of training dataset --------

model_name = 'u2net' #'u2netp'
model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

epoch_num = 100000
batch_size_train = 12
train_num = 0
val_num = 0
 
train_ = DatasetPipline(src="data", split="Train")
train = DataLoader(train_, batch_size=32, shuffle=True, num_workers=1)

val_ = DatasetPipline(src="data", split="Validation")
val = DataLoader(val_, batch_size=16, shuffle=True, num_workers=1)


print("train size: ", train_.__len__())
print("val size: ", val_.__len__())
# ------- 3. define model --------
# define the net
if(model_name=='u2net'):
    net = U2NET(1, 1)
elif(model_name=='u2netp'):
    net = U2NETP(1,1)
elif(model_name=='u2net_lite_refactor'):
	net = U2NET_lite()


# if torch.cuda.is_available():
#     net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
patience = 20
ite_num4val = 0
save_frq = 500 # save the model every 2000 iterations

def new():
    dirs = os.listdir("saved_models/expeiraments/")
    dirs = sorted([int(i) for i in dirs if i.lower() != "none"])
    name = str(dirs[-1]+1)

    return name
 
expeirament_name = input("expeirament id: ")
if expeirament_name.strip() == "":
    expeirament_name = new()
    os.system("mkdir saved_models/expeiraments/"+expeirament_name)

print("** expeirament id is "+expeirament_name+" **")


wandb.init(project='u2net', tags=[expeirament_name, model_name]) 


fit(net=net, epoch_num=epoch_num, val_ds=val, train_ds=train, save_frq=save_frq,
         muti_bce_loss_fusion=muti_bce_loss_fusion, ite_num=ite_num, optimizer=optimizer,
         ite_num4val=ite_num4val, batch_size_train=batch_size_train, train_num=train_num, 
		 trail_name=expeirament_name, patience=patience)

