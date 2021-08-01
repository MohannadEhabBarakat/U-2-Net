
import torch
from torch.autograd import Variable
from callbacks import EarlyStopping, IOU
import numpy as np
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau

def val(net, val_ds, muti_bce_loss_fusion):
    net.train(mode=False)
    total_loss = 0
    total_tar_loss = 0
    i = 0
    iou = IOU()

    for i, data in enumerate(val_ds):        
        
        inputs, labels = data['image'], data['label']        
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)
            net.to(torch.device("cpu"))

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        tar_loss, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
        iou.calc(labels_v, d0)
        # print("type(d6) ",type(d6))
        # print("type(labels_v) ",type(labels_v))
        
        total_loss = total_loss+loss.data.item() if total_loss else loss.data.item()
        total_tar_loss = total_tar_loss+tar_loss.data.item() if total_tar_loss else tar_loss.data.item()
        print("[sub ite: %d, loss: %4f, tar: %4f, iou: %4f]" %(i, loss.data.item(), tar_loss, iou.mean()))
        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss, tar_loss

        
    return total_tar_loss/(i+1), total_loss/(i+1), iou.mean()






def train(net, epoch_num, val_ds, train_ds, save_frq,
             muti_bce_loss_fusion, ite_num, optimizer, ite_num4val,
             batch_size_train, train_num, trail_name="test", patience=50):
    running_loss = 0.0
    running_tar_loss = 0.0
    early_stopping = EarlyStopping(patience)
    iou = IOU()
    val_ite_num = 0
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=20,
                                   threshold=1e-4, verbose=True,
                                   factor=0.25)


    if torch.cuda.is_available():
        net.cuda()
    
    def size(x):    
        res = 1
        for i in x.shape:
            res *= i
        return res 
    n_prams = np.sum([size(x) for x in net.parameters()])
    print("model parameters: ", n_prams)
    print("model mem size estimate: %4f Gb" %((n_prams/1000000000)*32))
        
    break_ = False

    for epoch in range(0, epoch_num):
        net.train()
        
        for i, data in enumerate(train_ds):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                         requires_grad=False)
                net.cuda()
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)
            
            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

            loss.backward()
            optimizer.step()
            iou.calc(labels_v, d0)

            # # print statistics
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()

            
            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f, max: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val, d0.max().data.item()))
            
            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss


            if ite_num % save_frq == 0:
                print("start val ite: %d"%(val_ite_num))
                val_tar_loss, val_loss, val_iou = val(net, val_ds, muti_bce_loss_fusion)
                wandb.log({
                    "val_tar_loss": val_tar_loss,
                    "val_loss":val_loss,
                    "val_iou":val_iou 
                })
                torch.save(net.state_dict(), "./saved_models/expeiraments/" + trail_name +"/_bce_itr_%d_train_%3f_tar_%3f_val_loss_%4f_val_iou_%4f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val, val_loss, val_iou))
                scheduler.step(running_loss/save_frq)
                wandb.log({
                    "train_iou":iou.mean(),
                    "train_loss":running_loss/save_frq,
                    "running_tar_loss":running_tar_loss/save_frq
                })

                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0
                val_ite_num += 1
                del iou
                iou = IOU()

                if early_stopping.eval(val_loss, epoch): 
                    print("Early stopping:\n    best loss is %4f, on epoch %d" %(early_stopping.last_loss, early_stopping.epoch))
                    break_ = True
                    break
         
        if break_: break