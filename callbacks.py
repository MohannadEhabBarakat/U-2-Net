import torch

class EarlyStopping():
    def __init__(self, patience):

        self.patience = patience
        self.epoch = 0
        self.last_patience = 0
        self.last_loss = None

    def eval(self, val_loss, epoch):
        if not self.last_loss:
            self.last_loss = val_loss
            self.epoch = epoch
            return False
        
        if val_loss <= self.last_loss:
            self.last_loss = val_loss
            self.epoch = epoch
            self.last_patience = 0

        else:
            self.last_patience += 1

        if self.last_patience == self.patience: return True


class IOU():
    def __init__(self):
        self.result = 0
        self.n_calls = 0
    
    def calc(self, y_true, y_pred):
        res = self.mean_iou(y_true, y_pred)
        self.result += res.data.item()
        self.n_calls += 1
        
        del res
        # return self.mean()
        
    def mean(self): return self.result / self.n_calls

    def mean_iou(self, y_true, y_pred):
        y_pred = (y_pred>0.5).float()
        y_true = (y_true>0.5).float()
        
        intersect = torch.sum(y_true * y_pred, dim=[1, 2, 3])
        union = torch.sum(y_true, dim=[1, 2, 3]) + torch.sum(y_pred, dim=[1, 2, 3])
        smooth = torch.ones_like(intersect)*1e-5

        result = torch.mean((intersect + smooth) / (union - intersect + smooth))

        del y_pred, intersect, union, smooth

        return result

if __name__=='__main__':
    early_stopping = EarlyStopping(3)
    val_losses = [0.11, 0.2, 0.1, 0.2, 0.3, 0.4, 0.5]

    for epoch, val_loss in enumerate(val_losses):
        # print("epoch: ", epoch)
        # print("val_loss: ", val_loss)

        eval_ = early_stopping.eval(val_loss, epoch)
        # print("early_stopping.epoch: ", early_stopping.epoch)
        # print("early_stopping.last_loss: ", early_stopping.last_loss)
        # print("early_stopping.last_patience: ", early_stopping.last_patience)
        # print("\n")
        if eval_: break
    
    assert early_stopping.epoch == 2
    assert early_stopping.last_loss == 0.1
    assert early_stopping.last_patience == 3

    iou = IOU()
    y_true = torch.ones((32, 1, 10, 10))
    y_pred = torch.ones((32, 1, 10, 10))
    assert iou.mean_iou(y_true, y_pred) == 1.0
    
    assert iou(y_true, y_pred) == 1.0
    
    assert iou(y_true, y_pred) == 1.0

    # iou = IOU()
    # iou(y_true, y_pred-1)
    # print(iou.result)
    # iou(y_true, y_pred-1)
    # print(iou.result)
    # print(iou.mean_iou(y_true, y_pred*0.5))



    