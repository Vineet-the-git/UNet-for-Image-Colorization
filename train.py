import matplotlib.pyplot as plt
import statistics
import numpy as np
import torch
from torch import nn, optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from dataloader import data_loader
from unet import UNet 
from saving import  lab_to_rgb, save_fig ,save_checkpoint
from skimage.metrics import structural_similarity as ssim

LEARNING_RATE = 0.003
MAX_ITER = 100
CHECKPOINT = 5
loss_function = nn.L1Loss()
# loss_function = nn.MSELoss()

def ss_Index(pred , ab):
    ss = []
    for i in range(pred.shape[0]):
        org = ab[i]
        pre = pred[i]
        org = org.astype(np.float64)
        pre = pre.astype(np.float64)

        ssi = ssim(org, pre, data_range=pre.max() - pre.min(), channel_axis = 2)
        ss.append(ssi)
    return statistics.mean(ss)

# Implement loss function and optimizer
def lossFunc(pred, y):
    
    return loss_function(pred, y)


def train_op(model, x, y, optimizer):
    pred = model(x)
    loss = lossFunc(pred, y)  
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return pred, loss

# Code the training loop, validation loop
if __name__ == "__main__":
    print("Dataloader are being prepared!!!")
    train_dl = data_loader(split='train')
    val_dl = data_loader(split='val')

    if train_dl!= None and val_dl!= None:
        print("Train and validation dataloaders are loaded successfullly!!")


    model = UNet(n_channels = 1, n_classes = 2)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr = LEARNING_RATE)
    Loss_train = []
    Loss_val = []
    SSIM_val = []

    for epoch in range(MAX_ITER):
        print("Epoch: ",str(epoch+1))

        loss_batch = []
        for data in train_dl:
            Ls, ab = data['L'], data['ab']
            Ls = Ls.to(device)
            ab = ab.to(device)

            pred, loss = train_op(model, Ls, ab, optimizer)
            loss = loss.detach().cpu().numpy()
            loss = float(loss)
            loss_batch.append(loss)

        avg_loss = statistics.mean(loss_batch)
        print("Training loss after epoch {} : {}".format(epoch+1, avg_loss))

        Loss_train.append(avg_loss)

        # validation loop
        flag = False
        loss_val_batch = []
        ssim_val_list = []
        for data_val in val_dl:
            Ls, ab = data_val['L'], data_val['ab']
            Ls = Ls.to(device)
            ab = ab.to(device)

            model.eval()
            with torch.no_grad():
                pred = model(Ls)
                loss = loss_function(pred, ab)
                pred_rgb = lab_to_rgb(Ls, pred)
                org_rgb = lab_to_rgb(Ls, ab)
                ssim_val_list.append(ss_Index(pred_rgb, org_rgb))

                # Saving a figure for visualization
                if flag == False:
                    save_fig(pred, ab, Ls, epoch, show = False)
                    flag = True
            loss = loss.detach().cpu().numpy()
            loss = float(loss)
            loss_val_batch.append(loss)

            

        avg_val_loss = statistics.mean(loss_val_batch)
        ssim_val = statistics.mean(ssim_val_list)
        Loss_val.append(avg_val_loss)
        SSIM_val.append(ssim_val)
        print("Validation loss after epoch {} : {}".format(epoch+1, avg_val_loss))
        print("Validation SSIM after epoch {} : {}".format(epoch+1, ssim_val))

        # Saving the checkpoint
        if (epoch+1)%CHECKPOINT == 0:
            checkpoint={
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'learning_rate': LEARNING_RATE,
                            'loss_train': Loss_train,
                            'loss_val': Loss_val,
                            'ssim_val': SSIM_val
                        }
            save_checkpoint(checkpoint, epoch+1)