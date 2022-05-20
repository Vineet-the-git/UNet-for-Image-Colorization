import torch
import numpy as np
from PIL import Image
import os
from skimage.color import lab2rgb
import matplotlib.pyplot as plt


CHECKPOINT_PATH = "/home/vineet/Desktop/task/drive-download-20220407T041406Z-001/experiments/Final_L1loss"

# Function to save the checkpoint
def save_checkpoint(checkpoint, epoch):
    torch.save(checkpoint, os.path.join(CHECKPOINT_PATH,str(epoch)+'.pt'))
    print("Model saved successfully after {} epochs".format(checkpoint['epoch']))

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


# Function to save the figure as 3 *4 grid
def save_fig(pred, ab, Ls, epoch, show):
    rgb_images = lab_to_rgb(Ls, ab)
    rgb_pred = lab_to_rgb(Ls, pred)
    fig = plt.figure(figsize=(15, 8))
    for i in range(4):

        ax = plt.subplot(3, 4, i + 1)
        ax.imshow(Ls[i][0].cpu(), cmap='gray')
        ax.axis("off")

        ax = plt.subplot(3, 4, i + 1 + 4)
        ax.imshow(rgb_pred[i])
        ax.axis("off")

        ax = plt.subplot(3, 4, i + 1 + 8)
        ax.imshow(rgb_images[i])
        ax.axis("off")
    if show:
        plt.show()
    fig.savefig("/home/vineet/Desktop/task/drive-download-20220407T041406Z-001/predictions/visualization_L1loss_final/{}.png".format(epoch+1))
