# Prepare a script that takes as input grayscale image, model path and produces a color image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.color import rgb2lab, lab2rgb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from unet import UNet 
import argparse
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description="Colourize image")
    parser.add_argument("--p_img", help="path to the grayscale image", required=True)
    parser.add_argument("--p_model", help = "path to the model checkpoint", required=True)
    
    args = parser.parse_args()

    return args

def rgb_to_lab(img):
    img = np.array(img)
    img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b
    img_lab = transforms.ToTensor()(img_lab)
    L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
    ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1
    L = L.unsqueeze(0)
    ab = ab.unsqueeze(0)
        
    return {'L': L, 'ab': ab}

def lab_to_rgb(L, ab):
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    Lab = Lab[0]
    img_rgb = lab2rgb(Lab)
    return img_rgb



if __name__ == "__main__":
    args = parse_args()
    checkpoint = torch.load(args.p_model)
    img_org = Image.open(args.p_img)
    img_org = img_org.resize((256, 256), Image.BICUBIC)
    image = rgb_to_lab(img_org)

    # Loading the model
    model = UNet(n_channels = 1, n_classes = 2)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    with torch.no_grad():
        # predicting using the model
        L = image["L"].to(device)
        ab = image["ab"].to(device)
        ab_pred = model(L)
        rgb_pred = lab_to_rgb(L, ab_pred)
        
        fig = plt.figure(figsize=(20,15))

        ax = plt.subplot(1, 3, 1)
        ax.imshow(image['L'][0][0].cpu(), cmap='gray')
        ax.axis("off")

        ax = plt.subplot(1, 3, 2)
        ax.imshow(rgb_pred)
        ax.axis("off")

        ax = plt.subplot(1, 3, 3)
        ax.imshow(img_org)
        ax.axis("off")
        plt.show()

    ssim_val = checkpoint["ssim_val"]
    train_loss = checkpoint['loss_train']
    val_loss = checkpoint['loss_val']
    epoch = checkpoint['epoch']
    print(len(ssim_val),len(train_loss),len(val_loss),epoch)




