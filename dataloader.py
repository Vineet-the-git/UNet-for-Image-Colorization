import numpy as np
from PIL import Image
from skimage.color import rgb2lab
import glob

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim


SIZE = 256

path = "/home/vineet/Desktop/task/data/raw"
    
paths = glob.glob(path + "/*.jpg") # Grabbing all the image file names
np.random.seed(123)
paths_subset = np.random.choice(paths, 4000, replace=False) # choosing 4000 images randomly
rand_idxs = np.random.permutation(4000)
train_idxs = rand_idxs[:3200] # choosing the first 3200 as training set
val_idxs = rand_idxs[3200:] # choosing last 800 as validation set
train_paths = paths_subset[train_idxs]
val_paths = paths_subset[val_idxs]
print(len(train_paths), len(val_paths))

class ColorizationDataset(Dataset):
    def __init__(self, paths, split='train'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE),  Image.BICUBIC),
                transforms.RandomHorizontalFlip(), # A little data augmentation!
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((SIZE, SIZE),  Image.BICUBIC)
        
        self.split = split
        self.size = SIZE
        self.paths = paths
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1
        
        return {'L': L, 'ab': ab}
    
    def __len__(self):
        return len(self.paths)

def make_dataloaders(batch_size=4, n_workers=4, pin_memory=True, **kwargs): # A handy function to make our dataloaders
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader

def data_loader(split):
    if split == "train":
        return make_dataloaders( paths=train_paths, split='train')
    else:
        return make_dataloaders(paths=val_paths, split='val')
    


# if __name__ == "__main__":
#     train_dl = make_dataloaders(paths=train_paths, split='train')
#     val_dl = make_dataloaders(paths=val_paths, split='val')

#     data = next(iter(train_dl))
#     Ls, abs_ = data['L'], data['ab']
#     print(Ls.shape, abs_.shape)
#     ss = []
#     for i in range(4):
#         img = abs_[0].numpy()
#         img = img.astype(np.float64)
#         ssim_none = ssim(img, img+0.02, data_range=img.max() - img.min(), channel_axis = 0)
#         ss.append(ssim_none)
#     print(ss)
    
