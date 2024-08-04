import os
import torch
import random
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

def get_suffix(type):
    if type == 'indoor':
        return '.png'
    elif type == 'outdoor':
        return '.jpg'
    elif type == 'Dense' or type == 'NH_Haze':
        return '_GT.png'
    elif type == 'I_HAZY' or type == 'O_HAZY':
        return '_GT.jpg'
    elif type == 'NH-Haze2':
        return ''
    else:
        raise FileNotFoundError("cant found this type of dataset")
        
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        super(TrainDataset, self).__init__()
        self.opt = opt
        self.names = os.listdir(opt['dataroot_hazy'])
        self.randomcrop = transforms.RandomCrop(opt['patch_size'])
        self.suffix = get_suffix(self.opt['type'])

    def __getitem__(self, index):

        hazy_name = self.names[index]
        infors = self.names[index].split('_')
        clear_name = infors[0] + self.suffix

        hazy_img = Image.open(os.path.join(self.opt['dataroot_hazy'], hazy_name)).convert('RGB')
        clear_img = Image.open(os.path.join(self.opt['dataroot_GT'], clear_name)).convert('RGB')
        # clear_img = Image.open(os.path.join(self.opt['dataroot_GT'], hazy_name)).convert('RGB')
        
        if (self.opt['patch_size'] == 'whole_img') is False:

            crop_params = self.randomcrop.get_params(hazy_img, self.opt['patch_size'])
            hazy_img = TF.crop(hazy_img, *crop_params)
            clear_img = TF.crop(clear_img, *crop_params)

        if self.opt['use_flip'] and random.randint(0, 1) == 1:
            hazy_img = TF.hflip(hazy_img)
            clear_img = TF.hflip(clear_img)

        if self.opt['use_rot']:
            angle = random.randint(0, 4)
            hazy_img = TF.rotate(hazy_img, 90 * angle)
            clear_img = TF.rotate(clear_img, 90 * angle)

        hazy_img = TF.to_tensor(hazy_img)
        clear_img = TF.to_tensor(clear_img)

        return hazy_img, clear_img, hazy_name

    def __len__(self):

        return len(self.names)
    
