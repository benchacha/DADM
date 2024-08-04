import os
import torch
from PIL import Image
import torchvision.transforms.functional as TF

class SingleDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        super(SingleDataset, self).__init__()
        self.opt = opt
        self.names = os.listdir(opt['dataroot_hazy'])

    def __getitem__(self, index):

        hazy_name = self.names[index]
        # infors = self.names[index].split('_')
        # clear_name = infors[0] + self.suffix

        hazy_img = Image.open(os.path.join(self.opt['dataroot_hazy'], hazy_name)).convert('RGB')
        # clear_img = Image.open(os.path.join(self.opt['dataroot_GT'], clear_name)).convert('RGB')

        if (self.opt['patch_size'] == 'whole_img') is False:
            hazy_img = hazy_img.resize(self.opt['patch_size'])
            # clear_img = clear_img.resize(self.opt['patch_size'])

        hazy_img = TF.to_tensor(hazy_img)
        # clear_img = TF.to_tensor(clear_img)

        return hazy_img, hazy_name

    def __len__(self):

        return len(self.names)