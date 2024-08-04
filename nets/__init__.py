# from .DGHRNet import *
from .DADNet import *


def get_net(opt):

    # if opt['name'] == 'DGHRNet':
    #     net = CondDGHRNet(opt)
    if opt['name'] == 'DADNet':
        net = CondDADNet(opt)

    else:
        raise FileNotFoundError("cant found this net")
    
    return net

        
