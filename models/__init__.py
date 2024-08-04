from .DensityAwareDiffusionModel import DensityAwareDiffusionModel
from .SDE import *

def get_model(opt):

    if opt['model'] == 'DADM':
        return DensityAwareDiffusionModel(opt, opt['phase'] == 'train')
    else:
        raise FileNotFoundError("cant found this model")