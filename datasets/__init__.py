
from torch.utils.data import DataLoader
from .TrainDataset import TrainDataset
from .ValidDataset import ValidDataset
from .SingleDataset import SingleDataset

def get_datasets(opt):
    if opt['mode'] == 'TrainDataset':
        dataset = TrainDataset(opt)
    elif opt['mode'] == 'ValidDataset':
        dataset = ValidDataset(opt)
    elif opt['mode'] == 'SingleDataset':
        dataset = SingleDataset(opt)
    
    # elif opt['mode'] == 'Pairloader':
    #     dataset = PairLoader(opt)
    else:
        raise FileNotFoundError("cant found this dataset")

    return dataset
    
    

def get_dataloader(opt):
    dataset = get_datasets(opt)
    loader = DataLoader(dataset, 
                        batch_size=opt['batch_size'], 
                        shuffle=opt['use_shuffle'], 
                        num_workers=opt['n_workers'])
    
    return loader