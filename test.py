import utils as util
import torch
from models import get_model
from datasets import get_dataloader
import argparse

if __name__ == '__main__':

    torch.manual_seed(259)

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_path', type=str, required=False, default=None)
    parser.add_argument('--times', type=int, required=True, default=50)
    args = parser.parse_args()
    print(args.times)
    # config = util.get_option(args.opt_path, args.times)
    config = util.get_option(args.opt_path)

    # exit()


    model = get_model(config)

    valid_loader = get_dataloader(config['datasets']['valid'])

    # model.show_state(valid_loader)

    type = ['ori', 'imp', 'final', 'base']

    # t1, center1 = model.test4(valid_loader, type=type[1], r=int(args.times), num=500, range_center=int(40))
    t1, center1 = model.test(valid_loader, type=type[0], r=50, num=500, range_center=int(40))
