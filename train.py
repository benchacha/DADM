import utils as util
from models import get_model
from datasets import get_dataloader
import yaml
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_path', type=str, required=False, default=None)
    args = parser.parse_args()
    config = util.get_option(args.opt_path)

    model = get_model(config)

    train_loader = get_dataloader(config['datasets']['train'])
    valid_loader = get_dataloader(config['datasets']['valid'])

    model.train(train_loader, valid_loader)

