import os
import yaml

# load *.yml
def yamlread(path):
    with open (path, 'r') as f:
        config = yaml.safe_load(f)
        return config
    
def mkfile(filepath):
    if os.path.exists(filepath) is False:
        os.mkdir(filepath)
        print('create {}'.format(filepath))

# change gpu
def get_option(path):
# def get_option(path, times):
    config = yamlread(path)

    if config["gpu_ids"] is not None:
        gpu_list = ",".join(str(x) for x in config["gpu_ids"])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        print("export CUDA_VISIBLE_DEVICES=" + gpu_list)

    # config['datasets']['name'] += str(times)
    
    mkfile(config['path']['save_path'])

    mkfile(os.path.join(config['path']['save_path'], config['model']))
    mkfile(os.path.join(config['path']['save_path'], config['model'], config['net_G']['name']))
    ckpt = os.path.join(config['path']['save_path'], config['model'], 
                        config['net_G']['name'], config['datasets']['name'])
    
    # ckpt = os.path.join(config['save_path'], 
    #                            config['model'] + '_' + config['net'], config['datasets']['datatype'])

    mkfile(ckpt)
    if config['phase'] == 'test':
        mkfile(os.path.join(ckpt, 'val_images'))
        # if config['path']['state']:
        #     mkfile(os.path.join(ckpt, 'states'))

    elif config['phase'] == 'train':
        mkfile(os.path.join(ckpt, 'model_path'))

    config['ckpts'] = ckpt
        
    return config

def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg
