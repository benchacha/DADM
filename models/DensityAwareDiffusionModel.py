import json
import torch
import logging
import numpy as np
import torch.nn as nn
from utils import *
import random
from collections import OrderedDict
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel
from models.optimizer import Lion
from models.SDE import IRSDE
import torchvision.utils as vutils
from torch.cuda.amp import autocast, GradScaler
from nets import get_net
from skimage.metrics import peak_signal_noise_ratio
import csv


class DensityAwareDiffusionModel():
    def __init__(self, opt, trainflag = True):
        self.opt = opt
        self.device = torch.device("cuda" if opt["gpu_ids"] is not None else "cpu")
        self.trainflag = trainflag
        self.use_cuda = torch.cuda.is_available()
        self.net = get_net(opt['net_G'])
        self.sde = IRSDE(opt['sde'], self.device)
        self.resume = None

        if trainflag:
            self.criterion = nn.L1Loss()
            if opt['train_params']['optimizer'] == 'Lion':
                self.optim = Lion( 
                    self.net.parameters(),
                    lr=opt['train_params']['lr'],
                    betas=opt['train_params']['betas']
                )
            elif opt['train_params']['optimizer'] == 'Adamw':
                self.optim = torch.optim.AdamW(
                    self.net.parameters(),
                    lr=opt['train_params']['lr']
                )
            else:
                raise FileNotFoundError("cant found this optimer")
            

            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optim, 
                T_max = opt['train_params']['niter'], 
                eta_min = opt['train_params']['eta_min']
            )
            self.scaler = GradScaler()

        self._compile()

    def _compile(self):
        if self.use_cuda:
            self.net = self.net.cuda()
            self.net = DataParallel(self.net)

            if self.trainflag:
                self.criterion = self.criterion.cuda()

        if self.opt['path']['checkpoint'] is not None:

            try :
            
                self.load_network(self.opt['path']['checkpoint'], self.net)
                print('load checkpoint {}'.format(self.opt['path']['checkpoint']))

                if self.trainflag:
                    state_path = self.opt['path']['checkpoint'].split('_netG')[0] + '.state'
                    self.resume = self.resume_training(state_path)
                    print('load resume {}'.format(state_path))
            except:
                FileNotFoundError('{} is not exists'.format(self.opt['path']['checkpoint']))

        self.sde.set_model(self.net)
    
    def save_infor(self, stats, train_loss, epoch):
        stats['train_loss'].append(train_loss)

        # Save stats to JSON
        fname_dict = '{}/stats.json'.format(self.opt['ckpts'])
        with open(fname_dict, 'w') as fp:
            json.dump(stats, fp, indent=2)

        # Plot stats
        if self.opt['train_params']['plot']:
            # print('plot')
            loss_str = 'L1 loss'
            plot_per_epoch(self.opt['ckpts'], 'Valid Loss', stats['valid_loss'], loss_str)
            plot_per_epoch(self.opt['ckpts'], self.opt['datasets']['name'], stats['valid_psnr'], 'PSNR (dB)')
            plot_per_epoch(self.opt['ckpts'], 'Train Lr', stats['train_lr'], '')

    # set logger
    def set_logger(self, name=None):

        if name is None:
            if self.opt['phase'] == 'train':
                name = 'train_' + self.opt['model'] + '_' + self.opt['net_G']['name']
            elif self.opt['phase'] == 'test':
                'val_' + self.opt['model'] + '_' + self.opt['net_G']['name']
            else:
                raise FileNotFoundError("cant found this model")
        setup_logger('base', self.opt['ckpts'], name, level=logging.INFO, screen=True, tofile=True)
        
        logger = logging.getLogger('base')

        logger.info(dict2str(self.opt))

        seed = self.opt['seed']
        if seed is None:
            seed = random.randint(1, 10000)
        logger.info('Random seed: {}'.format(seed))
        
        return logger
    
    def show_state(self, loader):
        with torch.no_grad():
            self.net.eval()
            for i, (haze, clear, haze_name) in enumerate(loader):
                if haze_name[0]=='01.png':
                    
                    haze = haze.cuda()
                    clear = clear.cuda()

                    noisy_state = self.sde.noise_state(haze)
                    vutils.save_image(noisy_state.data, 'nh2_10/noisy_state.png', normalize=False)
                    
                    self.sde.forward(clear, save_dir='nh2_10')

                    self.sde.set_mu(noisy_state)
                    dehaze = self.sde.reverse_sde_ori_val(noisy_state, save_dir='nh2_10', save_states=False, r = 0)



    def train(self, train_loader, valid_loader):

        logger = self.set_logger()

        logger.info('train dataset length: {}'.format(len(train_loader.dataset)))
        logger.info('valid dataset length: {}'.format(len(valid_loader.dataset)))

        # self.print_network()

        self.net.train()

        num_batches = len(train_loader)
        dec = int(np.ceil(np.log10(num_batches)))

        total_iters = int(self.opt["train_params"]["niter"])
        total_epochs = int(math.ceil(total_iters / num_batches))
        # total_epochs = self.opt['train_params']['epochs']

        self.best_psnr = 0.
        current_iter = 0

        stats = {
            'dataset_name': self.opt['datasets']['name'], 
            'valid_psnr': [], 'train_loss': [],
            'valid_loss': [], 'train_lr': [],
        }

        if self.resume:
            current_iter = self.resume['current_iter']
            self.best_psnr = self.resume['best_psnr']
            
        logger.info('best_psnr: {}, current_iter: {}'.format(self.best_psnr, current_iter))
        train_loss_meter = AvgMeter()

        
        for epoch in range(total_epochs + 1): 

            for j, (haze, clear, _) in enumerate(train_loader):
                batch_start = datetime.now()

                current_iter += 1

                if current_iter > total_iters:
                    break
                timesteps, states, _ = self.sde.generate_random_states(x0=clear, mu=haze)

                if self.use_cuda:
                    haze = haze.cuda()
                    clear = clear.cuda()
                    timesteps = timesteps.cuda()

                self.optim.zero_grad()
                self.sde.set_mu(haze)
                with autocast(True):
                    # Get noise and score
                    
                    noise = self.sde.noise_fn(states, timesteps.squeeze())
                    score = self.sde.get_score_from_noise(noise, timesteps)

                    # Learning the maximum likelihood objective for state x_{t-1}
                    xt_1_expection = self.sde.reverse_sde_step_mean(states, score, timesteps)
                    xt_1_optimum = self.sde.reverse_optimum_step(states, clear, timesteps)

                    loss = self.criterion(xt_1_expection, xt_1_optimum)

                self.scaler.scale(loss).backward()
                if self.opt['train_params']['use_grad_clip']:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.01)
                self.scaler.step(self.optim)
                self.scaler.update()
                # self.scheduler.step()

                train_loss_meter.update(loss.item())

                print('\rEpoch {:>{dec}d} / {:d} | Iter {:>{dec}d} | Avg loss: {:>1.5f} | Avg train time / batch: {:d} ms'.format(
                    epoch, total_epochs, current_iter, train_loss_meter.avg, 
                    int(time_elapsed_since(batch_start)[1]), dec=dec), end='', flush=True)
                
                # log
                if current_iter % self.opt['train_params']['print_freq'] == 0:

                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(epoch, current_iter)
                    message += 'loss: {:.4f}'.format(loss.item())

                    # logger.info(message)

                if current_iter % self.opt['train_params']['eval_freq'] == 0:
                    print('\nIter {} / {} finished!'.format(current_iter, total_iters))

                    # logger.info('Saving models and training states.')
                    
                    stats = self.eval(valid_loader, stats, current_iter)
                    
                    logger.info('Current_iter: {:>7d}, current psnr: {:.2f}, best psnr: {:.2f}'.format(
                        current_iter, stats['valid_psnr'][-1], self.best_psnr))

                    self.save_infor(stats, train_loss_meter.avg, current_iter)

                    train_loss_meter.reset()

                # break
                self.scheduler.step()

    def eval(self, valid_loader, stats, current_iter):

        loss_meter = AvgMeter()
        psnr_meter = AvgMeter()
        

        with torch.no_grad():
            for j, (haze, clear, haze_name) in enumerate(valid_loader):

                window_size = 4
                mod_pad_h, mod_pad_w = 0, 0
                _, _, h, w = haze.size()
                if h % window_size != 0:
                    mod_pad_h = window_size - h % window_size
                if w % window_size != 0:
                    mod_pad_w = window_size - w % window_size
                haze_pad = F.pad(haze, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

                
                noisy_state = self.sde.noise_state(haze_pad)
                
                if self.use_cuda:
                    haze_pad = haze_pad.cuda()
                    clear = clear.cuda()
                    noisy_state = noisy_state.cuda()

                self.sde.set_mu(haze_pad)
                self.net.eval()

                dehaze = self.sde.reverse_sde_ori_val(noisy_state, save_states=False)
                _, _, h, w = dehaze.size()
                dehaze = dehaze[:, :, 0: h - mod_pad_h * 1, 0: w - mod_pad_w * 1]

                loss = self.criterion(dehaze, clear)
                loss_meter.update(loss.item())
                psnr_meter.update(psnr(dehaze, clear))

                ts = torch.squeeze(dehaze.clamp(0, 1).cpu())

        train_lr = self.optim.param_groups[0]["lr"]

        stats['valid_loss'].append(loss_meter.avg)
        stats['valid_psnr'].append(psnr_meter.avg)
        stats['train_lr'].append(train_lr)

        self.save_network(self.net, 'current_' + str(current_iter), current_iter)
        self.save_training_state(best_psnr=self.best_psnr, state_label='current', current_iter=current_iter)

        if psnr_meter.avg > self.best_psnr:
            self.best_psnr = psnr_meter.avg

            self.save_network(self.net, 'best', current_iter)
            self.save_training_state(best_psnr=self.best_psnr, state_label='best', current_iter=current_iter)

            print('Iter:{} ckeckpoint {} saved!!!'.format(current_iter,
                        os.path.join(self.opt['ckpts'], 'best_netG.pth')))
        # 
        loss_meter.reset()
        psnr_meter.reset()

        return stats

    def test(self, val_data_loader, save_tag = True , type=None, r = 55, num=500, range_center=23):

        print(self.opt['path']['checkpoint'])

        logger = self.set_logger('test_' + type)
        self.print_network()
        psnr_list, ssim_list = [], []
        
        # 写文件
        f_result = open(os.path.join(self.opt['ckpts'], 'results_' + type + '.csv'), 'w')

        with torch.no_grad():
            self.net.eval()
            for i, (haze, clear, haze_name) in enumerate(val_data_loader):

                if haze_name[0]=='02.png':

                    window_size = 4
                    mod_pad_h, mod_pad_w = 0, 0
                    _, _, h, w = haze.size()
                    if h % window_size != 0:
                        mod_pad_h = window_size - h % window_size
                    if w % window_size != 0:
                        mod_pad_w = window_size - w % window_size
                    haze_pad = F.pad(haze, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

                    noisy_state = self.sde.noise_state(haze_pad)

                    if self.use_cuda:
                        
                        haze = haze.cuda()
                        haze_pad = haze_pad.cuda()
                        noisy_state = noisy_state.cuda()


                    self.sde.set_mu(haze_pad)

                    save_dir = os.path.join(self.opt['ckpts'], self.opt['path']['ckpts'], haze_name[0].split('.')[0])

                    if type == 'ori':
                        dehaze = self.sde.reverse_sde_ori_val(noisy_state, save_dir='nh2_10', save_states=True, r=r)
                    elif type == 'imp' or type == 'final':
                        dehaze = self.sde.reverse_sde_imp_val(noisy_state, save_dir=save_dir, save_states=False, r=r)
                    else:
                        FileNotFoundError("cant found this str")

                    _, _, h, w = dehaze.size()
                    dehaze = dehaze[:, :, 0: h - mod_pad_h, 0: w - mod_pad_w]

                    # ts = torch.squeeze(dehaze.clamp(0, 1).cpu())

                    img_psnr = psnr(dehaze.cpu(), clear)
                    img_ssim = ssim2(dehaze.cpu(), clear)

                    psnr_list.append(img_psnr)
                    ssim_list.append(img_ssim)

                    f_result.write('{:<20}, {:.02f}, {:.04f}\n'.format(haze_name[0], img_psnr, img_ssim))
                    # print('\n')
                    logger.info('{:<20}, {:.02f}, {:.04f}'.format(haze_name[0], img_psnr, img_ssim))

                    if save_tag:
                        vutils.save_image(dehaze, os.path.join(self.opt['ckpts'], 'val_images', haze_name[0]), normalize=False)
                        # vutils.save_image(ts, os.path.join('results/region', haze_name[0]))


            f_result.close()
            print('Average PSNR: {}'.format(np.mean(psnr_list)))
            print('Average SSIM: {}'.format(np.mean(ssim_list)))
            logger.info('{}, {}, {}'.format(len(val_data_loader.dataset), np.mean(psnr_list), np.mean(ssim_list)))
            logging.shutdown()

    def test2(self, val_data_loader, save_tag = True, type=None, r = 55, num=500, range_center=23):

        print(self.opt['path']['checkpoint'])

        logger = self.set_logger('sample_' + type)

        logger.info('r: {}, Center: {}, Num: {}'.format(r, range_center, num))

        psnr_list, ssim_list = [], []
        
        # 写文件
        psnr_max_point, ssim_max_point = [], []
        f_result = open(os.path.join(self.opt['ckpts'], 'results_' + type + '.csv'), 'w')

        with torch.no_grad():
            count_1, count_2 = [], []
            t0_psnr, t0_ssim = [], []
            t_break_psnr, t_break_ssim = [], []
            max_psnr, max_ssim = [], []
            imgs_psnr_mean, imgs_psnr_std = [], []
            imgs_ssim_mean, imgs_ssim_std = [], []
            for i, (haze, clear, haze_name) in enumerate(val_data_loader):

                window_size = 4
                mod_pad_h, mod_pad_w = 0, 0
                _, _, h, w = haze.size()
                if h % window_size != 0:
                    mod_pad_h = window_size - h % window_size
                if w % window_size != 0:
                    mod_pad_w = window_size - w % window_size
                haze_pad = F.pad(haze, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

                noisy_state = self.sde.noise_state(haze_pad)

                if self.use_cuda:

                    haze_pad = haze_pad.cuda()
                    noisy_state = noisy_state.cuda()

                self.sde.set_mu(haze_pad)
                self.net.eval()

                save_dir = os.path.join(self.opt['ckpts'], self.opt['path']['ckpts'])
                
                if type == 'ori' or type == 'base':
                    dehazes = self.sde.reverse_sde_original_sample(noisy_state, save_dir=save_dir, save_states=False, r=r)
                elif type == 'imp' or type == 'final':
                    dehazes = self.sde.reverse_sde_improved_sample(noisy_state, save_dir=save_dir, save_states=False, r=r)
                else:
                    FileNotFoundError("cant found this str")

                _, _, h, w = dehazes[0].size()

                imgs_psnr, imgs_ssim = [], []
                for _, dehaze in enumerate(dehazes):

                    dehaze = dehaze[:, :, 0: h - mod_pad_h * 1, 0: w - mod_pad_w * 1]
                    img_psnr = psnr(dehaze.cpu(), clear)
                    img_ssim = ssim2(dehaze.cpu(), clear)

                    imgs_psnr.append(img_psnr)
                    imgs_ssim.append(img_ssim)

                imgs_psnr_mean.append(np.mean(imgs_psnr))
                imgs_psnr_std.append(np.std(imgs_psnr))

                imgs_ssim_mean.append(np.mean(imgs_ssim))
                imgs_ssim_std.append(np.std(imgs_ssim))


                if (i + 1) % 5 == 0:
                    print('{} iamges finished!'.format(i + 1))

                # max
                index = imgs_psnr.index(max(imgs_psnr))
                psnr_max_point.append(index + 1)
                ssim_max_point.append(imgs_ssim.index(max(imgs_ssim)) + 1)
                max_psnr.append(imgs_psnr[index])
                max_ssim.append(imgs_ssim[index])
                # final
                t0_psnr.append(imgs_psnr[-1])
                t0_ssim.append(imgs_ssim[-1])
                # t2
                t_break_psnr.append(imgs_psnr[-range_center])
                t_break_ssim.append(imgs_ssim[-range_center])

                psnr_list.append(imgs_psnr)
                ssim_list.append(imgs_ssim)

                # result=pd.value_counts(list)
                count_1.append(imgs_psnr.index(max(imgs_psnr)) + 1)
                count_2.append(imgs_ssim.index(max(imgs_ssim)) + 1)

                Demo_dict1, Demo_dict2 = {}, {}
                
                for key1, key2 in zip(count_1, count_2):
                    Demo_dict1[key1]=Demo_dict1.get(key1, 0)+1
                    Demo_dict2[key2]=Demo_dict2.get(key2, 0)+1
                
                logger.info('{:<20}, t0: {:.02f}, {:.04f} | t{}: {:.02f}, {:.04f} | max: {:.02f}, {:.04f} | t0_avg: {:.02f}, {:.04f} | t{}_avg: {:.02f}, {:.04f} | max_avg: {:.02f}, {:.04f}'.format(
                    haze_name[0], t0_psnr[-1], t0_ssim[-1], range_center, t_break_psnr[-1], t_break_ssim[-1], max_psnr[-1], max_ssim[-1],
                    np.mean(t0_psnr), np.mean(t0_ssim), range_center, np.mean(t_break_psnr), np.mean(t_break_ssim), np.mean(max_psnr), np.mean(max_ssim)))

                if (i + 1) % 1 == 0:
                    type_name = 'n' + str(num) + '_r' + str(r)
                    
                    plot_all_point(self.opt['ckpts'], type + '_psnr_' + type_name, psnr_list, 'PSNR(dB)', r=r)
                    plot_all_point(self.opt['ckpts'], type + '_ssim_' + type_name, ssim_list, 'ssim', r=r)

                    color = 'green' if type == 'ori' or type == 'base' else 'blue'

                    psnr_count_mean, psnr_count_std = np.mean(psnr_max_point), np.std(psnr_max_point)
                    psnr_t1 = plot_per_epoch_count(self.opt['ckpts'], type + '_count_psnr_' + type_name, 
                                                   Demo_dict1, 'Count', r=r, color=color, mean=psnr_count_mean, std=psnr_count_std)
                    ssim_count_mean, ssim_count_std = np.mean(ssim_max_point), np.std(ssim_max_point)
                    ssim_t1 = plot_per_epoch_count(self.opt['ckpts'], type + '_count_ssim_' + type_name, 
                                                   Demo_dict2, 'Count', r=r, color=color, mean=ssim_count_mean, std=ssim_count_std)
               

                    logger.info('PSNR t1: {:.2f}, Center: {:.2f} | SSIM t1: {:.2f}, center: {:.2f}'.format(
                        psnr_t1, r + 1 - psnr_count_mean, ssim_t1, r + 1 - ssim_count_mean))
                    logger.info('psnr mean: {:.2f}, std {:.2f}, | ssim mean: {:.4f}, std: {:.4f}'.format(
                        np.mean(imgs_psnr_mean), np.mean(imgs_psnr_std), np.mean(imgs_ssim_mean), np.mean(imgs_ssim_std)))

                if i + 1 == num:
                    break

            f_result.close()
            logger.info('Average t0 : {}, {}'.format(np.mean(t0_psnr), np.mean(t0_ssim)))
            logger.info('Average t{}: {}, {}'.format(range_center, np.mean(t_break_psnr), np.mean(t_break_ssim)))
            logger.info('Average max: {}, {}'.format(np.mean(max_psnr), np.mean(max_ssim)))
        # logging.shutdown()
        return psnr_t1, r + 1 - psnr_count_mean

    def test4(self, val_data_loader, save_tag = True, type=None, r = 55, num=500, range_center=23):

        print(self.opt['path']['checkpoint'])

        logger = self.set_logger('sample_' + type)

        logger.info('r: {}, Center: {}, Num: {}'.format(r, range_center, len(val_data_loader.dataset)))


        with torch.no_grad():

            index = []
            for i in range(100, 0, -1):
                index.append(i)

            with open(os.path.join(self.opt['ckpts'], self.opt['datasets']['name'] + '_psnr.csv'), mode='w', newline='') as file:
                writer = csv.writer(file)
                row = ['haze_name'] + index
                writer.writerow(row)

            with open(os.path.join(self.opt['ckpts'], self.opt['datasets']['name'] + '_ssim.csv'), mode='w', newline='') as file:
                writer = csv.writer(file)
                row = ['haze_name'] + index
                writer.writerow(row)

            for i, (haze, clear, haze_name) in enumerate(val_data_loader):

                window_size = 4
                mod_pad_h, mod_pad_w = 0, 0
                _, _, h, w = haze.size()
                if h % window_size != 0:
                    mod_pad_h = window_size - h % window_size
                if w % window_size != 0:
                    mod_pad_w = window_size - w % window_size
                haze_pad = F.pad(haze, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

                noisy_state = self.sde.noise_state(haze_pad)

                if self.use_cuda:

                    haze_pad = haze_pad.cuda()
                    noisy_state = noisy_state.cuda()

                self.sde.set_mu(haze_pad)
                self.net.eval()

                save_dir = os.path.join(self.opt['ckpts'], self.opt['path']['ckpts'],haze_name[0])
                
                if type == 'ori' or type == 'base':
                    dehazes = self.sde.reverse_sde_original_sample(noisy_state, save_dir=save_dir, save_states=False, r=r)
                elif type == 'imp' or type == 'final':
                    # print('imp')
                    dehazes = self.sde.reverse_sde_improved_sample(noisy_state, save_dir=save_dir, save_states=False, r=r)
                else:
                    FileNotFoundError("cant found this str")

                _, _, h, w = dehazes[0].size()

                imgs_psnr, imgs_ssim = [], []

                imgs_psnr.clear()
                imgs_ssim.clear()

                for _, dehaze in enumerate(dehazes):

                    dehaze = dehaze[:, :, 0: h - mod_pad_h * 1, 0: w - mod_pad_w * 1]
                    img_psnr = psnr(dehaze.cpu(), clear)
                    img_ssim = ssim2(dehaze.cpu(), clear)

                    imgs_psnr.append(img_psnr)
                    imgs_ssim.append(img_ssim)

                with open(os.path.join(self.opt['ckpts'], self.opt['datasets']['name'] + '_psnr.csv'), mode='a', newline='') as file:
                    writer = csv.writer(file)

                    row = [haze_name[0]] + imgs_psnr
                    writer.writerow(row)


                # 创建并写入到 ssim.csv 文件中
                with open(os.path.join(self.opt['ckpts'], self.opt['datasets']['name'] + '_ssim.csv'), mode='a', newline='') as file:
                    writer = csv.writer(file)

                    row = [haze_name[0]] + imgs_ssim
                    writer.writerow(row)

    
    def test3(self, val_data_loader, save_tag = True, type=None, r = 55, num=500, range_center=23):

        print(self.opt['path']['checkpoint'])

        logger = self.set_logger('sample_' + type)

        logger.info('r: {}, Center: {}, Num: {}'.format(r, range_center, num))

        psnr_list, ssim_list = [], []
        
        # 写文件
        psnr_max_point, ssim_max_point = [], []
        f_result = open(os.path.join(self.opt['ckpts'], 'results_' + type + '.csv'), 'w')

        with torch.no_grad():

            t0_psnr, t0_ssim = [], []
            t_break_psnr, t_break_ssim = [], []
            max_psnr, max_ssim = [], []
            imgs_psnr_mean, imgs_psnr_std = [], []
            imgs_ssim_mean, imgs_ssim_std = [], []
            for i, (haze, clear, haze_name) in enumerate(val_data_loader):

                window_size = 4
                mod_pad_h, mod_pad_w = 0, 0
                _, _, h, w = haze.size()
                if h % window_size != 0:
                    mod_pad_h = window_size - h % window_size
                if w % window_size != 0:
                    mod_pad_w = window_size - w % window_size
                haze_pad = F.pad(haze, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

                noisy_state = self.sde.noise_state(haze_pad)

                if self.use_cuda:

                    haze_pad = haze_pad.cuda()
                    # clear = clear.cuda()
                    noisy_state = noisy_state.cuda()

                self.sde.set_mu(haze_pad)
                self.net.eval()

                save_dir = os.path.join(self.opt['ckpts'], self.opt['path']['ckpts'])
                
                if type == 'ori' or type == 'base':
                    dehazes = self.sde.reverse_sde_original_sample(noisy_state, save_dir=save_dir, save_states=False, r=r)
                elif type == 'imp' or type == 'final':
                    dehazes = self.sde.reverse_sde_improved_sample(noisy_state, save_dir=save_dir, save_states=False, r=r)
                else:
                    FileNotFoundError("cant found this str")

                _, _, h, w = dehazes[0].size()
                
                imgs_psnr, imgs_ssim = [], []
                for _, dehaze in enumerate(dehazes):
                    
                    dehaze = dehaze[:, :, 0: h - mod_pad_h * 1, 0: w - mod_pad_w * 1]
                    img_psnr = psnr(dehaze.cpu(), clear)
                    img_ssim = ssim2(dehaze.cpu(), clear)

                    imgs_psnr.append(img_psnr)
                    imgs_ssim.append(img_ssim)

                imgs_psnr_mean.append(np.mean(imgs_psnr))
                imgs_psnr_std.append(np.std(imgs_psnr))

                imgs_ssim_mean.append(np.mean(imgs_ssim))
                imgs_ssim_std.append(np.std(imgs_ssim))


                if (i + 1) % 50 == 0:
                    print('{} iamges finished!'.format(i + 1))

                # max
                index = imgs_psnr.index(max(imgs_psnr))
                psnr_max_point.append(index + 1)
                ssim_max_point.append(imgs_ssim.index(max(imgs_ssim)) + 1)
                max_psnr.append(imgs_psnr[index])
                max_ssim.append(imgs_ssim[index])
                # final
                t0_psnr.append(imgs_psnr[-1])
                t0_ssim.append(imgs_ssim[-1])
                # t2
                t_break_psnr.append(imgs_psnr[-range_center])
                t_break_ssim.append(imgs_ssim[-range_center])

                psnr_list.append(imgs_psnr)
                ssim_list.append(imgs_ssim)

                
                logger.info('{:<20}, t0: {:.02f}, {:.04f} | t{}: {:.02f}, {:.04f} | max: {:.02f}, {:.04f} | t0_avg: {:.02f}, {:.04f} | t{}_avg: {:.02f}, {:.04f} | max_avg: {:.02f}, {:.04f}'.format(
                    haze_name[0], t0_psnr[-1], t0_ssim[-1], range_center, t_break_psnr[-1], t_break_ssim[-1], max_psnr[-1], max_ssim[-1],
                    np.mean(t0_psnr), np.mean(t0_ssim), range_center, np.mean(t_break_psnr), np.mean(t_break_ssim), np.mean(max_psnr), np.mean(max_ssim)))

                if (i + 1) % 10 == 0:
                    type_name = 'n' + str(num) + '_r' + str(r)
                    
                    plot_all_point(self.opt['ckpts'], type + '_psnr_' + type_name, psnr_list, 'PSNR(dB)', r=r)
                    plot_all_point(self.opt['ckpts'], type + '_ssim_' + type_name, ssim_list, 'ssim', r=r)

                if i + 1 == num:
                    break

            f_result.close()
            logger.info('Average t0 : {}, {}'.format(np.mean(t0_psnr), np.mean(t0_ssim)))
            logger.info('Average t{}: {}, {}'.format(range_center, np.mean(t_break_psnr), np.mean(t_break_ssim)))
            logger.info('Average max: {}, {}'.format(np.mean(max_psnr), np.mean(max_ssim)))
        # logging.shutdown()
        return _, _

    def test_real(self, val_data_loader, save = True, type=None, r = 55, num = 19, range_center=23):
        print(self.opt['path']['checkpoint'])

        logger = self.set_logger('sample_' + type)

        logger.info('r: {}, Center: {}, Num: {}'.format(r, range_center, num))

        with torch.no_grad():
            for i, (haze, haze_name) in enumerate(val_data_loader):
                window_size = 4
                mod_pad_h, mod_pad_w = 0, 0
                _, _, h, w = haze.size()
                if h > 1600 and w > 1600:
                    continue
                if h % window_size != 0:
                    mod_pad_h = window_size - h % window_size
                if w % window_size != 0:
                    mod_pad_w = window_size - w % window_size
                haze_pad = F.pad(haze, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

                noisy_state = self.sde.noise_state(haze_pad)

                if self.use_cuda:

                    haze_pad = haze_pad.cuda()
                    # clear = clear.cuda()
                    noisy_state = noisy_state.cuda()

                self.sde.set_mu(haze_pad)
                self.net.eval()

                save_dir = os.path.join(self.opt['ckpts'], self.opt['path']['ckpts'], haze_name[0].split('.')[0])

                if type == 'ori' or type == 'base':
                    # _ = self.sde.reverse_sde_original_sample(noisy_state, save_dir=save_dir, save_states=True, r=r)
                    dehaze = self.sde.reverse_sde_ori_val(noisy_state, save_dir=save_dir, save_states=False, r=r)
                elif type == 'imp' or type == 'final':
                    _ = self.sde.reverse_sde_improved_sample(noisy_state, save_dir=save_dir, save_states=False, r=r)
                else:
                    FileNotFoundError("cant found this str")

                _, _, h, w = dehaze.size()
                dehaze = dehaze[:, :, 0: h - mod_pad_h * 1, 0: w - mod_pad_w * 1]
                print(save_dir)
                vutils.save_image(dehaze.data, save_dir + '.png', normalize=False)


    def comparative(self, val_data_loader, save_tag = True, type='com', r = 55, num=500, ori_center=25, imp_center=23):

        logger = self.set_logger(type)

        print(self.opt['path']['checkpoint'])
        # self.net.load_state_dict(torch.load(self.opt['path']['pretrained'])['state_dict'])
        ori_psnr_means, ori_psnr_stds = [], []
        ori_ssim_means, ori_ssim_stds = [], []

        imp_psnr_means, imp_psnr_stds = [], []
        imp_ssim_means, imp_ssim_stds = [], []

        with torch.no_grad():
            ori_psnrs, ori_ssims = [], []
            imp_psnrs, imp_ssims = [], []

            ori_count_psnr, imp_count_psnr = [], []
            ori_count_ssim, imp_count_ssim = [], []

            ori_t0_psnr, ori_t0_ssim = [], []
            ori_center_psnr, ori_center_ssim = [], []
            ori_max_psnr, ori_max_ssim = [], []

            imp_t0_psnr, imp_t0_ssim = [], []
            imp_center_psnr, imp_center_ssim = [], []
            imp_max_psnr, imp_max_ssim = [], []

            type_name = 'n' + str(num) + '_r' + str(r)

            for i, (haze, clear, haze_name) in enumerate(val_data_loader):

                window_size = 4
                mod_pad_h, mod_pad_w = 0, 0
                _, _, h, w = haze.size()
                if h % window_size != 0:
                    mod_pad_h = window_size - h % window_size
                if w % window_size != 0:
                    mod_pad_w = window_size - w % window_size
                haze_pad = F.pad(haze, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

                noisy_state = self.sde.noise_state(haze_pad)

                if self.use_cuda:

                    haze_pad = haze_pad.cuda()
                    noisy_state = noisy_state.cuda()

                self.sde.set_mu(haze_pad)
                self.net.eval()
                # clear = clear.cuda()

                save_dir = os.path.join(self.opt['ckpts'], self.opt['path']['ckpts'])


                ori_imgs = self.sde.reverse_sde_original_sample(noisy_state, save_dir=save_dir, save_states=False, r=r)
                imp_imgs = self.sde.reverse_sde_improved_sample(noisy_state, save_dir=save_dir, save_states=False, r=r)

                ori_psnr, ori_ssim = fun1(ori_imgs, clear, mod_pad_h, mod_pad_w)
                imp_psnr, imp_ssim = fun1(imp_imgs, clear, mod_pad_h, mod_pad_w)

                ori_psnrs.append(ori_psnr)
                ori_ssims.append(ori_ssim)
                imp_psnrs.append(imp_psnr)
                imp_ssims.append(imp_ssim)

                # ori
                ori_psnr_index, ori_psnr_mean, ori_psnr_std = metrics_tools(ori_psnr)
                ori_ssim_index, ori_ssim_mean, ori_ssim_std = metrics_tools(ori_ssim)

                ori_psnr_means.append(ori_psnr_mean)
                ori_psnr_stds.append(ori_psnr_std)
                ori_ssim_means.append(ori_ssim_mean)
                ori_ssim_stds.append(ori_ssim_std)

                # imp
                imp_psnr_index, imp_psnr_mean, imp_psnr_std = metrics_tools(imp_psnr)
                imp_ssim_index, imp_ssim_mean, imp_ssim_std = metrics_tools(imp_ssim)

                imp_psnr_means.append(imp_psnr_mean)
                imp_psnr_stds.append(imp_psnr_std)
                imp_ssim_means.append(imp_ssim_mean)
                imp_ssim_stds.append(imp_ssim_std)

                # ori psnr
                ori_t0_psnr.append(ori_psnr[-1])
                ori_center_psnr.append(ori_psnr[ori_center])
                ori_max_psnr.append(ori_psnr[ori_psnr_index])
                # ori ssim
                ori_t0_ssim.append(ori_ssim[-1])
                ori_center_ssim.append(ori_ssim[ori_center])
                ori_max_ssim.append(ori_ssim[ori_psnr_index])

                # imp psnr
                imp_t0_psnr.append(imp_psnr[-1])
                imp_center_psnr.append(imp_psnr[imp_center])
                imp_max_psnr.append(imp_psnr[imp_psnr_index])

                # imp ssim
                imp_t0_ssim.append(imp_ssim[-1])
                imp_center_ssim.append(imp_ssim[imp_center])
                imp_max_ssim.append(imp_ssim[imp_psnr_index])


                ori_count_psnr.append(ori_psnr_index + 1)
                ori_count_ssim.append(ori_ssim_index + 1)

                imp_count_psnr.append(imp_psnr_index + 1)
                imp_count_ssim.append(imp_ssim_index + 1)

                ori_psnr_dict, ori_ssim_dict={}, {}
                imp_psnr_dict, imp_ssim_dict={}, {}

                for ori_psnr, ori_ssim, imp_psnr, imp_ssim in zip(ori_count_psnr, ori_count_ssim, imp_count_psnr, imp_count_ssim):
                    
                    ori_psnr_dict[ori_psnr]=ori_psnr_dict.get(ori_psnr, 0) + 1 
                    ori_ssim_dict[ori_ssim]=ori_ssim_dict.get(ori_ssim, 0) + 1

                    imp_psnr_dict[imp_psnr]=imp_psnr_dict.get(imp_psnr, 0) + 1
                    imp_ssim_dict[imp_ssim]=imp_ssim_dict.get(imp_ssim, 0) + 1
                
                logger.info('{:<20}, t0: {:.02f}, {:.04f} | t{}: {:.02f}, {:.04f} | max: {:.02f}, {:.04f} | t0_avg: {:.02f}, {:.04f} | t{}_avg: {:.02f}, {:.04f} | max_avg: {:.02f}, {:.04f}'.format(
                    haze_name[0],  ori_t0_psnr[-1], ori_t0_ssim[-1], ori_center, ori_center_psnr[-1], ori_center_ssim[-1], ori_max_psnr[-1], ori_max_ssim[-1],
                    np.mean(ori_t0_psnr), np.mean(ori_t0_ssim), ori_center, np.mean(ori_center_psnr), np.mean(ori_center_ssim), np.mean(ori_max_psnr), np.mean(ori_max_ssim)))
                logger.info('{:<20}, t0: {:.02f}, {:.04f} | t{}: {:.02f}, {:.04f} | max: {:.02f}, {:.04f} | t0_avg: {:.02f}, {:.04f} | t{}_avg: {:.02f}, {:.04f} | max_avg: {:.02f}, {:.04f}'.format(
                    haze_name[0],  imp_t0_psnr[-1], imp_t0_ssim[-1], imp_center, imp_center_psnr[-1], imp_center_ssim[-1], imp_max_psnr[-1], imp_max_ssim[-1],
                    np.mean(imp_t0_psnr), np.mean(imp_t0_ssim), imp_center, np.mean(imp_center_psnr), np.mean(imp_center_ssim), np.mean(imp_max_psnr), np.mean(imp_max_ssim)))
                
                if (i + 1) % 50 == 0:
                    print('{} iamges finished!'.format(i + 1))
                
                if (i + 1) % 1 == 0:
                    logger.info('ori psnr mean: {:.2f}, std: {:.2f} | ori ssim mean: {:.4f}, std: {:.4f}'.format(
                    np.mean(ori_psnr_means), np.mean(ori_psnr_stds), np.mean(ori_ssim_means), np.mean(ori_ssim_stds)
                    ))
                    logger.info('imp psnr mean: {:.2f}, std: {:.2f} | imp ssim mean: {:.4f}, std: {:.4f}'.format(
                        np.mean(imp_psnr_means), np.mean(imp_psnr_stds), np.mean(imp_ssim_means), np.mean(imp_ssim_stds)
                    ))

                    plot_all_point(self.opt['ckpts'], 'PSNR_ori_' + type_name, ori_psnrs, 'PSNR(dB)', r=r, color='g')
                    plot_all_point(self.opt['ckpts'], 'SSIM_ori_' + type_name, ori_ssims, 'ssim', r=r, color='g')

                    plot_all_point(self.opt['ckpts'], 'PSNR_imp_' + type_name, imp_psnrs, 'PSNR(dB)', r=r, color='b')
                    plot_all_point(self.opt['ckpts'], 'SSIM_imp_' + type_name, imp_ssims, 'ssim', r=r, color='b')
                    
                    comparative_plot_all_point(self.opt['ckpts'], 'comparative_psnr_' + type_name, zip(ori_psnrs, imp_psnrs), 'PSNR(dB)', r=r)
                    comparative_plot_all_point(self.opt['ckpts'], 'comparative_ssim_' + type_name, zip(ori_ssims, imp_ssims), 'ssim', r=r)

                    comparative_plot_per_epoch_count(self.opt['ckpts'], 'comparative_psnr_count_' + type_name, 
                                                     ori_psnr_dict, imp_psnr_dict, 'Count', r=r, 
                                                      ori_mean=np.mean(ori_count_psnr), ori_std=np.std(ori_count_psnr),
                                                      imp_mean=np.mean(imp_count_psnr), imp_std=np.std(imp_count_psnr))
                    comparative_plot_per_epoch_count(self.opt['ckpts'], 'comparative_ssim_count_' + type_name,
                                                      ori_ssim_dict, imp_ssim_dict, 'Count', r=r,
                                                      ori_mean=np.mean(ori_count_ssim), ori_std=np.mean(ori_count_ssim),
                                                      imp_mean=np.mean(imp_count_ssim), imp_std=np.mean(imp_count_ssim))

                if i + 1 == num:
                    break

 
    def print_network(self):
        s, n = self.get_network_description(self.net)
        if isinstance(self.net, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.net.__class__.__name__,
                                             self.net.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.net.__class__.__name__)

        logger = logging.getLogger('base')

        logger.info(s)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))


    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n


    def save_network(self, network, network_label, iter_label):

        save_filename = '{}_netG.pth'.format(network_label)
        save_path = os.path.join(self.opt['ckpts'], save_filename)

        if isinstance(network, nn.DataParallel) or isinstance(
            network, DistributedDataParallel
        ):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(
            network, DistributedDataParallel
        ):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith("module."):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)

    def save_training_state(self, best_psnr, state_label, current_iter):
        """Saves training state during training, which will be used for resuming"""
        state = {"current_iter": current_iter, 
                 "best_psnr": best_psnr,
                 "scheduler": self.scheduler.state_dict(), 
                 "optimizer": self.optim.state_dict(),
                 "scaler": self.scaler.state_dict()}

        save_filename = "{}.state".format(state_label)
        save_path = os.path.join(self.opt['ckpts'], save_filename)

        torch.save(state, save_path)

    def resume_training(self, load_path):
        """Resume the optimizers and schedulers for training"""

        resume_state = torch.load(load_path)

        self.optim.load_state_dict(resume_state["optimizer"])
        self.scheduler.load_state_dict(resume_state["scheduler"])
        self.scaler.load_state_dict(resume_state["scaler"])
        return resume_state