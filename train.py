### This code is largely borrowed from pix2pixHD pytorch implementation
### https://github.com/NVIDIA/pix2pixHD

import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
import fractions
from models.models import create_model
import torch.nn as nn
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
import util.util as util
from util.visualizer import Visualizer
import torch.nn.functional as F


def lcm(a, b): return abs(a * b) / fractions.gcd(a, b) if a and b else 0


opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

train_loader = CreateDataLoader(opt, 'train')
train_dataset = train_loader.load_data()
dataset_size = len(train_loader)
print('#training patches = %d' % dataset_size)

val_loader = CreateDataLoader(opt, 'val')
val_dataset = val_loader.load_data()
val_size = len(val_loader)
print('#validation patches = %d' % val_size)
sf = torch.nn.Softmax(dim=1)

ler = opt.lr
print('weight of class:')
print(train_loader.weight)
weights = train_loader.weight
weights[0] = weights.max() / 2
# CE = nn.CrossEntropyLoss(weight=weights.cuda())
CE = nn.CrossEntropyLoss()

G = create_model(opt)
G.cuda()

print(G)
optimizer_G = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.001)
visualizer = Visualizer(opt)

true_label = torch.ones((4, 1)).cuda().long()
false_label = torch.zeros((4, 1)).cuda().long()
start_epoch, epoch_iter = 1, 0
total_steps = (start_epoch - 1) * dataset_size + epoch_iter
display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq
best_val = 0

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(train_dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ############## Forward Pass ######################
        # print(data['label'].shape)
        G.train()
        generated = G(Variable(data['img'].cuda()))
        loss_CE = CE(generated, Variable(data['label'].cuda()))
        loss_dict = dict(zip(['CE'], [loss_CE]))

        ############### Backward Pass ####################
        # update generator weights
        optimizer_G.zero_grad()
        loss = loss_CE
        loss.backward()
        optimizer_G.step()

        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visualizer.plot_current_errors(errors, total_steps)

        ### display output images
        if save_fake:
            pred_src_opt = F.softmax(generated, dim=1)
            opt_tensor = pred_src_opt.max(1)[1][0]
            pred = torch.unsqueeze(opt_tensor[:, :, 15], 0)
            visuals = OrderedDict([('input_image', util.tensor2im(torch.unsqueeze(data['img'][0, 0, :, :, 15], 0))),
                                   ('segmented_img', util.tensor2label(pred, opt.cls_num)),
                                   ('real_image',
                                    util.tensor2label(torch.unsqueeze(data['label'][0, :, :, 15], 0), opt.cls_num))])
            visualizer.display_current_results(visuals, epoch, total_steps)

        ## save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            # model.module.save('latest')
            torch.save(G.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'latest.pth'))
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break

    # end of epoch 
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    G.eval()
    val_dice = []
    with torch.no_grad():
        for val_i, val_data in enumerate(val_dataset):
            pred_val = G(Variable(val_data['img'].cuda()))
            pred_val = sf(pred_val).detach().cpu().numpy()
            pred_val = np.argmax(pred_val, axis=1).astype(np.float)
            gt_val = val_data['label'].numpy()
            tmp_dice = 0
            count = 0
            for val_tag in range(1, opt.cls_num):
                if (pred_val == val_tag).sum() > 0 or (gt_val == val_tag).sum() > 0:
                    tmp_dice += util.cal_dice(pred_val, gt_val, val_tag)
                    count += 1
            tmp_dice /= count
            val_dice.append(tmp_dice)
    G.train()
    if best_val < np.mean(val_dice):
        best_val = np.mean(val_dice)
        torch.save(G.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'best.pth'))
    errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
    errors.update({'val_dice': np.mean(val_dice)})
    t = (time.time() - iter_start_time) / opt.print_freq
    visualizer.print_current_errors(epoch, epoch_iter, errors, t)
    # visualizer.plot_current_errors(errors, total_steps)

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        torch.save(G.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'ckpt%d%d.pth' % (epoch, total_steps)))
        np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

    ## linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        ler -= (opt.lr) / (opt.niter_decay)
        for param_group in optimizer_G.param_groups:
            param_group['lr'] = ler
            print('change lr to ')
            print(param_group['lr'])
