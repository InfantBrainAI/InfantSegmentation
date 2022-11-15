import argparse
import os
from util import util
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--name', type=str, default='ckpt',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='models',
                                 help='models are saved here')
        self.parser.add_argument('--model', type=str, default='3DUNet', help='which model to use')
        self.parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
        self.parser.add_argument('--dataroot', type=str, default='')
        self.parser.add_argument('--label_nc', type=int, default=0, help='# of input label channels')
        self.parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=8, help='# of output image channels')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--cls_num', default=8, type=int, help='# threads for loading data')

        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')

        # for displays
        self.parser.add_argument('--display_winsize', type=int, default=512, help='display window size')
        self.parser.add_argument('--tf_log', action='store_true',
                                 help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for data:
        self.parser.add_argument('--extension', type=str, default='.nii', help='initial learning rate for adam')
        self.parser.add_argument('--normalize', type=bool, default=True, help='initial learning rate for adam')
        self.parser.add_argument('--remapping', type=bool, default=False, help='initial learning rate for adam')
        self.parser.add_argument('--remap_csv', type=str, default='/media/win_ssd/Ubuntu/Subcortical_Seg/Subcortical_mapping.csv',
                                 help='initial learning rate for adam')
        self.parser.add_argument('--norm_perc', type=float, default=99.5, help='initial learning rate for adam')
        self.parser.add_argument('--patch_size', type=tuple, default=(80, 80, 80),
                                 help='initial learning rate for adam')
        self.parser.add_argument('--patch_stride', type=tuple, default=(64, 64, 64),
                                 help='initial learning rate for adam')
        self.parser.add_argument('--remove_bg', type=bool, default=True, help='initial learning rate for adam')
        self.parser.add_argument('--aug', type=bool, default=True, help='initial learning rate for adam')
        self.parser.add_argument('--aug_prob', type=float, default=0.5, help='probability of augmentation = 1-p')

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args(args=[])
        self.opt.isTrain = self.isTrain  # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0 and torch.cuda.is_available():
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')


        # util.mkdirs(expr_dir)
        if save:
            # save to the disk
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            if not os.path.exists(expr_dir):
                os.makedirs(expr_dir)
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
