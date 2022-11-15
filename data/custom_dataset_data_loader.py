### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
### This script is modified based on the pix2pixHD official implementation (see license above)
### https://github.com/NVIDIA/pix2pixHD
import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt,phase):
    dataset = None
    from data.aligned_dataset import RAMDataset
    dataset = RAMDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt,phase)
    return dataset, dataset.weight

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt,phase):
        BaseDataLoader.initialize(self, opt)
        self.dataset, self.weight = CreateDataset(opt,phase)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.nThreads), pin_memory=True)

    def load_data(self):
        return self.dataloader

    def weight(self):
        return self.weight

    def __len__(self):
        return len(self.dataset)
