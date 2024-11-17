import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging


logger = logging.getLogger('MSA')

class MyDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATA_MAP = {
            'deap': self.__init_deap,
        }
        DATA_MAP[args.datasetName]()

    def __init_deap(self):
        with open(self.args.dataPath, 'rb') as f:
            data = pickle.load(f)
        self.visual = data[self.mode]['visual'].astype(np.float32)
        # print(self.vision.shape)
        self.eeg = data[self.mode]['eeg'].astype(np.float32)
        self.labels = data[self.mode][self.args.use_label].astype(np.float32)

        logger.info(f"{self.mode} , datasetName: {self.args.datasetName}, samples: {self.labels.shape}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {
            'visual': torch.Tensor(self.visual[idx]).cpu().detach(),
            'eeg': torch.Tensor(self.eeg[idx]).cpu().detach(),
            'labels': torch.tensor(self.labels[idx]).cpu().detach()
        }

        return sample

    def get_seq_len(self):
        return (self.visual.shape[1], self.eeg.shape[1])



def MyDataLoader(args):
    datasets = {
        'train': MyDataset(args, mode='train'),
        'val': MyDataset(args, mode='val'),
        'test': MyDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args.seq_lens = datasets['train'].get_seq_len()

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }

    return dataLoader
