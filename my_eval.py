import argparse

import torch

from core.dataset_deap import MyDataLoader
from models.MyModel_deap import MyMultimodal_Integrally


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

parser = argparse.ArgumentParser()
# dataset
parser.add_argument("--datasetName", type=str, default="deap", required=False)
# 数据集特征是通过MMSA-FET toolkit提取的，保存到了Processed文件夹下
parser.add_argument("--dataPath", type=str, default=r"G:\code\MyEmotion\data\deap_data3.pkl", required=False)
parser.add_argument("--use_bert", type=bool, default=False, required=False)
parser.add_argument("--need_data_aligned", type=bool, default=True, required=False)
parser.add_argument("--need_truncated", type=bool, default=True, required=False)
parser.add_argument("--data_missing", type=bool, default=False, required=False)
parser.add_argument("--seq_lens", type=tuple, default=[30,30,30], required=False)
#parser.add_argument("--batch_size", type=int, default=16, required=False)
parser.add_argument("--batch_size", type=int, default=32, required=False)
parser.add_argument("--num_workers", type=int, default=0, required=False)
parser.add_argument("--train_mode", type=str, default='regression', required=False)  # regression   classification
parser.add_argument("--use_label", type=str, default='valence_labels', required=False) # valence_labels arousal_labels dominance_labels
args = parser.parse_args()

if __name__ == '__main__':
    model = MyMultimodal_Integrally(num_classs=1,
                                    visual_seq_len=50,
                                    eeg_seq_len=50,
                                    batchsize=64,
                                    trans_depth=2,
                                    cross_depth=4
                                    ).to(device)

    model.load_state_dict(torch.load(r'G:\code\TMT\checkpoint\deap_model.pth'))
    model.eval()

    y_pred = []
    y_true = []

    model.eval()
    dataLoader = MyDataLoader(args)

    for cur_iter, sample in enumerate(dataLoader['test']):
        x_visual = sample['visual']
        x_eeg = sample['eeg']
        label = sample['labels']

        if args.train_mode == 'classification':
            label = label.long().squeeze().to(device)
        else:
            label = label.squeeze().to(device)

        with torch.no_grad():
            x_invariant, x_specific_v, x_specific_e, cls_output, x_visual, x_eeg = model(x_visual.to(device),
                                                                                         x_eeg.to(device))

        y_pred.append(cls_output.cpu())
        y_true.append(label.cpu())

    test_preds = torch.cat(y_pred).view(-1).cpu().detach().numpy()
    test_truth = torch.cat(y_true).view(-1).cpu().detach().numpy()

    for i in range(len(test_preds)):
        print(f'pred: {test_preds[i]}  truth: {test_truth[i]}')