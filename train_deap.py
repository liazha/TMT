import os
import torch
import numpy as np
from tqdm import tqdm

from core.dataset_deap import MyDataset, MyDataLoader
from core.dataset_new import MMDataLoader
from core.losses2 import MultimodalLoss, MultimodalDeapLoss
# from core.losses import MultimodalLoss, MultimodalLoss_Mixup, MultimodalLoss_Reg
from core.optimizer import get_optimizer
from core.scheduler import get_scheduler
from core.utils import AverageMeter, setup_seed, mixup_data_deap
from tensorboardX import SummaryWriter
#from models.MyModel_difusion import MyMultimodal_Integrally
from models.MyModel_deap import MyMultimodal_Integrally
import argparse
from core.metric import MetricsTop
from torch.autograd import Variable
from core.metric import cal_acc5
import time
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

parser = argparse.ArgumentParser()
# dataset
parser.add_argument("--datasetName", type=str, default="deap", required=False)
# 数据集特征是通过MMSA-FET toolkit提取的，保存到了Processed文件夹下
parser.add_argument("--dataPath", type=str, default=r"G:\code\MyEmotion\data\deap_data2.pkl", required=False)
parser.add_argument("--use_bert", type=bool, default=False, required=False)
parser.add_argument("--need_data_aligned", type=bool, default=True, required=False)
parser.add_argument("--need_truncated", type=bool, default=True, required=False)
parser.add_argument("--data_missing", type=bool, default=False, required=False)
parser.add_argument("--seq_lens", type=tuple, default=[30,30,30], required=False)
#parser.add_argument("--batch_size", type=int, default=16, required=False)
parser.add_argument("--batch_size", type=int, default=32, required=False)
parser.add_argument("--num_workers", type=int, default=0, required=False)
parser.add_argument("--train_mode", type=str, default='classification', required=False)  # regression   classification
parser.add_argument("--use_label", type=str, default='valence_labels', required=False) # valence_labels arousal_labels dominance_labels
args = parser.parse_args()

project_name = "train5_26" 
batchsize = args.batch_size
max_acc = 0
Has0_acc_2=0
Non0_acc_2=0
Has0_F1_score=0
F1_score_3=0
Acc_3=0

n_epochs=200
# learning_rate = 1e-4
learning_rate = 1e-4

Mult_acc_2=0
Mult_acc_3=0
Mult_acc_5=0
F1_score_5=0
F1_score=0
MAE=0.99
Corr=0

def main():
    log_path = os.path.join(os.path.join("train/log", project_name))
    print("log_path :", log_path)

    save_path = 'checkpoint'
    print("model_save_path :", save_path)
    model = MyMultimodal_Integrally(num_classs = 1,
                        visual_seq_len=50,
                        eeg_seq_len=50,
                        batchsize=64,
                        trans_depth=2,
                        cross_depth=4
                        ).to(device)

    dataLoader = MyDataLoader(args)

    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    scheduler_warmup = get_scheduler(optimizer, n_epochs)
    #0.001 0.01 0.1 1.0 10 100
    #alpha
    # 0.001 46.83
    #  0.01 48.14
    #   0.1 46.60
    #     1  46.82
    #    10  46.39  
    
    #beta
    # 0.001 47.70
    # 0.01 48.14
    # 0.1  47.48
    #1.0  46.82
    #10 46.78

    # if args.train_mode == 'classification':
    #     loss_fn = MultimodalLoss(alpha=0.01, beta=0.01, delta=1, batch_size=batchsize, device=device)
    #     loss_fn_mixup = MultimodalLoss_Mixup(alpha=0.01, beta=0.01, delta=1, batch_size=batchsize, device=device)
    # else:
    #     loss_fn = MultimodalLoss_Reg(alpha=0.01, beta=0.01, delta=1, batch_size=batchsize, device=device)

    loss_fn = MultimodalDeapLoss(alpha=0.01, beta=1, delta=10, device=device)
    # if args.train_mode == 'classification':
    #     loss_fn = MultimodalLoss(alpha=0.01, beta=0.01, delta=1, batch_size=batchsize, device=device)
    #     loss_fn = MultimodalLoss(alpha=0.01, beta=0.01, delta=1, batch_size=batchsize, device=device)
    #     loss_fn_mixup = MultimodalLoss_Mixup(alpha=0.01, beta=0.01, delta=1, batch_size=batchsize, device=device)
    # else:
    #     loss_fn = MultimodalLoss_Reg(alpha=0.01, beta=0.01, delta=1, batch_size=batchsize, device=device) 

    writer = SummaryWriter(logdir=log_path)

    for epoch in range(1, n_epochs+1):
        train(model, dataLoader['train'], optimizer, loss_fn, epoch, writer)
        evaluate(model, dataLoader['test'], optimizer, loss_fn, epoch, writer, save_path)
        scheduler_warmup.step()
    writer.close()
    torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))


def train(model, train_loader, optimizer, loss_fn, epoch, writer):
    model.train()
    for batch, sample in enumerate(train_loader):
        x_visual = sample['visual'].to(device)
        x_eeg = sample['eeg'].to(device)
        if args.train_mode == 'classification':
            label = sample['labels'].long().squeeze().to(device)
        else:
            label = sample['labels'].squeeze().to(device)

        x_visual, x_eeg, lebel_a, lebel_b, lam = mixup_data_deap(x_visual, x_eeg, label, 1.0, device=device)
        x_visual, x_eeg, lebel_a, lebel_b = map(Variable, (x_visual, x_eeg, lebel_a, lebel_b))

        model.zero_grad()
        x_invariant, x_specific_v, x_specific_e, cls_output, x_visual, x_eeg = model(x_visual, x_eeg)

        loss, orth_loss, sim_loss, cls_loss = loss_fn(
            x_invariant, x_specific_v, x_specific_e,
            x_visual, x_eeg,
            cls_output, label, epoch) # cls_output (64, 32) label (64)

        print(f'epoch: {epoch} | batch: {batch} | loss: {loss} | orth_loss: {orth_loss} | sim_loss: {sim_loss} | cls_loss: {cls_loss} ')
        loss.backward()
        optimizer.step()
        # optimizer.zero_grad()


def evaluate(model, eval_loader, optimizer, loss_fn, epoch, writer, save_path):
    global max_acc,Has0_acc_2,Has0_F1_score,Non0_acc_2,Acc_3,Mult_acc_2,Mult_acc_3,Mult_acc_5,F1_score,MAE,Corr,F1_score_3,F1_score_5
    eval_pbar = tqdm(enumerate(eval_loader))
    losses = AverageMeter()
    losses_cls = AverageMeter()
    losses_specific = AverageMeter()
    losses_invariant = AverageMeter()
    acc_specific = AverageMeter()
    acc_invariant = AverageMeter()
    acc_cls = AverageMeter()
    y_pred = []
    y_true = []

    model.eval()
    
    for cur_iter, sample in eval_pbar:
        x_visual = sample['visual']
        x_eeg = sample['eeg']
        label = sample['labels']

        if args.train_mode == 'classification':
            label = sample['labels'].long().squeeze().to(device)
        else:
            label = sample['labels'].squeeze().to(device)
        # # label = torch.where(label>=0, torch.tensor(1.), label)
        # # label = torch.where(label<0, torch.tensor(0.), label)
        # # label = torch.where(label==0, torch.tensor(2.), label)
        # # label = label.squeeze().long().to(device)

        with torch.no_grad():
            x_invariant, x_specific_v, x_specific_e, cls_output, x_visual, x_eeg = model(x_visual.to(device), x_eeg.to(device))
            #D0_visual_op, D0_audio_op, D0_text_op, D2_visual_op, D2_audio_op, D2_text_op, cls_output,a,b,c = model(x_vision.to(device), x_audio.to(device), x_text.to(device))

        y_pred.append(cls_output.cpu())
        y_true.append(label.cpu())

    # pred, true = torch.cat(y_pred).squeeze().numpy(), torch.cat(y_true).squeeze().numpy()
    # mae = np.mean(np.absolute(pred - true))
    # print(f'mae: {mae}')
    #
    # eval_results = MetricsTop(args.train_mode).getMetics('DEAP')(pred, true)

    test_preds1 = torch.cat(y_pred).view(-1).cpu().detach().numpy()
    test_truth = torch.cat(y_true).view(-1).cpu().detach().numpy()

    for i, j in enumerate(test_preds1):
        if -1 < j < 0:
            test_preds1[i] = -1
        if 0 < j < 1:
            test_preds1[i] = 1
    test_preds1 = np.clip(test_preds1, a_min=-1., a_max=2.)
    test_preds = np.around(test_preds1)

    # non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])
    # test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    # test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a5 = np.sum(np.round(test_preds) == np.round(test_truth)) / float(len(test_truth))

    _, _, f1, _ = precision_recall_fscore_support(test_preds, test_truth, average='weighted')
    print("-" * 50)
    print("MAE: ", mae)
    print("Correlation Coefficient: ", corr)
    print("mult_acc4: ", mult_a5)
    print('f1_score:', f1)
    print("-" * 50)


if __name__ == '__main__':
    setup_seed(12345)
    start_timne = time.time()
    main()
    end_time = time.time()
    print(f'执行时间：{end_time - start_timne}秒')
