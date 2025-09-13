import itertools
import tabnanny
import torch
import torch.utils.data as Torchdata
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm
from scipy import io
import os
from tools import *
from loss import *
import time
import random
from model import VSSM
import torch.nn.functional as F


# Parameters setting
DATASET = 'PaviaU' #Salinas、PaviaU、Houston2013

if DATASET == 'Salinas':
    PATCH_SIZE = 29
    LEARNING_RATE = 0.001
    numComponents = 30
    batch_size_per_class = 1
    thres_bvsb = 0.8
if DATASET == 'PaviaU':
    PATCH_SIZE = 25
    LEARNING_RATE = 0.001
    numComponents = 10
    batch_size_per_class = 2
    thres_bvsb = 0.4
if DATASET == 'Houston2013':
    PATCH_SIZE = 15
    LEARNING_RATE = 0.001
    numComponents = 15
    batch_size_per_class = 1
    thres_bvsb = 0.5

SAMPLE_ALREADY = True
N_RUNS = 10
SAMPLE_SIZE = 5
FLIP_ARGUMENT = True
ROTATED_ARGUMENT = True
ITER_NUM = 1000
SAMPLING_MODE = 'fixed_withone'
FOLDER = './Datasets/'
GPU = 0
RESULT_DIR = f'./Results/{DATASET}/'
training_time = []

#############################Data###################################
img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, numComponents, FOLDER)
N_CLASSES = len(LABEL_VALUES) - 1
N_BANDS = img.shape[-1]

def data_init(run):
    if SAMPLE_ALREADY:
        train_gt, test_gt = get_sample(DATASET, SAMPLE_SIZE, run, 1000)
    else:
        train_gt, test_gt, _, _ = sample_gt(gt, SAMPLE_SIZE, mode=SAMPLING_MODE)
        save_sample(train_gt, test_gt, DATASET, SAMPLE_SIZE, run, SAMPLE_SIZE, 1000)
    task_train_dataset = HyperX(img, train_gt, DATASET, PATCH_SIZE, FLIP_ARGUMENT, ROTATED_ARGUMENT)
    task_test_dataset = HyperX(img, train_gt, DATASET, PATCH_SIZE, FLIP_ARGUMENT, ROTATED_ARGUMENT) #注意，这时候的测试集不包括val集！
    print("{} samples selected (over {})".format(np.count_nonzero(train_gt),np.count_nonzero(gt)))
    return train_gt,task_train_dataset,task_test_dataset

focal_loss = FocalLoss(alpha=None, gamma=2.0)
scaler = torch.cuda.amp.GradScaler()
#############################Set seed##################################
def set_random_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

#############################Train##################################
def train_run(run):
    print("Running an experiment with run {}/{}".format(run + 1, N_RUNS))
    display_iter = 10
    losses = np.zeros(ITER_NUM+1)
    mean_losses = np.zeros(ITER_NUM+1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hmamba = VSSM(N_CLASSES, patch_size=[PATCH_SIZE,PATCH_SIZE], PCA_num=numComponents, thres=thres_bvsb)
    hmamba.to(device)
    optimizer1 = torch.optim.Adam(hmamba.parameters(), lr=LEARNING_RATE, weight_decay=0.0005)
    scheduler1 = StepLR(optimizer1, step_size=ITER_NUM // 2, gamma=0.1)

    train_gt,task_train_dataset,task_test_dataset = data_init(run)
    start_time = time.time()
    
    for iter_ in tqdm(range(1, ITER_NUM + 1), desc='Training the network'):

        task_test_gt, rest_gt, _, _ = sample_gt(train_gt, 1, mode='fixed_withone')
        task_train_gt, rest_gt, task_index, special_list = sample_gt(train_gt, batch_size_per_class, mode='fixed_withone')
        task_test_dataset.resetGt(task_test_gt)
        task_test_loader = Torchdata.DataLoader(task_test_dataset, batch_size=N_CLASSES, shuffle=False)
        task_train_dataset.resetGt(task_train_gt)
        task_batch_size = np.count_nonzero(task_train_gt)
        task_train_loader = Torchdata.DataLoader(task_train_dataset, batch_size=task_batch_size, shuffle=False)

        samples, sample_labels = next(iter(task_train_loader))
        query_batches, query_batch_labels = next(iter(task_test_loader))
        hmamba.train()

        avg_loss_list = []
        result_list = []
        batches_list = []

        with torch.cuda.amp.autocast():
            result, loss_moe = hmamba(query_batches.cuda(), samples.cuda(), batch_size_per_class)
            if loss_moe is None:
                loss_moe = 0
            loss = cal_loss(focal_loss, result, query_batch_labels.cuda()) + loss_moe

        optimizer1.zero_grad()
        scaler.scale(loss).backward() 
        scaler.step(optimizer1) 
        scaler.update()
        scheduler1.step()

        losses[iter_] = loss

        mean_losses[iter_] = np.mean(losses[max(0, iter_ - 10):iter_ + 1])
        if display_iter and iter_ % display_iter == 0:
            string = 'Train (ITER_NUM {}/{})\tLoss: {:.6f}'
            string = string.format(iter_, ITER_NUM, mean_losses[iter_])
            tqdm.write(string)

        with torch.cuda.device(GPU):
            if iter_ in [1000, 1500, 2000]:
                model_module1_dir = RESULT_DIR + 'Checkpoints/' + '/' + task_train_loader.dataset.name + '/' + str(SAMPLE_SIZE) + '_' + str(iter_) + '/'
                if not os.path.isdir(model_module1_dir):
                    os.makedirs(model_module1_dir)
                model_module1_file = model_module1_dir + 'sample{}_run{}.pth'.format(SAMPLE_SIZE,run)
                torch.save(hmamba.state_dict(), model_module1_file)
        
    end_time = time.time()
    training_time.append(end_time - start_time)

    loss_dir = RESULT_DIR + 'Losses/' + DATASET + '/' + str(SAMPLE_SIZE) + '_' + str(ITER_NUM) + '/'
    if not os.path.isdir(loss_dir):
        os.makedirs(loss_dir)
    loss_file = loss_dir + '/' + 'sample' + str(SAMPLE_SIZE) + '_run' + str(run) + '_dim' + '.mat'
    io.savemat(loss_file, {'losses':losses})
####################################################################
def main():
    for run in range(N_RUNS):
        train_run(run)
    ACCU_DIR = RESULT_DIR + 'Accu_File/' + DATASET + '/' + str(SAMPLE_SIZE) + '_' + str(ITER_NUM) + '/'
    if not os.path.isdir(ACCU_DIR):
        os.makedirs(ACCU_DIR)
    time_path = ACCU_DIR+str(DATASET)+'_'+str(SAMPLE_SIZE)+'.txt'
    f = open(time_path, 'a')
    sentence0 = 'avergare training time:' + str(np.mean(training_time)) + '\n'
    f.write(sentence0)
    sentence1 = 'training time for each iteration are:' + str(training_time) + '\n'
    f.write(sentence1)
    f.close()

if __name__ == '__main__':
    main()     


