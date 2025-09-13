import torch
import torch.utils.data as Torchdata
import numpy as np
from tqdm import tqdm
from tools import *
import scipy.io as sio
import time
from model import VSSM
import os

# Parameters setting
DATASET = 'PaviaU'  #Salinas、PaviaU、Houston2013
if DATASET == 'Salinas':
    PATCH_SIZE = 29
    numComponents = 30
    thres_bvsb = 0.8
if DATASET == 'PaviaU':
    PATCH_SIZE = 25
    numComponents = 10
    thres_bvsb = 0.4
if DATASET == 'Houston2013':
    PATCH_SIZE = 15
    numComponents = 15
    thres_bvsb = 0.5

N_RUNS = 10
SAMPLE_SIZE = 5
FOLDER = './Datasets/'
PRE_ALL = False
GPU = 0
SAVE_ITER_NUM = 1000
ITER_NUM = 1000
BATCH_SIZE_PER_CLASS =  SAMPLE_SIZE // 2
DRAW_OR_NOT = True
RESULT_DIR = f'./Results/{DATASET}/'
ACCU_DIR = f'./Results/{DATASET}/Accu_File/{DATASET}/{SAMPLE_SIZE}_{ITER_NUM}/'
CHECKPOINT_Module1 = RESULT_DIR + 'Checkpoints/' + DATASET + '/' + str(SAMPLE_SIZE) + '_' + str(SAVE_ITER_NUM) + '/'
testing_time = []
query_batch_size = 128
##############################Data##################################
img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, numComponents, FOLDER)
N_CLASSES = len(LABEL_VALUES) - 1
N_BANDS = img.shape[-1]
def data_init(run):
    train_gt, test_gt = get_sample(DATASET, SAMPLE_SIZE, run, 1000)
    task_index = []
    special_list = []
    for c in np.unique(train_gt):
        if c == 0:
            continue
        X = np.count_nonzero(train_gt == c)
        if X < SAMPLE_SIZE:
            task_index.append(c)
            special_list.append(X)
    if PRE_ALL:
        test_gt = np.ones_like(test_gt)
    print("{} samples selected (over {})".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
    train_dataset = HyperX(img, train_gt, DATASET, PATCH_SIZE, False, False)
    test_dataset = HyperX(img, test_gt, DATASET, PATCH_SIZE, False, False)
    return train_dataset, test_dataset, task_index, special_list, np.count_nonzero(train_gt)
##############################Test##################################
            
def test_run(run):
    print("Running an experiment with run {}/{}".format(run + 1, N_RUNS))
    oa=[]

    train_dataset, test_dataset ,task_index,special_list,loader_size = data_init(run)
    train_loader = Torchdata.DataLoader(train_dataset, batch_size=loader_size, shuffle=False)
    test_loader = Torchdata.DataLoader(test_dataset, batch_size=query_batch_size, shuffle=False)
    train_loader_iter = iter(train_loader)  
    tr_data, tr_labels = next(train_loader_iter)  

    tr_data = tr_data.cuda(GPU)

    hmamba = VSSM(N_CLASSES,[PATCH_SIZE,PATCH_SIZE],PCA_num=numComponents)

    if CHECKPOINT_Module1 is not None:
        Module_file1 = CHECKPOINT_Module1 + 'sample{}_run{}.pth'.format(SAMPLE_SIZE, run)
        with torch.cuda.device(GPU):
            checkpoint = torch.load(Module_file1)
            hmamba.load_state_dict(checkpoint)
    else:
        raise ('No Checkpoints for hmamba Net')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hmamba.to(device)
    
    hmamba.eval()
    test_labels = []
    pre_labels = []
    pad_pre_gt = np.zeros_like(test_dataset.label)
    pad_test_indices = test_dataset.indices
    start_time = time.time()
    query_bs = query_batch_size

    for batch_idx, (te_data, te_labels) in tqdm(enumerate(test_loader),total=len(test_loader)):

        if te_data.size(0) < query_bs:
            query_bs = te_data.size(0)
        with torch.no_grad():
            te_data, te_labels = te_data.cuda(GPU), te_labels.cuda(GPU)
            outputs, _ = hmamba(te_data, tr_data, SAMPLE_SIZE)
            outputs = outputs[0]
            pred_list = []
            for i in range(query_bs):
                max_value, max_index = torch.max(outputs[i:i+1], 1)
                pred_list.append(max_index)
            labels_pred = torch.tensor(pred_list)

            pre_labels.extend(labels_pred.cpu().numpy().tolist())
            test_labels.extend(te_labels.cpu().numpy().tolist())
    end_time = time.time()
    for i in range(len(pad_test_indices)):
        pad_pre_gt[pad_test_indices[i]] = pre_labels[i] + 1
    p = PATCH_SIZE // 2
    pad_pre_gt = pad_pre_gt[p:-p, p:-p]
    if not PRE_ALL:
        accuracy, total = 0, 0
        for iten_i in range(len(pre_labels)):
            accuracy += test_labels[iten_i] == pre_labels[iten_i]
            total += 1
        rate = accuracy / total
        oa.append(rate)
        print('Accuracy:', rate)
    else:
        mask = np.zeros_like(gt)
        mask[np.where(gt != 0)] = 1
        pre_gt_label = pad_pre_gt * mask
        gt_label_f = gt[np.where(gt != 0)].flatten()
        pre_gt_label_f = pre_gt_label[np.where(pre_gt_label != 0)].flatten()
        accuracy = np.zeros_like(gt_label_f)
        accuracy[np.where(gt_label_f == pre_gt_label_f)] = 1
        rate = np.sum(accuracy) / gt_label_f.size
        oa.append(rate)
        print('Accuracy:', rate)

    testing_time.append(end_time - start_time)
    print('Testing Time:', end_time - start_time)

    #save sores
    results = dict()
    results['OA'] = rate
    if PRE_ALL:
        results['pre_all'] = np.asarray(pad_pre_gt,dtype='uint8')
        results['pre_gt'] = np.asarray(pre_gt_label, dtype='uint8')
    else:
        results['pre_gt'] = np.asarray(pad_pre_gt, dtype='uint8')
    results['test_labels'] = test_labels
    results['pre_labels'] = pre_labels
    save_result(results, ACCU_DIR, SAMPLE_SIZE, run)
####################################################################
def main():
    for run in range(N_RUNS):
        test_run(run)
        if DRAW_OR_NOT:
            gt = get_gt(DATASET, FOLDER)
            gt = gt.tolist()
            if PRE_ALL:
                result = sio.loadmat(ACCU_DIR + 'sample' + str(SAMPLE_SIZE) + '_run' + str(run) + '.mat')['pre_all']
            else:
                result = sio.loadmat(ACCU_DIR + 'sample' + str(SAMPLE_SIZE) + '_run' + str(run) + '.mat')['pre_gt']
            result = result.tolist()
            pic_dir = RESULT_DIR + 'Pic/' + DATASET + '/'+ str(SAMPLE_SIZE) + '_' + str(ITER_NUM) + '/'
            if not os.path.isdir(pic_dir):
                os.makedirs(pic_dir)
            drawresult(result, DATASET, str(SAMPLE_SIZE), str(ITER_NUM), gt, pic_dir, run)
    matrix(ACCU_DIR, DATASET, SAMPLE_SIZE, ITER_NUM, N_RUNS, testing_time)
####################################################################
if __name__ == '__main__':
    main()     

