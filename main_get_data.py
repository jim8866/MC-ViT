import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import pandas as pd

import os, time
import datetime
import sys
#import cv2
#srcDIR = "/home/chaoqiang/user/imagenet3D2D/GvcnnFasterSlices_feature_multi"#F:\20211115_new_test\GvcnnFasterSlices_feature_multi
#srcDIR = r"F:\20211115_new_test\GvcnnFasterSlices_feature_multi"
#DATADIR = '/home/chaoqiang/user/imagenet3D2D/ADNI_3Ddata/'

#/home/hufe/code/20211115_batch/GvcnnFasterSlices_feature_multi
#DATADIR = r"G:/BaiduNetdiskDownload/ADNI_3DDATA/ADNI_3Ddata/"
#DATADIR = '/data/ADNI_3Ddata'
#DATADIR = "/run/media/liyuan/38ca6a92-55f7-482c-ba60-6ac01c5177d5/liyuan/MedLab/ADNI_3Ddata/ADNI_3Ddata"

#DATADIR = r"G:/BaiduNetdiskDownload/ADNI_3DDATA/ADNI_3Ddata/"
#DATADIR = r"/home/hufe/ADNI_data3/NewFolder"
#DATADIR = '/data/ADNI_3Ddata'
DATADIR = r"./data"
from lib import ADNI2_data_subset_new as adni
from lib import cnn_utils
#from lib import extract
from lib.extract import extract_patch_50 as extract
#from lib import train_singleCNN
import time
#import ADNI2_data_subset_new as adni
import hdf5storage
from lib import test_singleCNN



np.set_printoptions(threshold=np.inf)

dataset = 'adni_gvcnn_train2'

groups = ['cn_ad', 'cn_mci', 'mci_ad']
C0ids  = [0, 0, 2]
C1ids  = [3, 2, 3]

groupid = 0

randomstate = 0
foldIDS = 1
foldIDS = 1
patch_size = 50#40#50#45#W40#50
group     = groups[groupid]
C0ID      = C0ids[groupid]
C1ID      = C1ids[groupid]


REG_thick_MAT =  './data/demo.csv'


slicesindex = 1
direction = 0
valid_ratio = 0.2

params = dict()
params['n_splits'] = 1
params['dropout'] = 0.5
params['model'] = "Conv4_FC3"
#params['model'] = "resnet18"
#params['model'] = "densenet"

params['optimizer'] = 'Adam'   #optimizer
params['weight_decay'] = 1e-3##5e-3
params['gpu'] = 0
params['batch_size'] = 32#16#32#256#32#256#128#32
params['learning_rate'] = 5e-5#2e-5
params['DATADIR'] = DATADIR
params['epoch'] = 80 #100
params['gpu'] = True#beginning_epoch

params['beginning_epoch'] = 0
params['patience']=10
params['tolerance']=0.0
params['classes'] = 2
params['cross_validation'] = 5#五折交叉验证


stride_size = 0

print("time:", time.localtime(time.time()))
print("time:", time.localtime(time.time()).tm_year)
time_now = time.time()
dir = str(time.localtime(time.time()).tm_year) + str(time.localtime(time.time()).tm_mon) + str(
    time.localtime(time.time()).tm_mday) + str(time.localtime(time.time()).tm_hour) + str(
    time.localtime(time.time()).tm_min)
print("dir:", dir)

"""
Label, SubjID, SubjIDIndex, thick_select, Age, Gender, thick_base, Age_base, Gender_base, test_filepaths = \
    adni.adni_get_subset_3D2D_slices(REG_thick_MAT, C0ID, C1ID, DATADIR, slicesindex, training=1, direction=direction)
"""


from lib import util

arc_fold = "./result_3d_cnn/result"

for foldIDS in np.arange(1):
    print("----------------------------------------------------------------------------")
    print("foldIDS-%s begin"%(str(foldIDS)))
    #arc_fold = os.path.join(arc_fold, "fold-" + str(foldIDS) + "/models/model_best.pth.tar")
    print("acc_fold:", arc_fold)
    params['output_dir'] = arc_fold
    #model = util.init_model(params["model"], gpu=params["gpu"], dropout=params["dropout"])


    Label, SubjID, SubjIDIndex, thick_select, Age, Gender, thick_base, Age_base, Gender_base, filepaths, waveid_s, SubjID_s, subjtype_s = \
        adni.adni_get_subset_3D2D_slices_for_data(REG_thick_MAT, C0ID, C1ID, DATADIR, slicesindex=100, training=0)
    print("=========================")
    print("Label.shape:", Label.shape)
    print("SubjID.shape:", SubjID.shape)
    print("SubjIDIndex.shape:", SubjIDIndex.shape)
    print("thick_select.shape:", thick_select.shape)
    print("Age.shape:", Age.shape)
    print("Gender.shape:", Gender.shape)
    print("thick_base.shape:", thick_base.shape)
    print("Age_base.shape:", Age_base.shape)
    print("Gender_base.shape:", Gender_base.shape)
    print("filepaths.shape:", filepaths.shape)
    print("waveid_s.shape:", waveid_s.shape)
    print("SubjID_s.shape:", SubjID_s.shape)
    print("subjtype_s.shape:", subjtype_s.shape)
    print("==========================")
    print("=========================")

    #SubjID_s = SubjID_s.reshape(1, -1)

    #print("subjID_s:", SubjID_s)
    #print("subjtype_s:", subjtype_s)
    #print("waveid_s:", waveid_s)
    #os.exit()

    # data_train, label_train, file_train, num_train
    # extract.extract_patch(target_valid_filepath, target_valid_label, patch_size, stride_size, DATADIR)
    #data_test, label_test, file_test, num_test = extract.extract_patch(filepaths, Label, patch_size, stride_size, DATADIR)
    data_test, label_test, file_test, num_test = extract(filepaths, Label, patch_size, stride_size, DATADIR)
    label_test = label_test
    """
    #数据打乱
    shuffle_ix = np.random.permutation(np.arange(len(label_test)))
    print("shuffle_ix", shuffle_ix)
    data_test = data_test[shuffle_ix]
    label_test = label_test[shuffle_ix]
    file_test = file_test[shuffle_ix]
    num_test = num_test[shuffle_ix]
    print("label_test", label_test)
    """
    print("data_test.shape:", data_test.shape)
    print("label_test.shape:", label_test.shape)
    print("file_test.shape", file_test.shape)
    print("num_train.shape", num_test.shape)
    #os.exit()
    import torch
    fi = foldIDS
    criterion = torch.nn.CrossEntropyLoss()
    #label_results, fc_datas, cnn_datas, logitss

    #label_results, fc_datas, cnn_datas = test_singleCNN.test_cnn_get_data(params['output_dir'], data_test, label_test, file_test, num_test, "test", fi, criterion, params, gpu=params['gpu'])
    label_results, fc_datas = test_singleCNN.test_cnn_get_data(params['output_dir'], data_test, label_test, file_test, num_test, "test", fi, criterion,
                             params, gpu=params['gpu'])

    print("fc_datas:", fc_datas.shape)
    #print("cnn_datas.shape:", cnn_datas.shape)

    pred_list = []
    datas_400 = []
    for i in np.arange(filepaths.shape[0]):
        find = filepaths[i]
        index_act = np.where(label_results['file'] == find)[0]
        fc_data_select = fc_datas[index_act]

        index_pre_act = np.where(label_results['file'] == find)[0]
        pred = label_results['pred'][index_pre_act]

        if len(pred[pred == 0]) > len(pred[pred == 1]):
            each_pred = 0
        else:
            each_pred = 1
        pred_list.append(each_pred)
        #datas_400.append(cnn_data_select)

    #os.exit()
    #print("label_results",label_results)
    pred_list = np.array(pred_list)
    datas_400 = np.array(datas_400)
    print("pred_list.shape:", pred_list.shape)
    print("pred_list.shape:", pred_list)

    subject_results = cnn_utils.evaluate_prediction(label_test.astype(int), pred_list.astype(int))
    print("label_results:", label_results['output_dir'])
    subject_path = os.path.join(label_results['output_dir'], "subject_performance.csv")
    #print("subject results:", subject_results)
    pd.DataFrame(subject_results, index=[0]).to_csv(subject_path, index=False, sep='\t')


    print("datas_400.shape:", datas_400.shape)

    params['foldIDS'] = fi

    Labelmatfilename_400 = os.path.join(params['output_dir'], 'fold-%i' % params['foldIDS'], dataset + '_' + group + '_' + 'D' + '_' + 'patch' +  '_' + '400' + '_label.mat')

    adict = {}
    adict['Age'] = Age
    adict['Gender'] = Gender
    adict['SubjID'] = SubjID.tolist()
    SubjID_s = SubjID_s.tolist()
    adict['Study'] = SubjID_s
    #adict['Viscode'] = Viscode
    adict['Actual'] = Label#actual label
    adict['Subjtype'] = subjtype_s
    adict['Predicted'] = pred_list
    adict['waveid'] = waveid_s.tolist()
    adict['filepaths'] = filepaths.tolist()

    adict['fcdatas'] = datas_400
    hdf5storage.write(adict, '.', Labelmatfilename_400, matlab_compatible=True)

    print("foldIDS-%s end"%(str(foldIDS)))
    print("----------------------------------------------------------------------------")


