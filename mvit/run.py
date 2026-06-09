import os
import time
import math
import argparse
import numpy as np
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as Data
import torch.nn.functional as F
import wandb
from spectralSpacialMamba.run import getClassOutputForEachClass
from spectralSpacialMamba.run import getClassOutputForEachClass
from torchsummary import summary
from einops import rearrange, repeat
from timm.models.vision_transformer import Block


from .data_prepare import mirror_hsi
from .data_prepare import choose_train_and_test
from .data_prepare import choose_all_pixels, all_data
from .data_prepare import train_and_test_data, train_and_test_label


from .utils import applyPCA, output_metric, plot_confusion_matrix
from .CNNUtils import train, test, valid
from .utils import list_to_colormap, classification_map, print_args
from .utils import ActivationOutputData
from download_dataset import downloadAndLoadDataset
from .model import MViT
from .utils import class_accuracy_percent
from .quantize_mvit import test_batch_quantized
from spectralSpacialMamba.quantize_mamba import getParamCount

# %matplotlib inline


def run_mvit(args):

    # Load data
    data, label = downloadAndLoadDataset(args.dataset)
    num_classes = int(np.max(label))

    # Preprocess data

    # apply normalization
    shapeor = data.shape
    data = data.reshape(np.prod(data.shape[:2]), np.prod(data.shape[2:]))

    std_scaler = StandardScaler()
    std_data = std_scaler.fit_transform(data)
    data = std_data.reshape(shapeor)

    K = 30
    data, pca = applyPCA(data, numComponents=K)

    # data size
    height, width, band = data.shape
    print("height={0}, width={1}, band={2}".format(height, width, band))

    mirror_data = mirror_hsi(height, width, band, data, patch_size=args.patch_size_mvit)

    if args.dataset == 'Indian':
        train_num = 10
    else:
        train_num= args.train_num
    total_pos_train, total_pos_test, total_pos_valid, number_train, number_test, number_valid = choose_train_and_test(
        label, num_train_per_class=train_num, seed=args.seed
    )

    x_train, x_test, x_valid = train_and_test_data(
        mirror_data, band, total_pos_train, total_pos_test, total_pos_valid, patch_size=args.patch_size_mvit
    )
    y_train, y_test, y_valid = train_and_test_label(number_train, number_test, number_valid, num_classes)

    # load data
    x_train = torch.from_numpy(x_train.transpose(0, 3, 1, 2)).unsqueeze(1).type(torch.FloatTensor)  # (90, 30, 15, 15)
    print(x_train.shape)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)  # (13,)
    train_label = Data.TensorDataset(x_train, y_train)

    x_test = torch.from_numpy(x_test.transpose(0, 3, 1, 2)).unsqueeze(1).type(torch.FloatTensor)  # (5198, 30, 15, 15)
    print(x_test.shape)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)  # (5198,)
    test_label = Data.TensorDataset(x_test, y_test)

    x_valid = torch.from_numpy(x_valid.transpose(0, 3, 1, 2)).unsqueeze(1).type(torch.FloatTensor)  # (5211, 30, 15, 15)
    print(x_valid.shape)
    y_valid = torch.from_numpy(y_valid).type(torch.LongTensor)
    valid_label = Data.TensorDataset(x_valid, y_valid)

    train_loader = Data.DataLoader(train_label, batch_size=args.batch_size, shuffle=True)
    test_loader = Data.DataLoader(test_label, batch_size=args.batch_size, shuffle=True)
    valid_loader = Data.DataLoader(valid_label, batch_size=args.batch_size, shuffle=True)

    model = MViT(num_classes=num_classes).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate_mvit, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay_mvit
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma_mvit)

    getParamCount(model, printLayers=True)
    
    print('start training')
    acc_list = [0.00]
    os.makedirs('./model', exist_ok=True)
    path = './model/mvit.pt'
    tic = time.time()
    for epoch in range(args.epoch):
        # 计算的是移动平均准确率
        train_acc, train_loss = train(model, train_loader, criterion, optimizer)
        valid_acc, valid_loss = valid(model, valid_loader, criterion)
        print("Epoch: {:03d} - train_loss: {:.4f} - train_acc: {:.4f} - valid_loss: {:.4f} - valid_acc: {:.4f}".\
            format(epoch+1, train_loss, train_acc, valid_loss, valid_acc))
        scheduler.step()

        acc_list.append(valid_acc)
        if acc_list[-1] > acc_list[-2]:
            print("val_acc improved from {:.4f} to {:.4f}, saving model to mvit.pt".format(acc_list[-2], acc_list[-1]))
            torch.save(model.state_dict(), path)
        else:
            print("val_acc did not improve from {:.4f}".format(acc_list[-2]))
            acc_list[-1] = acc_list[-2]

    toc = time.time()
    print("Running Time: {:.2f}".format(toc-tic))
    print("**************************************************")

    print("started testing")
    model.load_state_dict(torch.load(path))
    model.eval()

    # test model before quantizing
    test_tar, test_pre = test(model, test_loader)
    OA, AA_mean, kappa, AA = output_metric(test_tar, test_pre)

    # quantize model and test 
    quantized_model = test_batch_quantized(args, model)

    #test quantized model
    test_tar_quantized, test_pre_quantized = test(quantized_model, test_loader)
    OA_quantized, AA_mean_quantized, kappa_quantized, AA_quantized = output_metric(test_tar_quantized, test_pre_quantized)

    # get per class accuracy for both original and quantized model
    class_acc = class_accuracy_percent(test_tar, test_pre, num_classes)
    clas_acc_quantized = class_accuracy_percent(test_tar_quantized, test_pre_quantized, num_classes)

    results = {
        'model' : 'MViT',
        'dataset': args.dataset,
        'OA': OA,
        'AA': AA_mean,
        'Kappa': kappa,
        'OA_quantized': OA_quantized,
        'AA_quantized': AA_mean_quantized,
        'Kappa_quantized': kappa_quantized,
        **getClassOutputForEachClass(args.dataset, class_acc),
        **getClassOutputForEachClass(args.dataset, clas_acc_quantized, is_quantized=True),
    }

    for key, value in results.items():
        print(f"{key}: {value}")

    if args.wandb_mode != 'disabled':
        wandb.log(results)
        wandb.finish()
