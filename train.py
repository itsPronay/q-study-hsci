import os
import time
import argparse
import numpy as np
import argparse

from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as Data
from utils.data_prepare import mirror_hsi
from utils.data_prepare import choose_train_and_test
from utils.data_prepare import train_and_test_data, train_and_test_label
from utils.data_prepare import applyPCA
from utils.train_utils import train, valid
from utils.download_dataset import downloadAndLoadDataset
from utils.get_model_summary import getParamCount
from utils.load_model import model_loader


parser = argparse.ArgumentParser(description='Quantization study')

def get_base_args():

    parser.add_argument('--model', type=str,choices=['sf', 'ssm', 'mvit', 'mf'], default='mvit')
    parser.add_argument('--dataset', type=str, choices=['UP', 'NF', 'HC', 'Pavia', 'Indian', 'Houston'], default='UP')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--patch_size', type=int, default=15)
    parser.add_argument('--band_patch', type=int, default=1)
    parser.add_argument('--pca_band', type=int, default=30)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_num', type=int, default=20) 
    return parser

def main():
    parser = get_base_args()
    args = parser.parse_args()

    print("*****************************************************************")
    print("Printing All Arguments while Training:")
    print("*****************************************************************")
    for arg in vars(args):
        if arg == 'train_num' and getattr(args, arg) == 20 and args.dataset == 'Indian':
            print(f"{arg}: 10")
            print("For Indian Pines, the train number value is set to 10 per class")
        else:
            print(f"{arg}: {getattr(args, arg)}")
    print("*****************************************************************")

    # Load data
    data, label = downloadAndLoadDataset(args.dataset)
    num_classes = int(np.max(label))

    # apply normalization
    shapeor = data.shape
    data = data.reshape(np.prod(data.shape[:2]), np.prod(data.shape[2:]))

    std_scaler = StandardScaler()
    std_data = std_scaler.fit_transform(data)
    data = std_data.reshape(shapeor)

    data, pca = applyPCA(data, numComponents=args.pca_band)

    # data size
    height, width, band = data.shape
    print("height={0}, width={1}, band={2}".format(height, width, band))

    mirror_data = mirror_hsi(height, width, band, data, patch_size=args.patch_size)

    if args.dataset == 'Indian':
        train_num = 10
    else:
        train_num= args.train_num
    total_pos_train, total_pos_test, total_pos_valid, number_train, number_test, number_valid = choose_train_and_test(
        label, num_train_per_class=train_num, seed=args.seed
    )

    x_train, x_test, x_valid = train_and_test_data(
        mirror_data, band, total_pos_train, total_pos_test, total_pos_valid, patch_size=args.patch_size
    )
    y_train, _, y_valid = train_and_test_label(number_train, number_test, number_valid, num_classes)

    # load data
    x_train = torch.from_numpy(x_train.transpose(0, 3, 1, 2)).type(torch.FloatTensor)  
    if args.model == 'mvit' or args.model == 'mf':
        x_train = x_train.unsqueeze(1)
    print(x_train.shape)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)  
    train_label = Data.TensorDataset(x_train, y_train)

    x_valid = torch.from_numpy(x_valid.transpose(0, 3, 1, 2)).type(torch.FloatTensor)
    if args.model == 'mvit' or args.model == 'mf':
        x_valid = x_valid.unsqueeze(1)
    print(x_valid.shape)
    y_valid = torch.from_numpy(y_valid).type(torch.LongTensor)
    valid_label = Data.TensorDataset(x_valid, y_valid)

    print("*****************************************************************")
    print(f"x_train shape: {x_train.shape}, \nx_test shape: {x_test.shape} \n x_valid shape: {x_valid.shape}")
    print("*****************************************************************")

    train_loader = Data.DataLoader(train_label, batch_size=args.batch_size, shuffle=True)
    valid_loader = Data.DataLoader(valid_label, batch_size=args.batch_size, shuffle=True)

    model = model_loader(args, num_class=num_classes)
    print(model)
    print("Model loaded successfully")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    print("*****************************************************************")
    print("Printing Model:")
    print("*****************************************************************")
    print(model)
    params = getParamCount(model, printLayers=False)
    print(f"Total number of trainable params in the model: {params}")
    print("*****************************************************************")

    print('started training')
    acc_list = [0.00]
    os.makedirs('./model', exist_ok=True)
    path = './model/' + args.model + '_' + args.dataset + '.pt'
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
            print("val_acc improved from {:.4f} to {:.4f}, saving model to {}".format(acc_list[-2], acc_list[-1], path))
            torch.save(model.state_dict(), path)
        else:
            print("val_acc did not improve from {:.4f}".format(acc_list[-2]))
            acc_list[-1] = acc_list[-2]

    toc = time.time()
    print("*****************************************************************")
    print("Training completed in: {:.2f}".format(toc-tic))
    print(f"Model saved to: {path}")
    print("*****************************************************************")


if __name__ == "__main__":
    main()