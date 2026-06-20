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
import wandb
from utils.data_prepare import mirror_hsi
from utils.data_prepare import choose_train_and_test
from utils.data_prepare import train_and_test_data, train_and_test_label
from utils.train_utils import getClassOutputForEachClass
from utils.data_prepare import applyPCA
from utils.train_utils import train, test, valid, output_metric, class_accuracy_percent
from utils.download_dataset import downloadAndLoadDataset
from utils.get_model_summary import getParamCount
from utils.load_model import model_loader
from utils.get_model_summary import print_quantization_summary

from quantizer.quantize_hqq import hqq_quantization
from quantizer.quantize_torchao import torchao_quantization


parser = argparse.ArgumentParser(description='Quantization study')

parser.add_argument('--model', type=str,choices=['sf', 'ssm', 'mvit', 'mf'], default='mvit')
parser.add_argument('--dataset', type=str, choices=['UP', 'NF', 'HC', 'Pavia', 'Indian', 'Houston'], default='UP')
parser.add_argument('--quant_method', type=str, choices=['hqq', 'torchao'], default='hqq')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--patch_size', type=int, default=15)
parser.add_argument('--band_patch', type=int, default=1)
parser.add_argument('--pca_band', type=int, default=30)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--drop_rate', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--lr_decay', type=float, default=0.5)
# chec this out

# sdak n kds fs
parser.add_argument('--train_num', type=int, default=10) # sohjfa isfhdihasiudfh iusahf isdd asdf sdf adsf 

# quantization specific args
parser.add_argument('--nbits', type=int, default=8)
parser.add_argument('--print_layers', type=int, default=0) # 0 for false, 1 for true
parser.add_argument('--print_quantization_summary', type=int, default=1) # 0 for false, 1 for true
parser.add_argument('--group_size', type=lambda x: None if x.lower() == 'none' else int(x), default=64)
parser.add_argument('--del_orig', type=lambda x: x.lower() == 'true', default=True, help='if True, delete the original Linear weight inside HQQLinear')
parser.add_argument('--verbose', type=lambda x: x.lower() == 'true', default=True, help='if True, print replacement information')

# these are the args for spectralSpacialMamba specifically, 
# Pre training
parser.add_argument('--windowsize', type=int, default=27) # 13 * 2 + 1 = 27 ---- 7 * 2 + 1 = 15
parser.add_argument('--type', type=str, default='none')
parser.add_argument('--spe_windowsize', type=int, default=3)
parser.add_argument('--hid_chans', type=int, default=64)
parser.add_argument('--embed_dim', type=int, default=64)
parser.add_argument('--depth', type=int, default=4)

parser.add_argument('--use_bi', default=True, type=lambda x: (str(x).lower() == 'true'), help='use bidirection or not' )  
parser.add_argument('--use_global', default=True, type=bool,help='use token meaning or not') 
parser.add_argument('--use_cls', default=True, type=bool,help='use class tken or not') 
parser.add_argument('--use_fu', default=True, type=lambda x: (str(x).lower() == 'true'),help='use center augmentation fusion or not') 

# parser.add_argument('--gpu_id', default='0', help='gpu id')

# wandb args
parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
parser.add_argument('--wandb_project', type=str, default='Quantization Study', help='wandb project name')

args = parser.parse_args()

def main():
    # if not os.path.exists('classification_maps'):
    #     os.makedirs('classification_maps')
        
    if args.wandb_mode != 'disabled':
        wandb.init(
            project = args.wandb_project,
            name = f"{args.model}_{args.dataset}_quantization_{args.nbits}bits_group{args.group_size}",
            mode = args.wandb_mode,
            config = vars(args)
        )

    print("*****************************************************************")
    print("Printing All Arguments:")
    print("*****************************************************************")
    for arg in vars(args):
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
    y_train, y_test, y_valid = train_and_test_label(number_train, number_test, number_valid, num_classes)

    # load data
    x_train = torch.from_numpy(x_train.transpose(0, 3, 1, 2)).type(torch.FloatTensor)  
    if args.model == 'mvit' or args.model == 'mf':
        x_train = x_train.unsqueeze(1)
    print(x_train.shape)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)  
    train_label = Data.TensorDataset(x_train, y_train)

    x_test = torch.from_numpy(x_test.transpose(0, 3, 1, 2)).type(torch.FloatTensor) 
    if args.model == 'mvit' or args.model == 'mf':
        x_test = x_test.unsqueeze(1)
    print(x_test.shape)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)  
    test_label = Data.TensorDataset(x_test, y_test)

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
    test_loader = Data.DataLoader(test_label, batch_size=args.batch_size, shuffle=True)
    valid_loader = Data.DataLoader(valid_label, batch_size=args.batch_size, shuffle=True)

    model = model_loader(args, num_class=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

    print("*****************************************************************")
    print("Printing Model:")
    print("*****************************************************************")
    print(model)
    getParamCount(model, printLayers=False)
    print("*****************************************************************")

    print('started training')
    acc_list = [0.00]
    os.makedirs('./model', exist_ok=True)
    path = './model/' + args.model + '.pt'
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
    print("Running Time: {:.2f}".format(toc-tic))
    print("*****************************************************************")

    print("started testing")
    model.load_state_dict(torch.load(path))
    model.eval()

    # test model before quantizing
    test_tar, test_pre = test(model, test_loader)
    OA, AA_mean, kappa, AA = output_metric(test_tar, test_pre)

    # quantize model
    if args.quant_method == 'hqq':
        quantized_model = hqq_quantization(args, model)
    elif args.quant_method == 'torchao':
        quantized_model = torchao_quantization(args, model)

    if args.print_quantization_summary:
        print("\n[INFO]__________________________________ Model after quantization: __________________________________")
        print_quantization_summary(quantized_model)

    #test quantized model
    test_tar_quantized, test_pre_quantized = test(quantized_model, test_loader)
    OA_quantized, AA_mean_quantized, kappa_quantized, AA_quantized = output_metric(test_tar_quantized, test_pre_quantized)

    # get per class accuracy for both original and quantized model
    class_acc = class_accuracy_percent(test_tar, test_pre, num_classes)
    clas_acc_quantized = class_accuracy_percent(test_tar_quantized, test_pre_quantized, num_classes)

    if args.model == 'mvit':
        model_name = 'MViT'
    elif args.model == 'ssm':
        model_name = 'SpectralSpacialMamba'
    elif args.model == 'sf':
        model_name = 'SpectralFormer'
    elif args.model == 'mf':
        model_name = 'MassFormer'

    results = {
        'model' : model_name,
        'quantization_method': args.quant_method,
        'dataset': args.dataset,
        'OA': OA * 100,
        'AA': AA_mean * 100,
        'Kappa': kappa * 100,
        'OA_quantized': OA_quantized * 100,
        'AA_quantized': AA_mean_quantized * 100,
        'Kappa_quantized': kappa_quantized * 100,
        **getClassOutputForEachClass(args.dataset + '_' + args.model, class_acc),
        **getClassOutputForEachClass(args.dataset + '_' + args.model, clas_acc_quantized, is_quantized=True),
    }

    for key, value in results.items():
        print(f"{key}: {value}")

    if args.wandb_mode != 'disabled':
        wandb.log(results)
        wandb.finish()

if __name__ == "__main__":
    main()