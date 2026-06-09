import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import loadmat
from scipy.io import savemat
from torch import optim
from torch.autograd import Variable
import wandb
from .vit_pytorch import ViT
from sklearn.metrics import confusion_matrix
from download_dataset import downloadAndLoadDataset
from mvit.data_prepare import choose_train_and_test
from mvit.CNNUtils import train, test, valid
from .quantize_sf import test_batch_quantized
from mvit.utils import class_accuracy_percent
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import time
import os
from mvit.utils import output_metric

# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
#-------------------------------------------------------------------------------
# 定位训练和测试样本
def chooose_train_and_test_point(label, train_num=20, seed=0):
    return choose_train_and_test(label, train_num, seed)

#-------------------------------------------------------------------------------
# 边界拓展：镜像
def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    #中心区域
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    #左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    #上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    #下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi
#-------------------------------------------------------------------------------
# 获取patch的图像数据
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image

def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=float)
    # 中心区域
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape
    #左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]
    #右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band
#-------------------------------------------------------------------------------
# 汇总训练数据和测试数据
def train_and_test_data(mirror_image, band, train_point, test_point, true_point, patch=5, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    for k in range(true_point.shape[0]):
        x_true[k,:,:,:] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)
    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    print("x_true  shape = {}, type = {}".format(x_true.shape,x_test.dtype))
    print("**************************************************")
    
    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    x_true_band = gain_neighborhood_band(x_true, band, band_patch, patch)
    print("x_train_band shape = {}, type = {}".format(x_train_band.shape,x_train_band.dtype))
    print("x_test_band  shape = {}, type = {}".format(x_test_band.shape,x_test_band.dtype))
    print("x_true_band  shape = {}, type = {}".format(x_true_band.shape,x_true_band.dtype))
    print("**************************************************")
    return x_train_band, x_test_band, x_true_band
#-------------------------------------------------------------------------------
def train_and_test_label(number_train, number_test, number_valid, num_classes):
    y_train = []
    y_test = []
    y_valid = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
        for n in range(number_valid[i]):
            y_valid.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_valid = np.array(y_valid)
    print("y_train: shape = {}, type = {}".format(y_train.shape, y_train.dtype))
    print("y_test: shape = {}, type = {}".format(y_test.shape, y_test.dtype))
    print("y_valid: shape = {}, type = {}".format(y_valid.shape, y_valid.dtype))
    print("*******************************************************")
    return y_train, y_test, y_valid
#-------------------------------------------------------------------------------
class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
#-------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res, target, pred.squeeze()
#-------------------------------------------------------------------------------
# train model
def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   

        optimizer.zero_grad()
        batch_pred = model(batch_data)
        loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()       

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre
#-------------------------------------------------------------------------------
# validate model
def valid_epoch(model, valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   

        batch_pred = model(batch_data)
        
        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
        
    return tar, pre

def test_epoch(model, test_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   

        batch_pred = model(batch_data)

        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre

#-------------------------------------------------------------------------------

def train_spectralformer(args):
    from download_dataset import downloadAndLoadDataset
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    cudnn.deterministic = True
    cudnn.benchmark = False

    data, label = downloadAndLoadDataset(args.dataset)
    num_classes = np.max(label)

    input_normalize = np.zeros(data.shape)
    for i in range(data.shape[2]):
        input_max = np.max(data[:,:,i])
        input_min = np.min(data[:,:,i])
        input_normalize[:,:,i] = (data[:,:,i]-input_min)/(input_max-input_min)
    # data size
    height, width, band = data.shape
    print("height={0},width={1},band={2}".format(height, width, band))
    #-------------------------------------------------------------------------------
    # obtain train and test data
    total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = chooose_train_and_test_point(label, args.train_num, args.seed)

    mirror_image = mirror_hsi(height, width, band, input_normalize, patch=args.patches_sf)
    x_train_band, x_test_band, x_true_band = train_and_test_data(mirror_image, band, total_pos_train, total_pos_test, total_pos_true, patch=args.patches, band_patch=args.band_patches)
    y_train, y_test, y_true = train_and_test_label(number_train, number_test, number_true, num_classes)
    #-------------------------------------------------------------------------------
    # load data
    x_train=torch.from_numpy(x_train_band.transpose(0,2,1)).type(torch.FloatTensor) #[695, 200, 7, 7]
    y_train=torch.from_numpy(y_train).type(torch.LongTensor) #[695]
    Label_train=Data.TensorDataset(x_train,y_train)
    x_test=torch.from_numpy(x_test_band.transpose(0,2,1)).type(torch.FloatTensor) # [9671, 200, 7, 7]
    y_test=torch.from_numpy(y_test).type(torch.LongTensor) # [9671]
    Label_test=Data.TensorDataset(x_test,y_test)
    x_true=torch.from_numpy(x_true_band.transpose(0,2,1)).type(torch.FloatTensor)
    y_true=torch.from_numpy(y_true).type(torch.LongTensor)
    Label_true=Data.TensorDataset(x_true,y_true)

    label_train_loader=Data.DataLoader(Label_train,batch_size=args.batch_size,shuffle=True)
    label_test_loader=Data.DataLoader(Label_test,batch_size=args.batch_size,shuffle=True)
    label_true_loader=Data.DataLoader(Label_true,batch_size=args.batch_size,shuffle=True)

    #-------------------------------------------------------------------------------
    # create model
    model = ViT(
        image_size = args.patches_sf,
        near_band = args.band_patches_sf,
        num_patches = band,
        num_classes = num_classes,
        dim = 64,
        depth = 5,
        heads = 4,
        mlp_dim = 8,
        dropout = 0.1,
        emb_dropout = 0.1,
        mode = args.mode
    )
    model = model.cuda()
    # criterion
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches//10, gamma=args.gamma)
    #-------------------------------------------------------------------------------
    # if args.flag_test == 'test':
    #     if args.mode == 'ViT':
    #         model.load_state_dict(torch.load('./ViT.pt'))      
    #     elif (args.mode == 'CAF') & (args.patches == 1):
    #         model.load_state_dict(torch.load('./SpectralFormer_pixel.pt'))
    #     elif (args.mode == 'CAF') & (args.patches == 7):
    #         model.load_state_dict(torch.load('./SpectralFormer_patch.pt'))
    #     else:
    #         raise ValueError("Wrong Parameters") 
    #     model.eval()
    #     tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
    #     OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)

    #     # output classification maps
    #     pre_u = test_epoch(model, label_true_loader, criterion, optimizer)
    #     prediction_matrix = np.zeros((height, width), dtype=float)
    #     for i in range(total_pos_true.shape[0]):
    #         prediction_matrix[total_pos_true[i,0], total_pos_true[i,1]] = pre_u[i] + 1
    #     plt.subplot(1,1,1)
    #     # plt.imshow(prediction_matrix, colors.ListedColormap(color_matrix))
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.show()
    #     savemat('matrix.mat',{'P':prediction_matrix, 'label':label})
    print('start training')
    acc_list = [0.00]
    os.makedirs('./model', exist_ok=True)
    path = './model/mvit.pt'
    tic = time.time()
    for epoch in range(args.epoches):
        # 计算的是移动平均准确率
        train_acc, train_loss = train(model, label_train_loader, criterion, optimizer)
        valid_acc, valid_loss = valid(model, label_true_loader, criterion)

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

    # test model after training
    test_tar, test_pre = test(model, label_test_loader)
    OA, AA_mean, kappa, AA = output_metric(test_tar, test_pre)
    per_class_acc = class_accuracy_percent(test_tar, test_pre, num_classes)

    # quantize model and test
    quantized_model = test_batch_quantized(args, model)
    test_tar_quantized, test_pre_quantized = test(quantized_model, label_test_loader)
    OA_quantized, AA_mean_quantized, kappa_quantized, AA_quantized = output_metric(test_tar_quantized, test_pre_quantized)
    per_class_acc_quantized = class_accuracy_percent(test_tar_quantized, test_pre_quantized, num_classes)   
    

    results = {
        'model' : 'SpectralFormer',
        'dataset': args.dataset,
        'OA': OA,
        'AA': AA_mean,
        'Kappa': kappa,
        'OA_quantized': OA_quantized,
        'AA_quantized': AA_mean_quantized,
        'Kappa_quantized': kappa_quantized,
    }

     ### extract per class accuracy 
    print("OA: {:.4f}, AA: {:.4f}, Kappa: {:.4f}".format(OA, AA_mean, kappa))
    for c in range(num_classes):
        results[f'class_{c+1}_acc'] = per_class_acc[c]
        results[f'class_{c+1}_acc_quantized'] = per_class_acc_quantized[c]

    for key, value in results.items():
        print(f"{key}: {value}")
        
    if args.wandb_mode != 'disabled':
        wandb.log(results)
        wandb.finish()


