import numpy as np

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
from utils.load_model import model_loader
from utils.get_model_summary import print_quantization_summary

from quantizer.quantize_hqq import hqq_quantization
from quantizer.quantize_quanto import quanto_quantization

from utils.load_model import model_loader
from train import get_base_args


def get_args():
    parser = get_base_args()
    parser.add_argument('--quant_method', type=str, choices=['hqq', 'quanto'], default='hqq')
    parser.add_argument('--nbits', type=int, default=8)
    parser.add_argument('--print_quantization_summary', type=int, default=1) # 0 for false, 1 for true
    parser.add_argument('--group_size', type=lambda x: None if x.lower() == 'none' else int(x), default=-1)
    parser.add_argument('--del_orig', type=lambda x: x.lower() == 'true', default=True, help='if True, delete the original Linear weight inside HQQLinear')
    parser.add_argument('--verbose', type=lambda x: x.lower() == 'true', default=True, help='if True, print replacement information')

    # wandb 
    parser.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument('--wandb_project', type=str, default='QHSIC_sfinal_studyf', help='wandb project name')

    args = parser.parse_args()
    return args

def main():
    """
    Main function to run the quantization evaluation.
    This function loads the saved model, evaluates it on the test set,
    then applies quantization, and evaluates the quantized model.
    """ 
    args = get_args()

    if args.wandb_mode != 'disabled':
        wandb.init(
            project = args.wandb_project,
            name = f"{args.model}_{args.dataset}_{args.quant_method}_nbits{args.nbits}_group{args.group_size}",
            mode = args.wandb_mode,
            config = vars(args)
        )

    print("*****************************************************************")
    print("Printing All Arguments:")
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

    if args.dataset == 'Indian': #hardcoding cause, a class in indian Pines has only 20 samples
        train_num = 10
    else:
        train_num= args.train_num
    total_pos_train, total_pos_test, total_pos_valid, number_train, number_test, number_valid = choose_train_and_test(
        label, num_train_per_class=train_num, seed=args.seed
    )

    _, x_test, _ = train_and_test_data(
        mirror_data, band, total_pos_train, total_pos_test, total_pos_valid, patch_size=args.patch_size
    )
    _, y_test, _ = train_and_test_label(number_train, number_test, number_valid, num_classes)

    x_test = torch.from_numpy(x_test.transpose(0, 3, 1, 2)).type(torch.FloatTensor) 
    if args.model == 'mvit' or args.model == 'mf':
        x_test = x_test.unsqueeze(1)
    print(x_test.shape)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)  
    test_label = Data.TensorDataset(x_test, y_test)

    print("*****************************************************************")
    print(f"x_test shape: {x_test.shape}")
    print("*****************************************************************")

    test_loader = Data.DataLoader(test_label, batch_size=args.batch_size, shuffle=False)

    saved_path = './model/' + args.model + '_' + args.dataset + '.pt'
    model = model_loader(args, num_class=num_classes)
    model.load_state_dict(torch.load(saved_path))
    model.eval()

    # test model before quantizing
    print("started testing model before quantization")
    test_tar, test_pre = test(model, test_loader)
    OA, AA_mean, kappa, AA = output_metric(test_tar, test_pre)

    # quantize model
    print("started quantization")
    if args.quant_method == 'hqq':
        quantized_model = hqq_quantization(args, model)
    elif args.quant_method == 'quanto':
        quantized_model = quanto_quantization(args, model)
    print("Model has been quantized successfully")

    # if args.print_quantization_summary:
    #     print("\n[INFO]__________________________________ Model after quantization: __________________________________")
    #     print_quantization_summary(quantized_model)

    print("started testing model after quantization")
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
        'model_name_full' : model_name,
        'OA': OA * 100,
        'AA': AA_mean * 100,
        'Kappa': kappa * 100,
        'OA_quantized': OA_quantized * 100,
        'AA_quantized': AA_mean_quantized * 100,
        'Kappa_quantized': kappa_quantized * 100,
        **getClassOutputForEachClass( class_acc),
        **getClassOutputForEachClass(clas_acc_quantized, is_quantized=True),
    }

    print("*****************************************************************")
    print("Final Results:")
    print("*****************************************************************")
    for key, value in results.items():
        print(f"{key}: {value}")

    if args.wandb_mode != 'disabled':
        wandb.log(results)
        wandb.finish()

if __name__ == "__main__":
    main()
    