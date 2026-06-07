import torch
from hqq.core.quantize import BaseQuantizeConfig
import hqq_wrapper
from spectralSpacialMamba.utils import test_batch
    

def test_batch_quantized(args, model, image, index, BATCH_SIZE, nTrain_perClass, nvalid_perClass, halfsize,):
        
    quant_config = BaseQuantizeConfig(
        nbits=args.nbits,
        group_size=args.group_size,
    )

# put all these in args
    model = hqq_wrapper.replace_all_linear_with_hqq(
        model = model,
        quant_config = quant_config,
        compute_dtype = torch.float32,
        del_orig = args.del_orig,
        verbose = args.verbose,
        exclude_names = args.exclude_names,
    )

    model.evail()

    true_cla_q, oa_q, aa_q, kappa_q, cm_q, pred_q = test_batch(
        model, image, index, BATCH_SIZE, nTrain_perClass, nvalid_perClass, halfsize
    )

    print('quantized overall_accuracy: {0:f}'.format(oa_q))
    print('quantized average_accuracy: {0:f}'.format(aa_q))
    print('quantized kappa:{0:f}'.format(kappa_q))

    return true_cla_q, oa_q, aa_q, kappa_q, cm_q, pred_q
    




        
