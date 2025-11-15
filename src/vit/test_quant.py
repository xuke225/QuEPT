import argparse
import os
import random
from utils import *
from quant import *


def get_args_parser():
    
    parser = argparse.ArgumentParser(description="QuEPT-ViT", add_help=False)
    parser.add_argument("--model", default="deit_small",
                            choices=['vit_tiny', 'vit_small', 'vit_base',
                                'deit_tiny', 'deit_small', 'deit_base', 
                                'swin_tiny', 'swin_small', 'swin_base'],
                            help="model")
    parser.add_argument('--dataset', default="/data/imagenet/",
                        help='path to dataset')
    parser.add_argument("--calib-batchsize", default=32,
                        type=int, help="batchsize of validation set")
    parser.add_argument("--val-batchsize", default=200,
                        type=int, help="batchsize of validation set")
    parser.add_argument("--seed", default=0, type=int, help="seed")
    parser.add_argument("--num-workers", default=4, type=int,
                        help="number of data loading workers (default: 4)")
    parser.add_argument("--print-freq", default=100,
                        type=int, help="print frequency")
    parser.add_argument(
        "--output_dir", default="../log/", type=str, help="direction of logging file"
    )

    parser.add_argument("--save_model", action="store_true")

    # setting for quantization
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=4)
    parser.add_argument("--scaleLinear", action="store_true")
    parser.add_argument("--lora_type", type=int, default=0)
    parser.add_argument("--lwc", action="store_true")
    parser.add_argument('--bit_candidate', type=list, default=[4,5,6,7,8], help='bits list for NN')
    parser.add_argument("--lora_scale", action="store_true")
    # setting for reconstruction
    parser.add_argument('--rec_batch_size', type=int, default=32)
    parser.add_argument('--group_L', type=list, default=[4])
    parser.add_argument('--group_M', type=list, default=[5,6])
    parser.add_argument('--group_H', type=list, default=[7,8])
    parser.add_argument('--mixer_level', default="./log/", choices=['Random-Selection', 'Uniform-Fusion','Selective-Merge'])
    parser.add_argument('--topk_token', type=float, default=0.5)
    parser.add_argument('--MFM_PARAM', type=list, default=[0.3,0.3,0.4]) # H M L
    parser.add_argument('--lmd', type=float, default=0.5)
    parser.add_argument('--rec_iters', type=int, default = 2000)
    parser.add_argument('--epochs', type=int, default = 20)
    parser.add_argument("--lr_lora_weight", type=float, default=1e-3)
    parser.add_argument("--lr_scale", type=float, default=1e-4)
    parser.add_argument("--clip_lr", type=float, default=1e-2)
    parser.add_argument("--loraa_lr", type=float, default=0.0075)
    parser.add_argument("--mae", action="store_true")
    return parser

def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_train_samples(train_loader, num_samples, label_info = False):
    train_data = []
    label_data = []
    for batch in train_loader:
        train_data.append(batch[0])
        label_data.append(batch[1])  # label info
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    if not label_info:
        return torch.cat(train_data, dim=0)[:num_samples]
    else:
        return torch.cat(train_data, dim=0)[:num_samples],torch.cat(label_data, dim=0)[:num_samples]

def validate(args, val_loader, model, criterion, device):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    for i, (data, target) in enumerate(val_loader):
        target = target.to(device)
        data = data.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(data)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))

        if i % args.print_freq == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    i,
                    len(val_loader),
                    loss=losses,
                    top1=top1,
                    top5=top5,
                )
            )


    return losses.avg, top1.avg, top5.avg


def validate_calibrate_data(args, calidata, calilabel, model, criterion, device):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to evaluate mode
    model.eval()
    bz = calidata.size(0) / 64
    
    with torch.no_grad():
        for i in range(int(bz)):
            data, target = calidata[i * 64: (i + 1) * 64].to(device), calilabel[
                i * 64: (i + 1) * 64
            ].to(device)
        
            output = model(data)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))
        
    return losses.avg, top1.avg, top5.avg


def main():
    seed(args.seed)
    model_zoo = {
        'vit_tiny' : 'vit_tiny_patch16_224', 
        'vit_small' : 'vit_small_patch16_224',
        'vit_base' : 'vit_base_patch16_224',

        'deit_tiny' : 'deit_tiny_patch16_224',
        'deit_small': 'deit_small_patch16_224',
        'deit_base' : 'deit_base_patch16_224',

        'swin_tiny' : 'swin_tiny_patch4_window7_224',
        'swin_small': 'swin_small_patch4_window7_224',
        'swin_base': 'swin_base_patch4_window7_224',
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Build logger
    logger = get_logger(args.output_dir)
    
    # Build dataloader
    print('Building dataloader ...')
    train_loader, val_loader = build_dataset(args)
    calib_data, calib_info = get_train_samples(train_loader, 1024, True)
    criterion = nn.CrossEntropyLoss().to(device)
    # Build model
    print('Building model ...')
    model = build_model(model_zoo[args.model])
    model.to(device)
    model.eval()

    args.weight_quant_params = {'n_bits': args.wbits, 
                                'bit_candidate': args.bit_candidate, 
                                'lwc': args.lwc, 
                                }
    args.act_quant_params = {'n_bits': args.abits, 'channel_wise': False, 
                            "prob": 0.5,'bit_candidate': args.bit_candidate}

    logger.info(args)
    
    q_model = QSwinViT(model, args=args) if 'swin' in args.model else QViT(model, args=args)
    set_8bit_headstem(q_model)
    q_model.to(device)
    q_model.eval()
    device = next(q_model.parameters()).device
   
    outlier_handle_kwargs = dict(cali_data = calib_data[:128], input_prob=0.5, 
                                 keep_gpu=True,args=args)    
    
    block_type = type(model.layers[0]) if 'swin' in args.model else type(q_model.blocks[0])
    
    def outlier_handler_model(model: nn.Module, kwargs):
        """
        handle outlier for transformer block
        """
        for name, module in model.named_children():
            if isinstance(module, (QuantLinear, QuantScalableLinear)):
                layer_handle(q_model, module, **kwargs)
            elif isinstance(module, block_type):
                print('Handle outlier for block {}'.format(name))
                block_handle(q_model, module, **kwargs)
            else:
                outlier_handler_model(module, kwargs) 

    outlier_handler_model(q_model, kwargs = outlier_handle_kwargs)
    set_quant_state(q_model,True,True)

    kwargs = dict(cali_data=calib_data, iters=args.rec_iters, 
                  lr_weight = args.lr_lora_weight,
                  batch_size = args.rec_batch_size,
                  lr_scale=args.lr_scale, input_prob=0.5, 
                  keep_gpu=True, logger = logger, args=args)
    

    rec_model(q_model = q_model, **kwargs)

    print("Validating ...")
    set_quant_state(q_model,True,True)
    criterion = nn.CrossEntropyLoss().to(device)

    for i in args.bit_candidate:
        model_bit_refactor(q_model, i)
        logger.info("-------------------***--------------------")
        val_loss, val_prec1, val_prec5 = validate(
            args, val_loader, q_model, criterion, device
        )
        logger.info("Under {}-bit * Prec@1 {:.3f} Prec@5 {:.3f}".format(i ,val_prec1, val_prec5))
        val_loss, val_prec1, val_prec5 = validate_calibrate_data(
            args, calib_data, calib_info, q_model, criterion, device
        )
        logger.info("Under {}-bit * Prec@1 {:.3f} Prec@5 {:.3f} on Calibration datas ".format(i, val_prec1, val_prec5))
    

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser('QuEPT-ViT', parents=[get_args_parser()])
    args = parser.parse_args()
    main()