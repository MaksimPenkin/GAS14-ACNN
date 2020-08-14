""" 
 @author   Maxim Penkin
https://github.com/MaksimPenkin

"""

import os
import argparse
import tensorflow as tf
import models.model as model

def parse_args():
    parser = argparse.ArgumentParser(description='deblur arguments')
    parser.add_argument('--max_value', type=float, default=1., help='max value of image pixel')
    parser.add_argument('--normalizer_data', type=str, default='byvalue', help='data normalizer [maxmin, standart_scaler, byvalue]')
    parser.add_argument('--normalizer_data_sub', type=float, default=0., help='if data normalizer=="byvalue" set sub value; (x-sub)/scale')
    parser.add_argument('--normalizer_data_scale', type=float, default=1., help='if data normalizer=="byvalue" set scale value; (x-sub)/scale')

    parser.add_argument('--phase', type=str, default='test', help='determine whether to train or test model')
    parser.add_argument('--checkpoints', type=str, default='checkpoints/GAS14-ACNN', help='checkpoint dir')
    parser.add_argument('--datalist', type=str,
                        default=r"E:\IXI_0_1_lowpass\train\train.txt", # Enter FULL path to train *.txt file
                        help='training datalist')
    parser.add_argument('--val_datalist', type=str,
                        default=r"E:\IXI_0_1_lowpass\val\val.txt", # Enter FULL path to validation *.txt file
                        help='validating datalist')
    parser.add_argument('--test_datalist', type=str,
                        default=r"E:\IXI_0_1_lowpass\test\test.txt", # Enter FULL path to test *.txt file
                        help='testing datalist')

    parser.add_argument('--restore_ckpt', type=int, default=0, help='step for restoring weights of model')
    
    parser.add_argument('--batch_size', help='training batch size', type=int, default=20)
    parser.add_argument('--num_random_crops', help='number of random pathes from each big training image', type=int, default=10)
    
    parser.add_argument('--epoch', help='number of training epochs', type=int, default=1000)
    parser.add_argument('--cnn_size', type=int, default=3, help='CNN filter size')
    parser.add_argument('--feature_size', type=int, default=64, help='number of feature channels')
    parser.add_argument('--conv_blocks_num', type=int, default=32, help='num of conv structural blocks')

    parser.add_argument('--lr', type=float, default=1e-4, dest='learning_rate', help='initial learning rate')
    parser.add_argument('--reg_scale', type=float, default=0.0001, dest='reg_scale', help='l2 regularization scale')
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0', help='use gpu or cpu')

    parser.add_argument('--input_path', type=str, default='./testing_set',
                        help='input path for testing/validation corrupted images')
    parser.add_argument('--output_path', type=str, default="./testing_res",
                        help='output path for testing/validation cnn-processed images')

    args = parser.parse_args()
    return args

def main(_):
    args = parse_args()
    # Set gpu/cpu mode.
    if int(args.gpu_id) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Set up deblur models.
    deblur = model.DEBLUR(args)
    if args.phase == 'test':
        deblur.validate()
    elif args.phase == 'train':
        deblur.train()
    else:
        print('Runtime error: args.phase can be only [train, test]')
        exit(1)

if __name__ == '__main__':
    tf.compat.v1.app.run()


