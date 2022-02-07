from model import *
from train import *
import math
import os
import argparse
from DataPre import *
import torch

parser = argparse.ArgumentParser(description='PyTorch Implementation of the paper: "TSR-TVD"')
parser.add_argument('--lr_G', type=float, default=1e-4, metavar='LR',
                    help='learning rate of Generator')
parser.add_argument('--lr_D', type=float, default=4e-4, metavar='LR',
                    help='learning rate of Discriminator')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--croptimes', type=int, default=4, metavar='N',
                    help='the number of crops for a pair of data')
parser.add_argument('--x', type=int, default=480, metavar='N',
                    help='x dimension of volume')
parser.add_argument('--y', type=int, default=720, metavar='N',
                    help='y dimension of volume')
parser.add_argument('--z', type=int, default=120, metavar='N',
                    help='z dimension of volume')
parser.add_argument('--crop_x', type=int, default=64, metavar='N',
                    help='crop size along x dimension of volume')
parser.add_argument('--crop_y', type=int, default=96, metavar='N',
                    help='crop size along y dimension of volume')
parser.add_argument('--crop_z', type=int, default=16, metavar='N',
                    help='crop size along z dimension of volume')
parser.add_argument('--init_channels', type=int, default=16, metavar='N',
                    help='init channels of generator')
parser.add_argument('--total_samples', type=int, default=100, metavar='N',
                    help='the samples we used for training TSR-VFD')
parser.add_argument('--interval', type=int, default=3, metavar='N',
                    help='interpolation step')
parser.add_argument('--critic', type=int, default=1, metavar='N',
                    help='number of updating discriminator per updating generator')
parser.add_argument('--l2', type=float, default=1.0, metavar='N',
                    help='weight of l2 loss')
parser.add_argument('--percep', type=float, default=5e-2, metavar='N',
                    help='weight of preceptual loss')
parser.add_argument('--adversarial', type=float, default=1e-3, metavar='N',
                    help='weight of adversarial loss')
parser.add_argument('--mode', type=str, default='train', metavar='N',
                    help='train or infer model')
parser.add_argument('--data_path', type=str, default='../data/', metavar='N',
                    help='the path where we read the scalar data')
parser.add_argument('--model_path', type=str, default='../model/', metavar='N',
                    help='the path where we stored the saved model')
parser.add_argument('--result_path', type=str, default='../result/', metavar='N',
                    help='the path where we stored the synthesized data')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def main():
    ScalarData = DataSet(args)
    ScalarData.ReadData()
    if args.mode == 'train':
        model = TSR(2,1,args.init_channels,args.interval)
        Discriminator = Dis(args.interval)
        if args.cuda:
            model.cuda()
            Discriminator = Discriminator.cuda()
        model.apply(weights_init_kaiming)
        Discriminator(weights_init_kaiming)
        trainGAN(model,Discriminator,args,ScalarData)
    elif args.mode == 'infer':
        model = torch.load(args.model_path+str(args.epochs)+'.pth',map_location=lambda storage, loc:storage)
        if args.cuda:
            model.cuda()
        inference(model,ScalarData,args)

if __name__== "__main__":
    main()
