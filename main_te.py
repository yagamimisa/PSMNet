from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import dfc_lt_Razieh as lt
from dataloader import dfc_da as DA
from models import *
import torchvision
import skimage
import skimage.io

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datapath', default='dataset/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default= None,
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# set gpu id used
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(args.datapath)
all_left_img, all_right_img, all_left_disp = lt.dataloader(args.datapath)

TrainImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(all_left_img[0:4000],all_right_img[0:4000],all_left_disp[0:4000], True), 
         batch_size= 2, shuffle= True, num_workers= 8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(all_left_img[4000:4200],all_right_img[4000:4200],all_left_disp[4000:4200], False),
         batch_size= 1, shuffle= False, num_workers= 4, drop_last=False)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic_ELU(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])


def test(imgL,imgR,disp_true):
        model.eval()
        imgL   = Variable(torch.FloatTensor(imgL))
        imgR   = Variable(torch.FloatTensor(imgR))

        if args.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()

        #---------
        #mask = disp_true < 192
        mask = disp_true > -1   # -999
        #----

        with torch.no_grad():
            output3 = model(imgL,imgR)
        
        output = torch.squeeze(output3.data.cpu(),1)[:,:,:]
        out = output[0,:,:]

        if len(disp_true[mask])==0:
           loss = 0
        else:
           loss = torch.mean(torch.abs(output[mask]-disp_true[mask]))  # end-point-error

        #computing 3-px error#
        pred_disp = output
        pred_disp = pred_disp - 63
        true_disp = disp_true
        index = np.argwhere(true_disp>0)
        disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
        correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3)|(disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)      
        torch.cuda.empty_cache()
        pxLoss =1-(float(torch.sum(correct))/float(len(index[0])))
        return loss, pxLoss

def main():

	#------------- TEST ------------------------------------------------------------
        print('=========================== test ===========================')
	total_test_loss = 0
        total_test_pxLoss = 0
	for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):

	       test_loss, test_pxLoss = test(imgL,imgR, disp_L)
	       print('Iter %d test loss = %.3f and 3-px loss = %.3f' %(batch_idx, test_loss, test_pxLoss))
	       total_test_loss += test_loss
               total_test_pxLoss += test_pxLoss

	print('total test loss = %.3f and total test 3pxLoss = %.3f' %(total_test_loss/len(TestImgLoader), total_test_pxLoss/len(TestImgLoader)))
	#-------------------------------------------------------------------------------
	#SAVE test information
	savefilename = args.savemodel+'testinformation.tar'
	torch.save({
		    'test_loss': total_test_loss/len(TestImgLoader),
		}, savefilename)


if __name__ == '__main__':
   main()
    

