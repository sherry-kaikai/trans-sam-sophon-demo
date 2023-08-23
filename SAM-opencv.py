import os
import json
import time
import cv2
import argparse
import numpy as np
import sophon.sail as sail

import logging
import warnings

import modeling_sam 
import configuration_sam



logging.basicConfig(level=logging.INFO)


def main(args):
    # check params
    if not os.path.exists(args.input):
        raise FileNotFoundError('{} is not existed.'.format(args.input))
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.bmodel))
    
    # creat save path
    output_dir = "./results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_img_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir) 
    
    # initialize net 
    # todo
    sam = SamModel('/home/sophgo/jingyu/SAM-ViT/tans-SAM-ViT/sam-vit-huge/config.json')

    # warm up (init)
    
    # process images
    # read img
    image = cv2.imread(args.input)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor = SamPredictor(sam)
    predictor.set_image(image)


    # calculate speed  

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/test', help='path of input')
    parser.add_argument('--bmodel', type=str, default='', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')