import os
import json
import time
import cv2
import argparse
import numpy as np
import sophon.sail as sail

import logging
import warnings


from modeling_sam import SamModel
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
    sam = SamModel(args)

    # warm up (init)
    sam.init()
    
    
    # read img
    image = cv2.imread(args.input)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # process images 
    # just resize ? 
    # input = cv2.resize(image,(1024,1024),interpolation=cv2.INTER_LINEAR)
    # print(input.shape)  #(1024, 1024, 3)

    '''
    use ori process 
    '''
    from transformers import  SamProcessor,SamImageProcessor
    config_processor = SamImageProcessor()
    processor = SamProcessor(config_processor)
    input = np.array(processor(image)['pixel_values']) # numpy
    input_points = np.array([[[[450, 600]]]])

    # predict  
    result = sam.predict(input,input_points)
    # predictor.set_image(image)

    # print(type(result))
    # calculate speed  

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='/home/sophgo/jingyu/SAM-ViT/Github-ori_Code/data/input/000000000632.jpg', help='path of input')
    parser.add_argument('--bmodel', type=str, default='/home/sophgo/jingyu/SAM-ViT/models_bmodel/models/BM1684X/sam-vit_f32.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')