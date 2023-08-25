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
    use ori process 使用hugging-face transformer的源码预处理
    '''
    from transformers import  SamProcessor,SamImageProcessor
    config_processor = SamImageProcessor()
    processor = SamProcessor(config_processor)
    input = np.array(processor(image)['pixel_values']) # numpy
    input_points = np.array([[[[302, 300]]]])

    # predict  
    result = sam.predict(input,input_points)
    iou_scores_Slice = result['iou_scores_Slice']
    pred_masks_Slice = result['pred_masks_Slice']


    print(type(iou_scores_Slice),type(pred_masks_Slice))
    print(iou_scores_Slice.shape,pred_masks_Slice.shape) #(1, 1, 3) (1, 1, 3, 256, 256)
    print(iou_scores_Slice)


    '''
    保存输出
    '''
    # masks = processor.image_processor.post_process_masks(pred_masks_Slice,[[480, 640]], [1, 1, 1, 2])
    np.save("/home/sophgo/jingyu/SAM-ViT/sophon-demo-trans-Sam/test_bmodel_with_ori/"+"pred_masks.npy",pred_masks_Slice)

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='/home/sophgo/jingyu/SAM-ViT/sophon-demo-trans-Sam/000000397639.jpg', help='path of input')
    parser.add_argument('--bmodel', type=str, default='/home/sophgo/jingyu/SAM-ViT/models_bmodel/models/BM1684X/sam-vit_f32.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')