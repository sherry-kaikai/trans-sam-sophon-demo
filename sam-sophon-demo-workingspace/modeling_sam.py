import os
import json
import time
import cv2
import argparse
import numpy as np
import sophon.sail as sail
import logging
logging.basicConfig(level=logging.INFO)

class SamModel:
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(args.bmodel))

        self.graph_name = self.net.get_graph_names()[0]
        self.input_names = self.net.get_input_names(self.graph_name) 
        self.output_names = self.net.get_output_names(self.graph_name)

        self.input_name_0 = self.input_names[0]
        self.input_name_1 = self.input_names[1]
        self.input_shape_0 = self.net.get_input_shape(self.graph_name, self.input_name_0) # [ 1 3 1024 1024 ] 
        self.input_shape_1 = self.net.get_input_shape(self.graph_name, self.input_name_1) # [ 1 1 1 2 ] 

        self.batch_size_0 = self.input_shape_0[0]
        self.net_h_0 = self.input_shape_0[2]
        self.net_w_0 = self.input_shape_0[3]
        
        
        # self.postprocess = PostProcess(
        #     conf_thresh=self.conf_thresh,
        #     nms_thresh=self.nms_thresh,
        #     agnostic=self.agnostic,
        #     multi_label=self.multi_label,
        #     max_det=self.max_det,
        # )
        
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0
    
    def init(self):
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def predict(self, input_img, input_points):
        print(type(input_img),type(input_points))
        print(input_img.shape,input_points.shape)
        # input_data_0 = {self.input_name_0: input_img}
        # input_data_1 = {self.input_name_1: input_points}
        input_data_0 = {self.input_name_0: input_img, self.input_name_1: input_points}
        outputs = self.net.process(self.graph_name, input_data_0)
        
        return outputs
    