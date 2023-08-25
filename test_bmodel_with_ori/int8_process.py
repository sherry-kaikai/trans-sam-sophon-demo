import time
import os
import numpy as np
import argparse
import json
import sophon.sail as sail
import math
import cv2 as cv2

engine_1b = sail.Engine(3)
path = "/data/mnt/sdb/jingyu.lu/yolov7_1b4b_bug_230725"

engine_1b.load(path + "/0815_xinlingxin/yolov7-lite-s_fp32_1b_x.bmodel")

graph_name_1b = engine_1b.get_graph_names()[0]
input_name_1b = engine_1b.get_input_names(graph_name_1b)[0]
output_name_1b = engine_1b.get_output_names(graph_name_1b)[0]
input_shape_1b = engine_1b.get_input_shape(graph_name_1b, input_name_1b)
engine_1b.set_io_mode(graph_name_1b, sail.IOMode.SYSIO)

bmcv = sail.Bmcv(engine_1b.get_handle()) 
input_dtype = engine_1b.get_input_dtype(graph_name_1b, input_name_1b)
# 
input = sail.Tensor(engine_1b.get_handle(), input_shape_1b, input_dtype, True, True)

# inputs=np.random.randint(-100,100,(input_shape_1b[1],input_shape_1b[2],input_shape_1b[3]))
# np.random.seed(0)

inputs = np.load(path+"/0815_xinlingxin"+"random.npy")


input_tensor_1b = {input_name_1b: np.expand_dims(inputs, axis=0)}
out_1b = engine_1b.process(graph_name_1b, input_tensor_1b)[output_name_1b]

print(out_1b)
print(np.shape(out_1b))

np.save(path+"/0815_xinlingxin"+"fp32.npy",out_1b)

