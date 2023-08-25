import time
import os
import numpy as np
import argparse
import json
import sophon.sail as sail
import math
import cv2 as cv2

path = "/data/mnt/sdb/jingyu.lu/yolov7_1b4b_bug_230725"

out_1b =np.load(path+"/0815_xinlingxin"+"fp32.npy")

out_4b = np.load(path+"/0815_xinlingxin"+"int8.npy")

max_num = min(len(out_1b), len(out_4b))
print(max_num)
abs_diff = (out_1b[:max_num, -6:] - out_4b[:max_num, -6:]) / out_1b[:max_num, -6:]
print(abs_diff)
print("diff_sum=", np.sum(abs_diff))
print("diff_mean=", np.mean(abs_diff))
print("diff_max=", np.max(abs_diff))

print("diff argmax=",np.argmax(abs_diff))