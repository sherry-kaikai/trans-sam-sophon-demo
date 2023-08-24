
## Ori model and class Link(hugging-face):

https://huggingface.co/docs/transformers/main/en/model_doc/sam
https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/image_processing_sam.py#L63
https://huggingface.co/facebook/sam-vit-base#model-details

https://huggingface.co/facebook/sam-vit-base

model-zoo中是sam-vit-base （hugging-face默认是sam-vit-huge）
## 算法背景


## TODO:
1. 本地运行transform-sam
2. config修改，成功加载bmodel
2. 处理输入为bmodel需要的输入 （预处理）
3. 输出后处理

## 目录结构
```bash
.
├── configuration_sam.py # SamConfig
├── image_processing_sam.py # SamImageProcess
├── modeling_sam.py # SamModel
├── packets # 需要引用的文件
├── processing_sam.py # SamProcess
├── __pycache__ 
├── readme.md 
├── SAM-opencv.py # 移植到bmodel的代码
├── segment_anything.ipynb # hugging-face 的sam运行例程 全流程
├── segment_anything.py # hugging-face 的sam运行例程 全流程
├── utils
└── utils.zip
```

## trans-sam代码处理方式：

- 预处理：
    - SAM的推理输入确实可以用原图，不用embed后再输入，如果是原图输入的话，SamModel的forward代码里面会再做embed然后传给模型的mask_decoder做处理. 
    - 模型是sam-vit-huge \ sam-vit-base 的时候 预处理的图片输出 pixel_values torch.Size([1, 3, 1024, 1024]) 
        - SamProcessor 调用 SamImageProcessor 进行resize 。resize指定了最长边是1024。

```
print(inputs.keys()) # array
print(inputs.pixel_values.shape,inputs["original_sizes"],inputs["reshaped_input_sizes"]) 

dict_keys(['pixel_values', 'original_sizes', 'reshaped_input_sizes'])
torch.Size([1, 3, 1024, 1024]) tensor([[1764, 2646]]) tensor([[ 683, 1024]])
```
##### Q1： 为什么reshaped_input_sizes和实际的pixel_values的shape不一致？
##### Q2: sam-vit-base 于sam-vit-huge区别是什么？


- embedding 做了什么 “压缩，嵌入”
    - “图嵌入（Graph Embedding，也叫Network Embedding）是一种将图数据（通常为高维稠密的矩阵）映射为低微稠密向量的过程，能够很好地解决图数据难以高效输入机器学习算法的问题”
    - tensor([[ 683, 1024]]) 经过SamVisionEncoder压缩变为 [1, 256, 64, 64] ；SamVisionEncoder在modeling_sam.py的forward()


- 推理：
    - 经过embedding后输入是 ['original_sizes', 'reshaped_input_sizes', 'input_points', 'image_embeddings']
        tensor([[1764, 2646]])
        torch.Size([1, 2])
        torch.Size([1, 1, 1, 2])
        torch.Size([1, 256, 64, 64])
    - sam-vit-base，不经过embedding的输入是 pixel_values([1, 3, 1024, 1024]) 
    - 推理是outputs = model(**inputs)
    - 推理输出是['iou_scores', 'pred_masks'] 是和bmodel对齐的
        torch.Size([1, 1, 3]) torch.Size([1, 1, 3, 256, 256])


