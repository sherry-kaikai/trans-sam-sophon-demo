
## Ori model and class Link(hugging-face):

https://huggingface.co/docs/transformers/main/en/model_doc/sam
https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/image_processing_sam.py#L63
https://huggingface.co/facebook/sam-vit-base#model-details

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

- 预处理（有两步）：
    - SamProcessor 调用 SamImageProcessor 进行resize 。源码的resize指定了最长边是1024，另一边按比例缩放：输入是tensor([[1764, 2646]])变为([[ 683, 1024]]) ；但是bmodel中的输入shape是 [3，1024，1024];
    - tensor([[ 683, 1024]]) 经过SamVisionEncoder压缩变为 [1, 256, 64, 64] ；SamVisionEncoder在modeling_sam.py的forward()
- 推理：
    - 输入是 ['original_sizes', 'reshaped_input_sizes', 'input_points', 'image_embeddings']
        tensor([[1764, 2646]])
        torch.Size([1, 2])
        torch.Size([1, 1, 1, 2])
        torch.Size([1, 256, 64, 64])
    - 推理是outputs = model(**inputs)
    - 推理输出是['iou_scores', 'pred_masks']
        torch.Size([1, 1, 3]) torch.Size([1, 1, 3, 256, 256])


