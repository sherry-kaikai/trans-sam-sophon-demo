import numpy as np
import matplotlib.pyplot as plt

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  

def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()

def show_points_on_image(raw_image, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    plt.axis('on')
    plt.show()

def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()


def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_masks_on_image(raw_image, masks, scores):
    if len(masks.shape) == 4:
      masks = masks.squeeze()
    if scores.shape[0] == 1:
      scores = scores.squeeze()

    nb_predictions = scores.shape[-1]
    fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))

    for i, (mask, score) in enumerate(zip(masks, scores)):
      mask = mask.cpu().detach()
      axes[i].imshow(np.array(raw_image))
      show_mask(mask, axes[i])
      axes[i].title.set_text(f"Mask {i+1}, Score: {score.item():.3f}")
      axes[i].axis("off")
    plt.show()






import torch
'''
从transformer包中引入
'''
# from transformers import SamModel, SamProcessor,SamConfig

'''
从本地文件中引入
'''
from configuration_sam import SamConfig
from modeling_sam import SamModel
from processing_sam import SamProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
初始化模型 SamModel 默认是sam-vit-huge
'''
configuration = SamConfig()
model = SamModel(configuration)
configuration = model.config
print(type(model.config))

'''
初始化模型 SamModel sam-vit-base
'''
# model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)


'''
初始化预处理 SamImageProcessor sam-vit-base
'''
from transformers import SamImageProcessor
# processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

'''
初始化预处理 SamImageProcessor 默认是sam-vit-huge 
'''
# from image_processing_sam import SamImageProcessor #但是这里如果从本地导入会报错一些库包找不到
config_processor = SamImageProcessor()
processor = SamProcessor(config_processor)



'''
初始化图片
'''
from PIL import Image
import requests

img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")


'''
对图片预处理
'''
inputs = processor(raw_image, return_tensors="pt").to(device) # tensor
# inputs = processor(raw_image) # numpy

print('inputs',inputs.keys(),inputs["pixel_values"].shape) # torch.Size([1, 3, 1024, 1024])
print(inputs["original_sizes"],inputs["reshaped_input_sizes"]) # tensor([[1764, 2646]]) tensor([[ 683, 1024]])

# image_embeddings = model.get_image_embeddings(inputs["pixel_values"])
# print(type(image_embeddings),image_embeddings.shape) # [1, 256, 64, 64]


'''
推理
'''
outputs = model(**inputs)


'''
后处理
'''
masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores


print(scores) # tensor([[[ 0.0049,  0.0243, -0.0214]]], grad_fn=<SliceBackward0>)