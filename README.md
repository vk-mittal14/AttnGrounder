# AttnGrounder: Talking to Cars with Attention
[AttnGrounder: Talking to Cars with Attention ](https://arxiv.org/pdf/2009.05684.pdf) by Vivek Mittal. 

Accepted at [ECCV'20 C4AV Workshop](https://c4av-2020.github.io/). Talk2Car dataset used for this paper is available at https://talk2car.github.io/.

## Model Overview
![complete_model](static/complete_model.png "AttnGrounder Complete Model")

**Abstract:** <p>We propose Attention Grounder (AttnGrounder), a singlestage end-to-end trainable model for the task of visual grounding. Visual
grounding aims to localize a specific object in an image based on a given
natural language text query. Unlike previous methods that use the same
text representation for every image region, we use a visual-text attention
module that relates each word in the given query with every region in
the corresponding image for constructing a region dependent text representation. Furthermore, for improving the localization ability of our
model, we use our visual-text attention module to generate an attention mask around the referred object. The attention mask is trained as
an auxiliary task using a rectangular mask generated with the provided
ground-truth coordinates. We evaluate AttnGrounder on the Talk2Car
dataset and show an improvement of 3.26% over the existing methods.</p>

## Attention Map in Action
![attention_map](static/examples_img_final.png "Attention Map")

## Usage
Preprocessed Talk2Car data is available at this [link](https://drive.google.com/drive/folders/11R3VTHKErToa78qZ51vbIoGCHKsrfJLe?usp=sharing) extract it under `ln_data` folder. Download the images following instruction given at this [link](https://talk2car.github.io/). Extract all the images in `ln_data\images` folder. All the hyperparameters are set, just run the following command in working directory (if you face memory issue try decreasing the batch size).
```
python train_yolo.py --batch_size 14
```
### Credits
Part of the code or models are from 
[DMS](https://github.com/BCV-Uniandes/DMS),
[MAttNet](https://github.com/lichengunc/MAttNet),
[Yolov3](https://pjreddie.com/darknet/yolo/), 
[Pytorch-yolov3](https://github.com/eriklindernoren/PyTorch-YOLOv3) and
[One Stage Grounding](https://github.com/zyang-ur/onestage_grounding).
