# AttnGrounder: Talking to Cars with Attention
Code for the [paper](https://arxiv.org/pdf/2009.05684.pdf) AttnGrounder: Talking to Cars with Attention by Vivek Mittal.

## Model Overview
![complete_model](complete_model.png "AttnGrounder Complete Model")

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
![attention_map](examples_img_final.png "Attention Map")

### Credits
Part of the code or models are from 
[DMS](https://github.com/BCV-Uniandes/DMS),
[MAttNet](https://github.com/lichengunc/MAttNet),
[Yolov3](https://pjreddie.com/darknet/yolo/), 
[Pytorch-yolov3](https://github.com/eriklindernoren/PyTorch-YOLOv3) and
[One Stage Grounding](https://github.com/zyang-ur/onestage_grounding).
