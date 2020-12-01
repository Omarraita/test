# -*- coding: utf-8 -*-
"""detr_eval.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-hnbRjxP25ZNCthqNQfeACpj3rSSUEHI
"""

# Commented out IPython magic to ensure Python compatibility.
import math
import pickle
import time

import sys
from PIL import Image
import requests
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'retina'
import ipywidgets as widgets
from IPython.display import display, clear_output
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T

torch.set_grad_enabled(False);

#Add directory to detr
#sys.path.append('/content/detr')
sys.path.append('/home/mlteam/eval/test/detr')

from hubconf import detr_resnet50, detr_resnet50_dc5, detr_resnet101, detr_resnet101_dc5


def evaluate_time(models, models_name, img, repeats):
  i=0
  average_time = 0
  for model_ in models:
    model_.eval()
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
      #input_batch = input_batch.to('cuda')
      model_.to('cuda')
    times = []
    for j in range(repeats):
      #start evaluation
      start_time = time.time()
      img = transform(im).unsqueeze(0)
      if torch.cuda.is_available():
            img.to('cuda')
      # propagate through the model
      outputs = model_(img)
      # keep only predictions with 0.7+ confidence
      probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
      keep = probas.max(-1).values > 0.9
      times.append(time.time() - start_time)
      #stop evaluation
    file_name = 'my_file.txt'
    f = open(file_name, 'a+')
    f.write('Average time for model '+models_name[i]+' is: ')
    f.write(str(sum(times)/len(times))+'\n')
    i = i+1
  f.close()
%env JOBLIB_TEMP_FOLDER=/tmp
detr_resnet50 = detr_resnet50(True,91,False)
detr_resnet50_dc5 = detr_resnet50_dc5(True,91,False)
detr_resnet101 = detr_resnet101(True,91,False)
detr_resnet101_dc5 = detr_resnet101_dc5(True,91,False)

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
models = [detr_resnet50,detr_resnet50_dc5,detr_resnet101,detr_resnet101_dc5 ]
models_name = ['detr_resnet50','detr_resnet50_dc5','detr_resnet101','detr_resnet101_dc5' ]
im = Image.open('cats.jpg')
repeats = 3
evaluate_time(models,models_name,im,repeats)

