# -*- coding: utf-8 -*-
"""backbones_time_eval

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WTzxb0hchEJmD1IZS7jSxKfn1ndtJWyn
"""

import torch
import torchvision.transforms as T
from torchvision.models import resnet50,resnet101, resnext50_32x4d
from PIL import Image
import time

#!pip install efficientnet_pytorch
from efficientnet_pytorch import EfficientNet

bb_resnet50 = resnet50()
bb_resnet101 = resnet101()
bb_resnext50_32x4d =resnext50_32x4d()
efficientnet_py = EfficientNet.from_pretrained('efficientnet-b0')
bb_resnet50.eval()
bb_resnet101.eval()
bb_resnext50_32x4d.eval()
efficientnet_py.eval()

backbones = [bb_resnet50, bb_resnet101, bb_resnext50_32x4d, efficientnet_py]
backbone_names = ['bb_resnet50', 'bb_resnet101', 'bb_resnext50_32x4d','efficientnet_py']

if torch.cuda.is_available():
  for backbone_ in backbones:
    backbone_.to('cuda')

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

im = Image.open('cats.jpg')

import matplotlib.pyplot as plt
from decimal import *

i=0
#torch.cuda.synchronize()
backbone_times = {}
for backbone_ in backbones:
  backbone_name = backbone_names[i]
  if torch.cuda.is_available():
    img = transform(im).unsqueeze(0).to('cuda')
    if(i==4):
      break
    times = []
    for j in range(20):
      
      start = torch.cuda.Event(enable_timing=True)
      end = torch.cuda.Event(enable_timing=True)

      # Record cuda time
      start.record()
      out = backbone_(img)
      end.record()

      # Waits for everything to finish running
      torch.cuda.synchronize()
      t = start.elapsed_time(end)

      a = round(Decimal(t), 7)
      times.append(a)
    backbone_times[backbone_name] = times
    #print(backbone_names[i])
    #print(time.time()- t0)
    i = i+1

fig, ax1 = plt.subplots()
ax1.set_xlabel('Runs')
ax1.set_ylabel('Time [ms]')
#plt.ylim(0, 0.2)
ax1.plot(backbone_times['bb_resnet50'], label='bb_resnet50')
ax1.plot(backbone_times['bb_resnet101'], label='bb_resnet101' )
ax1.plot(backbone_times['bb_resnext50_32x4d'], label ='bb_resnext50_32x4d')
ax1.plot(backbone_times['efficientnet_py'], label ='bb_efficientnet_py' )
ax1.legend(loc='upper right')
plt.savefig('backbone_times.png')
