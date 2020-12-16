
"""
This script computes the evaluation times on a single image from the coco dataset for different backbones, plots and saves the results over 20 runs.
"""

import torch
import torchvision.transforms as T
from torchvision.models import resnet50,resnet101, resnext50_32x4d
from PIL import Image
import time
import matplotlib.pyplot as plt
from decimal import *
#Make sure to have efficientnet_pytorch installed
from efficientnet_pytorch import EfficientNet

# Loading pretrained backbones
bb_resnet50 = resnet50()
bb_resnet101 = resnet101()
bb_resnext50_32x4d =resnext50_32x4d()
efficientnet_py = EfficientNet.from_pretrained('efficientnet-b0')
bb_resnet50.eval()
bb_resnet101.eval()
bb_resnext50_32x4d.eval()
efficientnet_py.eval()

# Defining list of backbone instances and corresponding names
backbones = [bb_resnet50, bb_resnet101, bb_resnext50_32x4d, efficientnet_py]
backbone_names = ['bb_resnet50', 'bb_resnet101', 'bb_resnext50_32x4d','efficientnet_py']

# Check if cuda is available
if torch.cuda.is_available():
  for backbone_ in backbones:
    backbone_.to('cuda')

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Coco image
im = Image.open('coco_image.jpg')

# Start the evaluation for each backbone
i=0
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
    i = i+1
# Plot the results
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

