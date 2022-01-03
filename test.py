from __future__ import print_function, division

import torch
from torchvision import transforms
import os, glob, cv2
from PIL import Image
from parameter import *
from model import *

## test on CPU
net_test = Net()
PATH = 'checkpoint/checkpoint_50.pth'
net_test.load_state_dict(torch.load(PATH))

## testing
total = 0
for name in glob.glob('./test_images/*.png'):
    print(name)
    savename = os.path.split(name)[-1]
    img_raw = Image.open(name)

    img = transforms.ToTensor()(img_raw)
    img = torch.unsqueeze(img, 0)
    output, _, __= net_test(img)

    out = torch.squeeze(output, 0)
    out = out.detach().numpy().transpose((1, 2, 0))
    result = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite('./restored_images/restore_' + savename, result*255)