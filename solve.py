import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from PIL import Image
import numpy as np

from ? import solve_sudoku
from cell_cutting import get_cells

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(1, 4, 3, padding='same'), 
                      nn.Conv2d(4, 8, 3, padding='same'), 
                      nn.Conv2d(8, 16, 3, padding='same'),
                      nn.Conv2d(16, 32, 3, padding='same'), 
                      nn.Conv2d(32, 64, 3, padding='same')])
        self.bns = nn.ModuleList([nn.BatchNorm2d(4),
                                  nn.BatchNorm2d(8),
                                  nn.BatchNorm2d(16),
                                  nn.BatchNorm2d(32),
                                  nn.BatchNorm2d(64),])
        self.do = nn.Dropout()
        self.out_linear = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = F.leaky_relu(bn(conv(x)))
            x = F.max_pool2d(x, 2)
        x = torch.flatten(x, start_dim=1)
        return self.out_linear(self.do(x)), x

model = torch.load('checkpoint_final.pth')

im = Image.open(f'../sudoku_data/v2_test/{sys.argv[0]}')
images = list(map(rgb2gray, list(get_cells(np.array(im), out_image_side=128))))
preds = []
for im in images:
    with torch.no_grad():
        logits = model(torch.FloatTensor(im).unsqueeze(0).unsqueeze(0).cuda() / 255)[0].cpu()
    preds.append(torch.argmax(logits).item())

print(solve_sudoku(preds))
