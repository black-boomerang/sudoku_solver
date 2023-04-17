import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from PIL import Image
import numpy as np

from solver import solve_sudoku
from cell_cutting import get_cells


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def mask_all(values):
    mask = np.ones(10, dtype=bool)
    for i in values:
        mask = mask & (np.arange(10) != i)
    
    return mask
def process_line(line_digits, adjacent_set, line_logits):
    for j in range(1, 10):
        if (line_digits == j).sum() > 1:
            saved_indices = np.where(line_digits == j)[0]
            line_digits[saved_indices] = 0
            best_fitting = saved_indices[np.argmax(line_logits[saved_indices, j])]
            line_digits[best_fitting] = j
            for k in saved_indices:
                if k == best_fitting:
                    continue
                mask = mask_all(np.unique(np.append(line_digits, adjacent_set[k])))
                line_digits[k] = np.arange(10)[mask][np.argmax(line_logits[k][mask])]
    return line_digits

def context_preprocessing(digits, logits):
    for i in range(9):
        digits[i] = process_line(digits[i],
                                 [np.append(digits[:, j], 
                                    digits[(i // 3) * 3:(i // 3 + 1) * 3, (j // 3) * 3:(j // 3 + 1) * 3].flatten()) for j in range(9)],
                                 logits[i])
        digits[:, i] = process_line(digits[:, i],
                                    [np.append(digits[:, j], 
                                    digits[(j // 3) * 3:(j // 3 + 1) * 3, (i % 3) * 3:(i % 3 + 1) * 3].flatten()) for j in range(9)],
                                    logits[:, i])
        digits[(i // 3) * 3:(i // 3 + 1) * 3, (i % 3) * 3:(i % 3 + 1) * 3] = \
        process_line(digits[(i // 3) * 3:(i // 3 + 1) * 3, (i % 3) * 3:(i % 3 + 1) * 3].flatten(),
                     [np.append(digits[(i // 3) * 3 + j // 3], 
                        digits[:, (i % 3) * 3 + j % 3]) for j in range(9)],
                     logits[(i // 3) * 3:(i // 3 + 1) * 3, (i % 3) * 3:(i % 3 + 1) * 3].reshape(9, 10)).reshape((3, 3))
    return digits

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


if __name__ == "__main__":
    model = torch.load('checkpoint_final.pth')

    im = Image.open(f'../sudoku_data/v2_test/{sys.argv[0]}')
    images = list(map(rgb2gray, list(get_cells(np.array(im), out_image_side=128))))

    preds = []
    saved_logits = []

    for im in images:
        with torch.no_grad():
            logits = model(torch.FloatTensor(im).unsqueeze(0).unsqueeze(0).cuda() / 255)[0].cpu()
        preds.append(torch.argmax(logits).item())
        saved_logits.append(logits.detach().numpy())

    print(solve_sudoku(context_preprocessing(np.array(preds).reshape(9, 9), np.array(saved_logits).reshape((9, 9, 10)))))

