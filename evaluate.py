import os

import numpy as np
import torch
from PIL import Image
from scipy.special import softmax
from tqdm.auto import tqdm

from cell_cutting import get_cells
from solve import context_preprocessing, rgb2gray, Classifier


def evaluate(images_path: str):
    image_names = list(filter(lambda x: x.endswith('.jpg'), os.listdir(images_path)))
    correct_solutions = 0
    model = torch.load('checkpoint_7_epoch_13.pth')
    logits_1 = []
    logits_2 = []
    for image_name in tqdm(image_names):
        image_path = os.path.join(images_path, image_name)
        im = Image.open(image_path)
        with open(image_path.replace('.jpg', '.dat'), 'r', encoding='utf-8') as dat_f:
            dat_f.readline()
            dat_f.readline()
            correct_answer = np.array([line.split() for line in dat_f.readlines()]).astype(int)

        try:
            images = list(
                map(rgb2gray, list(get_cells(np.array(im), out_image_side=128, bin_threshold=2))))

            preds = []
            saved_logits = []

            for im in images:
                with torch.no_grad():
                    logits = model(torch.FloatTensor(im).unsqueeze(0).unsqueeze(0).cuda() / 255)[0].cpu()
                preds.append(torch.argmax(logits).item())
                saved_logits.append(logits.detach().numpy())

            recognized_sudoku = context_preprocessing(np.array(preds).reshape(9, 9),
                                                      np.array(saved_logits).reshape((9, 9, 10)))
            if (recognized_sudoku == correct_answer).all():
                sm = softmax(np.array(saved_logits).reshape((9, 9, 10)), axis=2).max(axis=2)
                logits_1.append(min(sm.std(axis=1).min(), sm.std(axis=0).min()))
                correct_solutions += 1
            else:
                print(image_name)
                print(recognized_sudoku)
                sm = softmax(np.array(saved_logits).reshape((9, 9, 10)), axis=2).max(axis=2)
                logits_2.append(min(sm.std(axis=1).min(), sm.std(axis=0).min()))

        except Exception as e:
            print(image_name)

    print(logits_1)
    print(logits_2)
    print(f'Accuracy: {(correct_solutions / len(image_names) * 100):.1f}%')


if __name__ == '__main__':
    evaluate('sudoku_dataset\\datasets\\v2_test')
