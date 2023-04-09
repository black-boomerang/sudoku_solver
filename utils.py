import numpy as np
from matplotlib import pyplot as plt


def show_image(image: np.ndarray, title: str = '') -> None:
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
