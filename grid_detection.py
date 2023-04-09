import math
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import cv2 as cv
import numpy as np

from utils import show_image


class LineType(Enum):
    HORIZONTAL = 0
    VERTICAL = 1


class LineInfo:
    def __init__(self, line: Tuple[int, int, int, int]) -> None:
        self.x1, self.y1, self.x2, self.y2 = line
        self.theta = np.arccos(np.abs((self.x2 - self.x1) / math.hypot(self.x2 - self.x1, self.y2 - self.y1)))
        self.rho = self.x1 * np.cos(self.theta) + self.y1 * np.sin(self.theta)
        if self.theta < np.pi / 4:
            self.line_type = LineType.HORIZONTAL
        else:
            self.line_type = LineType.VERTICAL
        self.component = None

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2


def opposite_type(line_type: LineType) -> LineType:
    if line_type == LineType.HORIZONTAL:
        return LineType.VERTICAL
    return LineType.HORIZONTAL


def show_image_with_lines(image: np.ndarray, lines: List[LineInfo], title: str = '') -> None:
    image_copy = image.copy()

    for line in lines:
        cv.line(image_copy, (line.x1, line.y1), (line.x2, line.y2), (255, 0, 0), 4)

    show_image(image_copy, title)


def preprocess(image: np.ndarray) -> np.ndarray:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blured = cv.medianBlur(gray, 5)
    binarized = cv.adaptiveThreshold(blured, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    blured = cv.medianBlur(binarized, 5)
    return blured


def hough_transform(image: np.ndarray) -> List[LineInfo]:
    edges = cv.Canny(image, 0, 15, apertureSize=3)

    segments = cv.HoughLinesP(edges, 2, np.pi / 180, 120, minLineLength=50, maxLineGap=10).squeeze()
    lines_infos = []
    for segment in segments:
        lines_infos.append(LineInfo(segment))

    return lines_infos


def get_lines_mask(image: np.ndarray, lines_infos: List[LineInfo]) -> np.ndarray:
    lines_image = np.full(image.shape[:2], 0, dtype=np.uint8)
    for line in lines_infos:
        cv.line(lines_image, (line.x1, line.y1), (line.x2, line.y2), 255, 4)
    return lines_image


def find_biggest_component(image: np.ndarray, lines_infos: List[LineInfo]) -> List[LineInfo]:
    lines_image = get_lines_mask(image, lines_infos)
    _, labels = cv.connectedComponents(lines_image, 4, cv.CV_32S)

    components_sizes = defaultdict(int)
    for line in lines_infos:
        line.component = labels[line.center[1], line.center[0]]
        components_sizes[line.component] += 1

    biggest_component = max(components_sizes.items(), key=lambda x: x[1])[0]
    filtered_lines_infos = list(filter(lambda line: line.component == biggest_component, lines_infos))

    return filtered_lines_infos


def hv_split(lines_infos: List[LineInfo]) -> Tuple[List[LineInfo], List[LineInfo]]:
    h_lines = []
    v_lines = []
    for line in lines_infos:
        if line.line_type == LineType.HORIZONTAL:
            h_lines.append(line)
        else:
            v_lines.append(line)

    return h_lines, v_lines


def merge_lines(lines: List[LineInfo]) -> List[LineInfo]:
    line_type = lines[0].line_type
    if line_type == LineType.HORIZONTAL:
        dist = [line.center[1] for line in lines]
    else:
        dist = [line.center[0] for line in lines]

    groups = []
    cur_group = []
    cur_dist = -100
    for index in np.argsort(dist):
        if dist[index] - cur_dist > 15:
            groups.append(cur_group)
            cur_group = [lines[index]]
        else:
            cur_group.append(lines[index])
        cur_dist = dist[index]

    if len(cur_group) > 0:
        groups.append(cur_group)

    groups = groups[1:]
    center_lines = [group[len(group) // 2] for group in groups]

    return center_lines


def detect_grid(image: np.ndarray, show_steps: bool = False):
    # бинаризация
    binarized_image = preprocess(image)
    if show_steps:
        show_image(binarized_image, 'Препроцессинг')

    # вероятностное преобразование Хафа
    lines_infos = hough_transform(binarized_image)
    if show_steps:
        show_image_with_lines(image, lines_infos, 'Вероятностное преобразование Хафа')

    # поиск наибольшей связной компоненты из линий
    lines_infos = find_biggest_component(image, lines_infos)
    if show_steps:
        show_image_with_lines(image, lines_infos, 'Наибольшая компонента')

    # разбиение линий на горизонтальные и вертикальные
    h_lines, v_lines = hv_split(lines_infos)
    if show_steps:
        show_image_with_lines(image, h_lines, 'Горизонтальные сегменты')
        show_image_with_lines(image, v_lines, 'Вертикальные сегменты')

    # объединяем "близкие" линии
    h_lines = merge_lines(h_lines)
    v_lines = merge_lines(v_lines)

    return h_lines, v_lines
