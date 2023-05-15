from typing import Generator, Tuple

import cv2 as cv
import numpy as np

from grid_detection import detect_grid, LineInfo


def lines_intersection(h: LineInfo, v: LineInfo) -> Tuple[int, int]:
    if v.x1 == v.x2:
        return v.x1, int(h.y1 + (h.y2 - h.y1) * (v.x1 - h.x1) / (h.x2 - h.x1))

    if h.y1 == h.y2:
        y = h.y1
    else:
        k1 = (h.x2 - h.x1) / (h.y2 - h.y1)
        k2 = (v.x2 - v.x1) / (v.y2 - v.y1)
        y = int((h.x1 - v.x1 + k2 * v.y1 - k1 * h.y1) / (k2 - k1))

    x = int(v.x1 + (v.x2 - v.x1) * (y - v.y1) / (v.y2 - v.y1))
    return x, y


def get_cells(image: np.ndarray, out_image_side: int = 32, bin_threshold: float = 2) -> Generator[
    np.ndarray, None, None]:
    for bin_threshold in np.linspace(0.5, 10.0, 20):
        h_lines, v_lines = detect_grid(image, bin_threshold=bin_threshold)
        if len(h_lines) == 10 and len(v_lines) == 10:
            break

    # TODO: обработать случай, когда линий больше или меньше
    if len(h_lines) != 10 or len(v_lines) != 10:
        pass

    h_lines = sorted(h_lines, key=lambda l: (-l.x1 * (l.y2 - l.y1)) / (l.x2 - l.x1) + l.y1)
    v_lines = sorted(v_lines, key=lambda l: (-l.y1 * (l.x2 - l.x1)) / (l.y2 - l.y1) + l.x1)

    points = []
    for h_line in h_lines:
        points.append([])
        for v_line in v_lines:
            points[-1].append(lines_intersection(h_line, v_line))

    out_image_pts = np.float32([[0, 0], [out_image_side, 0], [0, out_image_side], [out_image_side, out_image_side]])
    for h in range(1, len(points)):
        for v in range(1, len(points[0])):
            pts = np.float32([points[h - 1][v - 1], points[h - 1][v], points[h][v - 1], points[h][v]])
            transform = cv.getPerspectiveTransform(pts, out_image_pts)
            yield cv.warpPerspective(image, transform, (out_image_side, out_image_side))
