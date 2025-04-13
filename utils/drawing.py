"""
本模块提供与图像上绘制相关的函数。

This module provides functions related to drawing on images.
"""

import cv2
import numpy as np

def draw_box_info(input_image: np.ndarray, label: str, scalar: tuple, x1: int, y1: int, x2: int, y2: int):
    """
    在图像上绘制检测框和标签。

    Draws a detection box and label on the image.

    :param input_image: 输入图像。Input image.
    :param label: 需要与检测框一起绘制的标签。Label to be drawn with the detection box.
    :param scalar: 颜色值：(R, G, B)。Color value: (R, G, B).
    :param x1: 左上角 x 坐标。Top-left x coordinate.
    :param y1: 左上角 y 坐标。Top-left y coordinate.
    :param x2: 右下角 x 坐标。Bottom-right x coordinate.
    :param y2: 右下角 y 坐标。Bottom-right y coordinate.

    :return: 成功绘制返回 True，否则返回 False。If successful, return True; otherwise, return False.
    """
    try:
        # 在 bounding box 的顶部绘制文字
        # Draw text at the top of the bounding box
        label_size: tuple = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        y1 = max(y1, label_size[1])

        # 左上角顶点坐标
        # Top-left vertex coordinates
        top_left_point: tuple = (x1, y1 - label_size[0][1])

        # 右下角顶点坐标
        # Bottom-right vertex coordinates
        bottom_right_point: tuple = (x2, y2)

        # 绘制指定颜色的矩形框
        # Draw a rectangle with the specified color
        cv2.rectangle(input_image, top_left_point, bottom_right_point, scalar, 2)

        # 在矩形框上绘制文字
        # Draw text on the rectangle
        cv2.putText(input_image, label, (x1, y1 + label_size[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    scalar, 2)

        # 指示绘制成功
        # Indicate successful drawing
        return True
    except Exception as draw_exception:
        # 反馈报错信息
        # Feedback error message
        print(f'Error: Errors occurred while drawing: {draw_exception}')
        return False
