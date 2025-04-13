"""
本模块包含一系列用于实现辣椒采摘点 2D 坐标定位的函数和类。

This module contains a series of functions and classes for implementing the 2D coordinate localization of pepper picking points.
"""

import numpy as np
import cv2


def get_picking_point_2d(mask: np.ndarray, criteria_pixel: int = 3) -> tuple:
    """
    获取辣椒采摘点在掩膜上的 2D 坐标。

    Retrieve the 2D coordinates of the pepper picking point on the mask.

    :param mask: 输入的掩膜图像，应为灰度图。Input mask image, should be a grayscale image.
    :param criteria_pixel: 深度优先搜索的终止条件，默认为 3 个像素。Condition for terminating the depth-first search, default is 3 pixels.

    :return: 掩膜上的辣椒采摘点 2D 坐标。2D coordinates of the pepper picking point on the mask.
    """

    ########## 对掩膜进行处理，便于后续计算 - Process the mask for subsequent calculations

    # 对与输入的二值化掩膜，使用 OpenCV 获取最大联通区域相关的信息
    # For the input binary mask, use OpenCV to get information about the largest connected region
    num_labels, labeled_mask, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))

    # 获取最大联通区域的索引，stats[1:] 忽略背景区域
    # Get the index of the largest connected region, stats[1:] ignores the background region
    max_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1

    # 获取最大联通区域的掩膜，并将其转换为 uint8 类型
    # Get the mask of the largest connected region, convert it to uint8 type
    largest_region = np.array(labeled_mask == max_label).astype(np.uint8) * 255

    # 保留掩膜中最大联通区域，将掩膜中其他区域置为 0（背景）
    # Keep the largest connected region in the mask and set other regions to 0 (background)
    mask_with_largest_region = largest_region * np.array(mask > 0).astype(np.uint8)

    # 将最大联通区域中的空洞填充为 255
    # Fill the holes in the largest connected region with 255
    contours, _ = cv2.findContours(mask_with_largest_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask_with_largest_region)
    cv2.drawContours(mask_filled, contours, -1, (255), thickness=cv2.FILLED)

    ##########

    ########## 获取掩膜的必要信息 - Get necessary information about the mask

    # 获取最大填区域掩膜的上下边界（以下简称掩膜）
    # Get the upper and lower boundaries of the largest filled region mask (hereinafter referred to as the mask)
    rows, cols = np.where(mask_filled == 255)
    mask_top = np.min(rows)
    mask_bottom = np.max(rows)

    # 从掩膜底部向上 2/3 处开始，以当前行中掩膜的最左和最右边界距离作为变量，向上计算至掩膜顶部
    # Start from 2/3 of the mask bottom, calculate the distance to the left and right boundaries of the mask in the current row

    # 计算初始点的行
    # Calculate the row of the initial point
    start_row = int(mask_bottom - (mask_bottom - mask_top) * (2 / 3))

    # widths 为与掩膜同高度的数组，存储每行中掩膜的宽度
    # delta_width_ratio 为与掩膜同高度的数组，存储当前行与上一行的宽度比率
    # edge_points 为与掩膜同高度的数组，存储每行中掩膜的最左和最右边界坐标
    # widths is an array with the same height as the mask, storing the width of the mask in each row
    # delta_width_ratio is an array with the same height as the mask, storing the width ratio of the current row to the previous row
    # edge_points is an array with the same height as the mask, storing the left and right boundary coordinates of the mask in each row
    widths: np.ndarray = np.zeros(mask_filled.shape[0], dtype=np.int32)
    delta_width_ratio: np.ndarray = np.zeros(mask_filled.shape[0], dtype=np.float32)
    edge_points: np.ndarray = np.zeros((mask_filled.shape[0], 2), dtype=np.int32)

    # 从当前行开始，手动循环计算每行中掩膜的最左和最右边界坐标与距离
    # Manually start from the current row, calculate the left and right boundary coordinates and distances of the mask in each row
    for row in range(start_row, (mask_top - 1), -1):
        # 获取当前行中所有掩膜的列坐标索引
        # Get all column index of the mask in the current row
        cols_indices = np.where(mask_filled[row, :] > 0)[0]
        # 若当前行中存在掩膜，即 cols_indices 不为空的时候
        # If there is a mask in the current row, i.e., cols_indices is not empty
        if len(cols_indices) > 0:
            # 获取当前行中掩膜的最左和最右边界
            # Get the left and right boundaries of the mask in the current row
            left_edge = np.min(cols_indices)
            right_edge = np.max(cols_indices)
            # 以左边界和右边界的距离作为当前行的宽度
            # Use the distance between the left and right boundaries as the width of the current row
            current_width = right_edge - left_edge + 1
            # 修改表中的当前行的宽度、最左和最右边界坐标
            # Modify the current row's width, left and right boundary coordinates in the table
            widths[row] = current_width
            edge_points[row, 0] = left_edge
            edge_points[row, 1] = right_edge
            # 计算当前行与上一行的宽度比率
            # Calculate the width ratio of the current row to the previous row
            if row < start_row:
                delta_width_ratio[row] = (widths[row] - widths[row + 1]) / widths[row + 1]

    # 查找下降率最快的行高
    # Find the row height with the fastest decrease
    # delta_width_ratio_min_height = np.argmin(np.diff(delta_width_ratio))
    delta_width_ratio_min_height = np.argmin(np.diff(delta_width_ratio[mask_top:mask_bottom])) + mask_top

    ##########

    ########## 以深度优先搜索的方式走到掩膜的顶部边界 - Step to the top boundary of the mask using depth-first search

    # 当前数据上下文的行与列坐标
    # Current data context's row and column coordinates
    temp_h, temp_w = -1, -1

    # 从左向右向掩膜 delta_width_ratio_min_height 行的中点移动，直至第一个掩膜边界或左右端点的中点
    # Move from left to right to the midpoint of the mask delta_width_ratio_min_height row until the first mask boundary or the midpoint of the left and right endpoints

    for col in range(edge_points[delta_width_ratio_min_height, 0], (edge_points[delta_width_ratio_min_height, 1] + 1)):
        # 使用 int 强转来确保 temp_h 和 temp_w 类型不会变为 np.int64
        # Use int to ensure that temp_h and temp_w types do not become np.int64
        temp_h = int(delta_width_ratio_min_height)
        temp_w = int(col)
        # 若当前列坐标超出掩膜的范围，即掩膜中部断开了，就跳出循环
        # If the current column coordinate exceeds the mask range, i.e., the middle of the mask is disconnected, break the loop
        if mask_filled[delta_width_ratio_min_height, col] == 0:
            # 减去多走的一列
            # Subtract the extra column
            temp_w -= 1
            break

    # 此时 temp_h 和 temp_w 分别为掩膜的 delta_width_ratio_min_height 行 和 从左向右移动的第一个掩膜边界的坐标
    # At this point, temp_h and temp_w are the coordinates of the delta_width_ratio_min_height row of the mask and the first mask boundary moved from left to right

    # 计算 temp_h（也就是 delta_width_ratio_min_height）行的中点
    # Calculate the midpoint of temp_h (which is delta_width_ratio_min_height) row
    temp_centre = int((edge_points[temp_h, 0] + temp_w) / 2)

    # 再以 temp_h 行的中点为起始点，向上遍历掩膜，直到找到掩膜的顶部边界
    # Then, starting from the midpoint of temp_h row, traverse the mask upwards until the top boundary of the mask is found
    while True:
        for row in range(temp_h, (mask_top - 1), -1):
            # 若列坐标超出掩膜的范围
            # If the column coordinate exceeds the mask range
            if mask_filled[row, temp_centre] == 0:
                # 更新边界点所在的行高，要加上多退的一行
                # Update the row height of the boundary point, add one more row
                temp_h = int(row + 1)
                # 更新边界点所在的列坐标
                temp_w = int(temp_centre)
                break

        # 重新计算 temp_h 行的中点，继续按照从左向右的方式遍历掩膜
        # Recalculate the midpoint of temp_h row, continue to traverse the mask from left to right
        for col in range(edge_points[temp_h, 0], (edge_points[temp_h, 1] + 1)):
            temp_w = int(col)
            # 若当前列坐标超出掩膜的范围，即掩膜中部断开了，就跳出循环
            if mask_filled[temp_h, col] == 0:
                # 减去多走的一列
                # Subtract the extra column
                temp_w -= 1
                break

        # 计算当前行的中点
        # Calculate the midpoint of the current row
        temp_centre = int((edge_points[temp_h, 0] + temp_w) / 2)

        # 若当前行中的左右边界坐标的距离小于 criteria，则认为已经到达掩膜的顶部
        if abs(temp_centre - temp_w) or abs(temp_centre - edge_points[temp_h, 0]) < criteria_pixel:
            break

    # 返回掩膜的中点坐标 (x, y)
    return temp_centre, temp_h

    ##########
