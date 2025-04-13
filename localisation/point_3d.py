"""
本模块包含一系列用于实现辣椒采摘点 3D 坐标定位的函数和类。

This module contains a series of functions and classes for implementing the 3D coordinate localization of chili picking points.
"""

import numpy as np

from pyzed import sl
from typing import Optional


def get_picking_point_3d(point_cloud_matrix: sl.Mat, depth_image_size: tuple, point_2d: tuple) -> Optional[tuple[float, float, float]]:
    """
    获取辣椒采摘点在实际空间中的三维坐标。

    Retrieve the 3D coordinates of the chili picking point in the actual space.

    :param point_cloud_matrix: 由 ZED SDK 提供的点云矩阵。Point cloud matrix provided by ZED SDK.
    :param depth_image_size: 深度图像的大小。Size of the depth image.
    :param point_2d:  辣椒采摘点的二维坐标。2D coordinates of the chili picking point.

    :return: 辣椒采摘点在实际空间中的三维坐标点，在不可用时返回 None。
        3D coordinates of the chili picking point in the actual space, returns None if not available.
    """

    # 校验传入的二维坐标是否在深度图像的范围内
    # Check if the passed 2D coordinates are within the range of the depth image
    if point_2d[0] < 0 or point_2d[0] >= depth_image_size[0] or \
       point_2d[1] < 0 or point_2d[1] >= depth_image_size[1]:
        print(f'ERROR: Invalid 2D point: {point_2d} is out of depth image range: {depth_image_size}')

    # 获取向点云矩阵映射的三维坐标返回值
    # Retrieve the 3D coordinates mapped to the point cloud matrix
    err, original_coordinates = point_cloud_matrix.get_value(point_2d[0], point_2d[1])

    # 只有当 err 为 SUCCESS 时，且 x、y 值在 [-1500, 1500]，z 在 [30, 2000] 毫米范围内才会被认为是有效的三维坐标
    # Only when err is SUCCESS, and x, y values are in the range of [-1500, 1500], and z is in the range of [20, 2000] mm, it will be considered as valid 3D coordinates
    if (err == sl.ERROR_CODE.SUCCESS and
            1500 > original_coordinates[0] > -1500 and
            1500 > original_coordinates[1] > -1500 and
            2000 > original_coordinates[2] > 30
    ):
        return original_coordinates[0], original_coordinates[1], original_coordinates[2]
    else:
        print(f'ERROR: Invalid 3D coordinates: {original_coordinates}')
        return None

def get_point_distance(point_3d: tuple) -> Optional[float]:
    """
    计算三维坐标点到原点的欧式距离。

    Calculate the Euclidean distance from the 3D coordinate point to the origin.
    :param point_3d: 需要计算距离的三维坐标点，格式 (x, y, z)。3D coordinate point to calculate the distance, format (x, y, z).
    :return (float): 三维坐标点到原点的欧式距离，若传入的坐标点无效则返回 None。Euclidean distance from the 3D coordinate point to the origin, returns None if the passed coordinate point is invalid.
    """
    # 校验传入的三维坐标是否为空
    # Check if the passed 3D coordinates are None
    if point_3d is not None:
        # 获取点云矩阵中的三维坐标的深度值
        # Extract the depth values from the 3D coordinates
        x, y, z = point_3d
        # 若 z 坐标值小于 0，则返回 None
        # If the z-coordinate value is less than 0, return None
        if z < 0:
            return None
        else:
            # 计算并返回欧式距离
            # Calculate and return the Euclidean distance
            return np.sqrt(x ** 2 + y ** 2 + z ** 2)
