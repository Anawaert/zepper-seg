"""
本模块用于实现获取（YOLO11-PAS-seg 等可被 Ultralytics 解析的）实例分割模型的后处理推理结果。

This module is used to implement the post-processing inference results of instance segmentation models (such as YOLO11-PAS-seg) that can be parsed by Ultralytics.
"""
from typing import Optional, Any

import numpy as np
import torch

from ultralytics import YOLO

def get_best_result(model_object: YOLO,
                          image: np.ndarray,
                          image_size: tuple = (1080, 1920),
                          conf_threshold: float = 0.70,
                          iou_threshold: float = 0.70,
                          device: str = 'cpu') -> Optional[tuple[tuple[int, int, int, int], np.ndarray[Any, Any], float, int]]:
    """
    获取实例分割模型的最佳实例推理结果，包含 bounding box、掩膜、置信度和类别。

    Retrieves the best instance segmentation inference result from the YOLO11-PAS-seg model, including bounding box, mask, confidence, and class.

    :param model_object: 需要推理的模型对象。Model object to be inferred.
    :param image: 需要推理的图像。Image to be inferred.
    :param image_size: 输入时的图像大小。Image size during input.
    :param conf_threshold: 置信度阈值。Confidence threshold.
    :param iou_threshold: IOU 阈值。IOU threshold.
    :param device: 推理设备，"0"、"1" ... 时使用 GPU，"cpu" 使用 CPU 进行推理。Inference device, "0", "1" ... for GPU, "cpu" for CPU inference.

    :returns: 返回包含最佳实例的 bounding box、掩膜、置信度（保留两位小数）和类别（整数）的元组，若无实例则返回 None。
        Returns a tuple containing the bounding box, mask, confidence (rounded to two decimal places), and class (integer) of the best instance. Returns None if no instance is found.
    """

    # 前向推理
    # Perform forward inference
    results = model_object.predict(source=image, imgsz=image_size, conf=conf_threshold, iou=iou_threshold, device=device)

    # 判断预测结果是否为空
    # Check if the prediction result is empty
    if len(results) == 0:
        return None

    # 遍历处理结果对象的 bounding box 的成员，找到最佳目标对应的索引
    # Iterate through the bounding box members of the result object to find the index of the best target
    best_index: int = 0
    for i in range(len(results[0].boxes)):
        if results[0].boxes.conf[i] > results[0].boxes.conf[best_index]:
            best_index = i

    # 获取所有 bounding box 的 xywh 张量并转换为 NumPy 数组
    # Get all bounding box xywh tensors and convert them to NumPy arrays
    boxes_xywh_tensor: torch.Tensor = results[0].boxes.xywh
    boxes_xywh_np: np.ndarray = boxes_xywh_tensor.cpu().numpy()

    # 如果 bounding box 的数量为 0 或者为 None，则返回 None
    # If the number of bounding boxes is 0 or None, return None
    if len(boxes_xywh_np) == 0 or boxes_xywh_np is None:
        return None

    # 获取最佳目标的 bounding box 的中心坐标和宽高，全部处理为 int 类型
    # Get the center coordinates and width and height of the best target's bounding box, all processed as int type
    best_box = (int(boxes_xywh_np[best_index][0]),
                int(boxes_xywh_np[best_index][1]),
                int(boxes_xywh_np[best_index][2]),
                int(boxes_xywh_np[best_index][3]))

    # 获取最佳目标的置信度和类别，将置信度保留 2 位小数，将类别转换为 int 类型
    # Get the confidence and class of the best target
    best_conf: float = round(results[0].boxes.conf[best_index].item(), 2)
    best_cls: int = int(results[0].boxes.cls[best_index].item())

    # 获取所有掩膜的张量并转换为 NumPy 数组，并进行后处理为 0-255 的灰度图
    # Get all masks tensors and convert them to NumPy arrays, and post-process them to 0-255 grayscale images
    masks_tensor: torch.Tensor = results[0].masks.data
    masks_np: np.ndarray = np.array(masks_tensor.cpu().numpy()).astype(np.uint8) * 255

    # 如果掩膜中第一维的数量是 0 或掩膜为 None，则返回 None。掩膜形状为 (N, H, W)，其中 N 为掩膜数量
    # If the first dimension of the mask is 0 or the mask is None, return None. The mask shape is (N, H, W), where N is the number of masks
    if masks_np.shape[0] == 0 or masks_np is None:
        return None

    # 获取最佳目标的掩膜
    # Get the mask of the best target
    best_mask = masks_np[best_index]

    # TODO: 由于 Ultralytics 可能会对输入特征图进行缩放，因此需要将 bounding box 参数和掩膜缩放回原始图像的大小
    # TODO: 编写算法自动将 bounding box 和掩膜缩放回原始图像的大小

    # 对于 1920x1080 的图像而言，已知 Ultralytics 为满足 32 的倍数将输入图像分辨率改为 1088 × 1920
    # 所以需要将 bounding box 的 y 坐标减去 4 而宽高不变；将掩膜直接上下对称裁切掉 4 个像素
    # For 1920x1080 images, Ultralytics changes the input image resolution to 1088 × 1920 to satisfy the multiple of 32
    # So the y-coordinate of the bounding box needs to be reduced by 4 while the width and height remain unchanged;
    if image_size == (1080, 1920):
        best_box = (int(best_box[0]),
                    int(best_box[1] - 4),
                    int(best_box[2]),
                    int(best_box[3]))
        best_mask = best_mask[4:-4, :]
    # 对于其它的图像分辨率，暂时不支持缩放回原始图像大小
    # For other image resolutions, scaling back to the original image size is not supported for now
    else:
        print(f'WARNING: The output image size is {best_mask.shape[0], best_mask.shape[1]}, which is not supported.\n'
              f'The bounding box and mask will not be scaled back to the original image size.')

    return best_box, best_mask, best_conf, best_cls
