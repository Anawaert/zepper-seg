"""
本模块用于实现基于 YOLO11-PAS-seg 模型的双目视觉辣椒采摘点定位方法

This module is used to implement the stereo vision pepper picking point localization method based on YOLO11-PAS-seg model
"""

import time
import cv2
import numpy as np
import inference  # 用于推理 - Used for inference
import localisation  # 用于定位 - Used for localization
import utils

from ultralytics import YOLO
from pyzed import sl


def main():
    """
    主函数，将一系列功能进行整合

    Main function to integrate a series of functions
    :return: 0 为 标准结束，-1 为 相机无法打开，运行过程出错则无返回值。
        Return 0 for standard termination, -1 for camera not opening, no return value for runtime errors.
    """

    # 初始化 ZED 相机
    # Initialize ZED camera
    camera = sl.Camera()

    # 实例化ZED相机参数对象
    # Instantiate ZED camera parameter object
    init_params = sl.InitParameters()

    # 设置分辨率为 1080p，帧率为 30fps
    # Set resolution to 1080p and frame rate to 30fps
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 30

    # 设置深度模式为超高精度模式，深度单位为毫米，深度探测范围 0.30 - 3.00 m
    # Set depth mode to ultra high precision mode, depth unit to millimeters, and depth detection range to 0.30 - 3.00 m
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.depth_minimum_distance = 300
    init_params.depth_maximum_distance = 2500

    # 打开相机
    # Launch the camera
    camera_launch_status = camera.open(init_params)

    # 检查相机是否成功打开
    # Check if the camera is successfully opened
    if camera_launch_status == sl.ERROR_CODE.SUCCESS:
        print("Opened camera successfully")
    else:
        print("ERROR: Failed to open camera")
        exit(-1)

    # 创建相机运行时参数对象
    # Create camera runtime parameters object
    runtime_params = sl.RuntimeParameters()

    # 创建图像对象。ZED 使用 sl.Mat 来存储图像，并非为 NumPy 数组
    # Create image object. ZED uses sl.Mat to store images, not NumPy arrays
    left_image_sl = sl.Mat()
    # right_image_sl = sl.Mat()
    depth_image_sl = sl.Mat()
    point_cloud_matrix_sl = sl.Mat()

    # 创建 OpenCV 窗口，允许自由比例缩放
    # Create OpenCV window, allowing free aspect ratio scaling
    cv2.namedWindow("Left Image", cv2.WINDOW_FREERATIO)
    # cv2.namedWindow("Right Image", cv2.WINDOW_FREERATIO)
    cv2.namedWindow("Depth Image", cv2.WINDOW_FREERATIO)
    cv2.namedWindow("Mask Image", cv2.WINDOW_FREERATIO)

    # OpenCV 等待按键时间
    # OpenCV wait key time
    wait_key_span: int = 1

    # 加载 YOLO-seg 模型
    # Load YOLO-seg model
    yolo_seg_model = YOLO(r'models/peppers_seg_best.pt')

    # 在循环中不断执行功能
    # Continuously execute functions in a loop
    while True:
        # 获取当前时间
        # Get current time
        loop_start_time = time.time()
        # 成功获取一帧图像时
        # When successfully obtaining a frame of image
        if camera.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # 分别获取左图像、右图像、深度图像和点云矩阵
            # Retrieve left image, right image, depth image and point cloud matrix
            camera.retrieve_image(left_image_sl, sl.VIEW.LEFT)
            # camera.retrieve_image(right_image_sl, sl.VIEW.RIGHT)
            camera.retrieve_image(depth_image_sl, sl.VIEW.DEPTH)
            camera.retrieve_measure(point_cloud_matrix_sl, sl.MEASURE.XYZRGBA)

            # 将左、右图像与深度图像转换为 NumPy 数组，其中左、右目的图像为 RGB 图，深度图像为灰度图
            # Convert left, right images and depth image to NumPy arrays, where left and right images are RGB images and depth image is grayscale
            left_image_np = cv2.cvtColor(left_image_sl.get_data(), cv2.COLOR_RGBA2RGB)
            # right_image_np = cv2.cvtColor(right_image_sl.get_data(), cv2.COLOR_RGBA2RGB)
            depth_image_np = cv2.cvtColor(cv2.cvtColor(depth_image_sl.get_data(), cv2.COLOR_RGBA2RGB), cv2.COLOR_RGB2GRAY)

            # 预设筛选数值
            # Set preset filter values
            conf = 0.60
            iou = 0.45

            # 标注类别名称与颜色的字典
            # Dictionary of class names and colors
            cls_names: dict = {"Immature Green": (0, 200, 200), "Mature Green": (0, 200, 0), 'Red': (200, 0, 0)}

            # 分别获取左、右图像的最佳目标结果
            # Get the best target results for left and right images
            predict_result_left = inference.get_best_result(yolo_seg_model, left_image_np, (1080, 1920), conf, iou, '0')
            # predict_result_right = segmentation.get_best_result(yolo_seg_model, right_image_np, (1080, 1920), conf, iou, '0')

            # 若左、右图像的最佳目标结果为 None，则表示未能检测到有效目标
            # If the best target result for left and right images is None, it means no valid target was detected
            if predict_result_left is None:
                print(f'WARNING: Failed to detect target in left image with confidence: {conf} and IOU: {iou}')

                # 显示左右画面与深度图像
                # Display left and right images and depth image
                cv2.imshow("Left Image", left_image_np)
                # cv2.imshow("Right Image", right_image_np)
                cv2.imshow("Depth Image", depth_image_np)

                # 展示全黑的掩膜图像，尺寸与左右图像相同
                # Display a completely black mask image, same size as left and right images
                mask_image_zeros = np.zeros((left_image_np.shape[0], left_image_np.shape[1]), dtype=np.uint8)
                cv2.imshow("Mask Image", mask_image_zeros)

                # 输出本轮循环的耗时与 FPS
                # Output the time and FPS of this loop
                print(f'Consumed time: {time.time() - loop_start_time:.2f} s')
                print(f'FPS: {1 / (time.time() - loop_start_time):.2f}')

                # 设置 Q 键退出，P 键暂停，R 键恢复
                # Set Q key to exit, P key to pause, R key to resume
                pressed_key = cv2.waitKey(wait_key_span)
                if pressed_key == ord('q'):
                    break
                elif pressed_key == ord('p'):
                    wait_key_span = 0
                elif pressed_key == ord('r'):
                    wait_key_span = 1

                continue

            # 从结果中获取最佳目标的检测框、掩膜、置信度和类别
            # Get the best target's bounding box, mask, confidence and class from the result
            left_best_box: tuple = predict_result_left[0]
            # 将掩膜转换为 uint8 类型并且要处于 0-255 范围内以满足 get_picking_point_2d 函数的要求
            # Convert the mask to uint8 type and ensure it is in the range of 0-255 to meet the requirements of get_picking_point_2d function
            left_best_mask_np: np.ndarray = predict_result_left[1]
            left_best_conf: float = predict_result_left[2]
            left_best_cls: int = predict_result_left[3]

            # right_best_box: np.ndarray = predict_result_right[0]
            # right_best_mask: np.ndarray = predict_result_right[1]
            # right_best_conf: torch.Tensor = predict_result_right[2]
            # right_best_cls: torch.Tensor = predict_result_right[3]

            # 利用检测框和掩膜在左、右图像上绘制检测框和区域
            # 获取左图像的检测框坐标的左上角与右下角坐标
            # Draw the bounding box and area on the left and right images using the bounding box and mask
            # Compute the coordinates of the top-left and bottom-right corners of the bounding box in the left image
            left_x1 = int(left_best_box[0] - left_best_box[2] / 2)
            left_y1 = int(left_best_box[1] - left_best_box[3] / 2)
            left_x2 = int(left_best_box[0] + left_best_box[2] / 2)
            left_y2 = int(left_best_box[1] + left_best_box[3] / 2)
            # 获取右图像的检测框坐标
            # Compute the coordinates of the top-left and bottom-right corners of the bounding box in the right image
            # right_x1 = int(right_best_box[0] - right_best_box[2] / 2)
            # right_y1 = int(right_best_box[1] - right_best_box[3] / 2)
            # right_x2 = int(right_best_box[0] + right_best_box[2] / 2)
            # right_y2 = int(right_best_box[1] + right_best_box[3] / 2)

            # 获取最佳类别对应的颜色键值对
            # Get the key-value pair of the best class corresponding to the color
            left_best_cls_key, left_best_cls_value = utils.get_dict_key_and_value_by_index(cls_names, left_best_cls)
            # right_best_cls_key, right_best_cls_value = utils.get_dict_key_and_value_by_index(cls_names, int(right_best_cls))

            # 在左右图像上绘制检测框
            # Draw the bounding box on the left and right images
            utils.draw_box_info(left_image_np, f'{left_best_cls_key}: {left_best_conf}',
                                left_best_cls_value, left_x1, left_y1, left_x2, left_y2)
            # utils.draw_label(right_image_np, f'{right_best_cls_key}: {float(right_best_conf)}',
            #                   right_best_cls_value, right_x1, right_y1, right_x2, right_y2)

            # 获取左图像的中的采摘点坐标
            # Get the coordinates of the picking point in the left image
            left_picking_point_2d = localisation.get_picking_point_2d(left_best_mask_np, 3)
            # right_picking_point_2d = localisation.get_picking_point_2d(right_best_mask, 3)

            # 获取左目对应的三维坐标
            # Get the corresponding 3D coordinates of the left eye
            left_picking_point_3d = localisation.get_picking_point_3d(point_cloud_matrix_sl,
                                                                      (depth_image_np.shape[0], depth_image_np.shape[1]),
                                                                      left_picking_point_2d)
            # right_picking_point_3d = localisation.get_picking_point_3d(point_cloud_matrix_sl,
            #                                                            (depth_image_np.shape[0], depth_image_np.shape[1]),
            #                                                            right_picking_point_2d)

            # 在左右图上绘制采摘点圆圈加十字，使用对应类别的颜色
            # Draw a circle and cross at the picking point on the left and right images using the corresponding class color
            cv2.circle(left_image_np, left_picking_point_2d, 5, left_best_cls_value, 2)
            cv2.line(left_image_np, (left_picking_point_2d[0] - 8, left_picking_point_2d[1]),
                     (left_picking_point_2d[0] + 8, left_picking_point_2d[1]), left_best_cls_value, 1)
            cv2.line(left_image_np, (left_picking_point_2d[0], left_picking_point_2d[1] - 8),
                        (left_picking_point_2d[0], left_picking_point_2d[1] + 8), left_best_cls_value, 1)

            # cv2.circle(right_image_np, right_picking_point_2d, 5, right_best_cls_value, -1)
            # cv2.line(right_image_np, (right_picking_point_2d[0] - 8, right_picking_point_2d[1]),
            #          (right_picking_point_2d[0] + 8, right_picking_point_2d[1]), right_best_cls_value, 1)
            # cv2.line(right_image_np, (right_picking_point_2d[0], right_picking_point_2d[1] - 8),
            #          (right_picking_point_2d[0], right_picking_point_2d[1] + 8), right_best_cls_value, 1)

            # 显示绘制了检测框和采摘点的左右图像、深度图像和掩膜图像
            # Display the left and right images, depth image and mask image with bounding box and picking point
            cv2.imshow("Left Image", left_image_np)
            # cv2.imshow("Right Image", right_image_np)
            cv2.imshow("Depth Image", depth_image_np)
            cv2.imshow("Mask Image", left_best_mask_np)

            # 打印二维点坐标
            # Print 2D point coordinates
            print(f'2D picking point in left image: {left_picking_point_2d}')

            # 打印三维点坐标
            # Print 3D point coordinates
            print(f'3D picking coordinates in left image: {left_picking_point_3d}')

            # 输出本轮循环的耗时与 FPS
            # Output the time and FPS of this loop
            print(f'Consumed time: {time.time() - loop_start_time:.2f} s')
            print(f'FPS: {1 / (time.time() - loop_start_time):.2f}')
        else:
            print("ERROR: Failed to grab image, please check the camera connection")
            break

        # 设置 Q 键退出，P 键暂停，R 键恢复
        # Set Q key to exit, P key to pause, R key to resume
        pressed_key = cv2.waitKey(wait_key_span)
        if pressed_key == ord('q'):
            break
        elif pressed_key == ord('p'):
            wait_key_span = 0
        elif pressed_key == ord('r'):
            wait_key_span = 1

    # 关闭相机并关闭所有窗口
    # Close the camera and all windows
    camera.close()
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    main()
