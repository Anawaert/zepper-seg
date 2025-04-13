# zepper-seg
## 项目简介 <br/> Introduction
zepper-seg 是一个基于 Ultralytics YOLOv8-seg / YOLO11-seg 实例分割模型的双目视觉辣椒检测与定位项目，旨在使用深度学习与双目视觉技术完成对辣椒的采摘点精确定位。"zepper" 一词由 "zed" 与 "pepper" 组合而成，意为 "双目视觉辣椒"。项目使用 `Python` 作为主要编程语言，因此理论上可以在 Windows、Linux 和 MacOS 等操作系统上运行。本项目的目标是实现实时、高效且准确的辣椒检测与定位，帮助广大农业工作者提高采摘辣椒的效率和准确性。

zepper-seg is a stereo vision pepper detection and positioning project based on the Ultralytics YOLOv8-seg / YOLO11-seg instance segmentation model, aiming to achieve accurate positioning of chili picking points using deep learning and stereo vision technology. The term "zepper" is a combination of "zed" and "pepper", meaning "stereo vision chili". The project uses `Python` as the main programming language, so it can theoretically run on operating systems such as Windows, Linux, and MacOS. The goal of this project is to achieve real-time, efficient, and accurate chili detection and positioning, helping agricultural workers improve the efficiency and accuracy of chili picking.

## 效果示意图 <br/> Effect Demonstration

![zepper-seg Sample 1](https://github.com/Anawaert/zepper-seg/blob/master/images/sample1.png?raw=true)

## 快速开始 <br/> Quick Start
### 克隆本项目 <br/> Clone This Project
使用以下命令克隆本项目 - Use the following command to clone this project:

```bash
git clone https://github.com/Anawaert/zepper-seg.git
```

### 安装项目依赖 <br/> Install Project Dependencies
使用以下命令安装项目依赖 - Use the following command to install project dependencies:

```bash
cd ./zepper-seg
pip install -r ./requirements.txt
```

### 安装 ZED SDK <br/> Install ZED SDK
前往 [ZED SDK 官网](https://www.stereolabs.com/developers/release/) 下载适合您操作系统的 ZED SDK 安装包，并根据 Stereolabs 提供的的官方文档进行安装 ZED SDK 的 `C++` 库与 `Python` 接口。请注意，需要将 `pyzed` 安装至与 zepper-seg 的依赖相同的解释器环境中。

若您使用 Ubuntu 20.04.6 LTS 操作系统，可以参阅 Anawaert 的 [如何安装 Stereolabs ZED SDK](https://blog.anawaert.tech/post/computer-vision/installation-of-zed-sdk/)（简体中文），此处不再赘述。

Visit the [ZED SDK official website](https://www.stereolabs.com/developers/release/) to download the ZED SDK installation package suitable for your operating system, and install the ZED SDK `C++` library and `Python` interface according to the official documentation provided by Stereolabs. Please note that you need to install `pyzed` in the same interpreter environment as the dependencies of zepper-seg.

If you are using Ubuntu 20.04.6 LTS, you can refer to Anawaert's [How to install Stereolabs ZED SDK](https://blog.anawaert.tech/post/computer-vision/installation-of-zed-sdk/) (Simplified Chinese), which will not be repeated here.

### 运行 `main.py` <br/> Run `main.py`
将 ZED 相机连接至计算机，确保相机可正常工作，并为 `main.py` 赋予可执行权限。

Connect the ZED camera to the computer, ensure that the camera is working properly, and grant executable permissions to `main.py`.

在终端中运行以下命令 - Run the following command in the terminal:

```bash
python ./main.py
```

若运行成功，您将看到 3 个窗口，分别为 ZED 相机的左视图、深度图像和实例分割的掩膜图像。在左视图画面中，您可以看到辣椒的目标检测与采摘点定位结果。在控制台中，您将看到辣椒采摘点在左视图中的坐标与辣椒采摘点相对于相机左目的三维坐标信息，以及一帧图像的处理时间与当前的 FPS。

在 3 个窗口的任一窗口获得焦点时，按下 P 键来暂停程序的运行，使用 R 键来恢复程序的运行，使用 Q 键来退出程序。

If the program runs successfully, you will see 3 windows, which are the left view of the ZED camera, the depth image, and the instance segmentation mask image. In the left view screen, you can see the target detection and picking point positioning results of the chili peppers. In the console, you will see the coordinates of the chili picking point in the left view and the three-dimensional coordinate information of the chili picking point relative to the left camera, as well as the processing time of a frame image and the current FPS.

When any of the 3 windows is focused, press P to pause the program, R to resume it, and Q to exit.

## 已知问题 <br/> Known Issues
* 由于 ZED 相机的捕获图像尺寸为 1080 × 1920，Ultralytics 会将图像填充至 1088 × 1920 来满足 32 的倍数要求，因此本程序暂时仅支持 1080p 分辨率的 ZED 相机，其他分辨率的输入可能会导致程序运行错误。 <br/> Due to the ZED camera's captured image size of 1080 × 1920, Ultralytics will pad the image to 1088 × 1920 to meet the requirement of multiples of 32, so this program currently only supports ZED cameras with a resolution of 1080p. Other input resolutions may cause the program to run incorrectly.
* 对模型路径进行了硬编码，如模型路径为 `./weights/*.pt`，请根据需要自行修改。 <br/> The model path is hard-coded, and the model path is `./weights/*.pt`. Please modify it as needed.
* 暂未引入对检测目标的掩膜图像的校验机制，因此可能会出现检测获得的掩膜图像不完整或错误的情况。 <br/> The verification mechanism for the mask image of the detected target has not been introduced yet, so there may be cases where the detected mask image is incomplete or incorrect.
* ...

## 另请参阅 <br/> Extra Reference
* [Ultralytics Documentation](https://docs.ultralytics.com/) 
* [ZED SDK Documentation](https://www.stereolabs.com/docs)
* [NumPy API Reference](https://numpy.org/doc/stable/reference/index.html#reference)
* [OpenCV 4.9.0 API Reference](https://docs.opencv.org/4.9.0/)
* [Anawaert Blog](https://blog.anawaert.tech/)
