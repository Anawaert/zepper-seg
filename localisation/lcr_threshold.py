"""
本模块定义了 LCRThreshold 类，用于控制有效的 LCR（Largest Connected Region）阈值。

This module defines the LCRThreshold class, which is used to control the effective LCR (Largest Connected Region) threshold.
"""

class LCRThreshold:
    """
    LCRThreshold 类用于控制有效的 LCR（Largest Connected Region）阈值，如面积、宽高比和绝对尺寸等。

    The LCRThreshold class is used to control the effective LCR (Largest Connected Region) threshold, such as area, aspect ratio, and absolute size.
    """
    def __init__(self, area_threshold: int = 2000,
                 aspect_ratio_threshold: float = 0.1,
                 absolute_width_threshold: int = 50,
                 absolute_height_threshold: int = 50):
        """
        实例化一个 LCRThreshold 对象。

        Instantiate an LCRThreshold object.

        :param area_threshold: 掩膜面积阈值，默认为 2000 像素平方。Mask area threshold, default is 2000 pixels squared.
        :param aspect_ratio_threshold: 掩膜边界宽高比阈值，默认为 0.1。Aspect ratio threshold, default is 0.1.
        :param absolute_width_threshold: 掩膜的绝对宽度阈值，默认为 50 像素。Absolute width threshold, default is 50 pixels.
        :param absolute_height_threshold: 掩膜的绝对高度阈值，默认为 50 像素。Absolute height threshold, default is 50 pixels.
        """
        self.area_threshold = area_threshold
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.absolute_width_threshold = absolute_width_threshold
        self.absolute_height_threshold = absolute_height_threshold
