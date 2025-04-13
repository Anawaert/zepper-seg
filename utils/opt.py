"""
本模块提供一些用于增删改查的工具函数。

This module provides some utility functions for CRUD operations.
"""

def get_dict_key_and_value_by_index(input_dict: dict, index: int) -> tuple:
    """
    获取字典中指定索引的键值对。

    Retrieve the key-value pair at the specified index in the dictionary.

    :param input_dict: 输入的字典。Input dictionary.
    :param index: 指定的索引。Index of the key-value pair.
    :return: 键值对元组：(key, value)。Tuple of key and value: (key, value).
    """
    # 将字典转换为列表
    # Convert the dictionary to a list
    dict_items = list(input_dict.items())

    # 获取指定索引的键值对
    # Get the key-value pair at the specified index
    key, value = dict_items[index]

    return key, value
