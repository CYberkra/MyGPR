#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Nov 7 15:40:15 2024


@author: 何海
"""

import numpy as np
# 导入读取数据的函数


def compensatingGain(
    infilename="",
    outfilename="",
    outimagename="",
    length_trace=60,
    Start_position=0,
    end_position=10.3125,
    Gainfunction=[1, 2, 4, 5],
):
    """
    Function declaration:
    ----------


    INPUT:
    ----------
    infilename        str,csv file of the data that needs to be processed
    outfilename       str,path to save the processed data
    outimagename      str,path to save the image
    length_trace      int,时间轴的最大值（采样的时窗大小）
    Start_position    int,空间轴的起始位置
    end_position      int,空间轴的终止位置
    Gainfunction      array,增益函数,假设输入数据矩阵为M*N,则Gainfunction的长度必须等于M
    OUTPUT:
    ----------
    Gaintrace  array,仅当K=0时才返回该数组，否则返回空数组
    twtt       float,one-dimensional array, timeline, the unit is (ns)
    x          float,one-dimensional array,space axis, the unit is (m)

    """


    # 延迟导入 read_file_data
    try:
        from read_file_data import readcsv, savecsv, save_image
    except ImportError:
        import sys
        import os
        # 在打包环境中添加父目录到路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from read_file_data import readcsv, savecsv, save_image
    # 错误码及错误信息
    error_sign = 0
    error_feedback = ""

    # 读取数据矩阵

    data = readcsv(infilename)

    # 创建时间轴
    twtt = np.linspace(0, length_trace, data.shape[0])
    # 创建空间轴
    x = np.linspace(Start_position, end_position, data.shape[1])

    # 将数组转换成np数组
    Gainfunction = np.array(Gainfunction)

    try:
        real_Gainfunction = 10 ** (Gainfunction / 20)

        Gain = np.tile(real_Gainfunction, (data.shape[1], 1)).T
        Gaindata = data * Gain

        # 将数据保存至指定路径的csv格式
        savecsv(Gaindata, outfilename)
        # 将数据保存为图像并指定路径
        if outimagename:
            save_image(
                Gaindata,
                outimagename,
                "Data[CompensatingGain]",
                time_range=(twtt[0], twtt[-1]),
                distance_range=(x[0], x[-1]),
                contrast=4,
            )

    except Exception as e:
        error_sign = 1
        error_feedback = str(e)

    loc_data = {
        "x": x.tolist(),
        "twtt": twtt.tolist(),
        "error_sign": error_sign,
        "error_feedback": error_feedback,
    }
    return loc_data


def GainPreview(
    infilename=r"F:\dradar\signal_processing_code\data_output\cpp_call_python_function/EXAMPLE1_DZT_IMAGEDATA.csv",
    length_trace=60,
    Start_position=0,
    end_position=10.3125,
    Gainfunction=[1, 2, 4, 5],
    trace=0,
):
    """
    Function declaration:
    ----------


    INPUT:
    ----------
    infilename        str,csv file of the data that needs to be processed
    length_trace      int,时间轴的最大值（采样的时窗大小）
    Start_position    int,空间轴的起始位置
    end_position      int,空间轴的终止位置
    Gainfunction      array,增益函数,假设输入数据矩阵为M*N,则Gainfunction的长度必须等于M
    trace             int,表示要增益第trave道数据
    OUTPUT:
    ----------
    Gaintrace  array,返回该数组
    twtt       float,one-dimensional array, timeline, the unit is (ns)
    x          float,one-dimensional array,space axis, the unit is (m)

    """

    # 错误码及错误信息
    error_sign = 0
    error_feedback = ""

    # 读取数据矩阵

    data = readcsv(infilename)

    # 创建时间轴
    twtt = np.linspace(0, length_trace, data.shape[0])
    # 创建空间轴
    x = np.linspace(Start_position, end_position, data.shape[1])

    # 将数组转换成np数组
    Gainfunction = np.array(Gainfunction)
    Gaintrace = []
    # 检查 Gainfunction 长度和数据行数是否匹配

    try:
        real_Gainfunction = 10 ** (Gainfunction / 20)
        rawdata = data[:, trace]
        Gaintrace = rawdata * real_Gainfunction
        Gaintrace = Gaintrace.tolist()

    except Exception as e:
        error_sign = 1
        error_feedback = str(e)

    loc_data = {
        "Gaintrace": Gaintrace,
        "x": x.tolist(),
        "twtt": twtt.tolist(),
        "error_sign": error_sign,
        "error_feedback": error_feedback,
    }
    return loc_data
