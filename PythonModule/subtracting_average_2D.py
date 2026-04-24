#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Nov 9 15:40:15 2023

@author: 何海
"""

import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# from tqdm import tqdm
def subtracting_average_2D(
    infilename="",
    outfilename="",
    outimagename="",
    length_trace=48 - 5.7,
    Start_position=0,
    Scans_per_meter=50,
    ntraces=501,
):
    """
    Function declaration:
    ----------
    Subtracts from each trace the average trace over
    a moving average window.

    Can be used to remove horizontal arrivals,
    such as the airwave.

    INPUT:
    ----------
    infilename        str,csv file of the data that needs to be processed
    outfilename       str,path to save the processed data
    outimagename      str,path to save the image
    length_trace      int,零时间校正后的新的 length of trace [ns]
    Start_position    int,头信息中 Start position [ns] 的值
    Scans_per_meter   int, 头信息中 Scans per meter 的值,如果该值为0，则可以用头信息中 Scans per second 的值代替。
    ntraces           int, window width; over how many traces to take the moving average.
                      假设数据共采集了N道数据，则取值范围是[1,N];
                      例如FILE____032.DZT数据采了1040道数据，则ntraces的取值范围是[1,1040]

    OUTPUT:
    ----------
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
    # 读取数据矩阵、时间轴和空间轴信息
    data = readcsv(infilename)
    # 创建时间轴
    twtt = np.linspace(0, length_trace, data.shape[0])
    # 创建空间轴
    x = np.linspace(
        Start_position, Start_position + data.shape[1] / Scans_per_meter, data.shape[1]
    )

    # 错误码及错误信息
    error_sign = 0
    error_feedback = ""
    newdata = []
    if ntraces < 1:
        loc_data = {
            "data": newdata,
            "x": x.tolist(),
            "twtt": twtt.tolist(),
            "error_sign": 2,
            "error_feedback": "The ntraces value must >= 1 ",
        }
        return loc_data
    try:  # 函数主体部分
        from scipy.ndimage import uniform_filter1d

        data_f = np.array(data, dtype=float)
        if ntraces >= data_f.shape[1]:
            newdata = data_f - np.mean(data_f, axis=1, keepdims=True)
        else:
            background = uniform_filter1d(data_f, size=ntraces, axis=1, mode="nearest")
            newdata = data_f - background

        # 将数据保存至指定路径的csv格式
        savecsv(newdata, outfilename)
        # 将数据保存为图像并指定路径
        if outimagename:
            save_image(
                newdata,
                outimagename,
                "Data[subtracting_average_2D]",
                time_range=(twtt[0], twtt[-1]),
                distance_range=(x[0], x[-1]),
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


def main():
    # 指定包含DZT文件的文件夹路径，替换为你的DZT文件路径
    infilename = "F:\dradar\signal_processing_code\data_output\my_data\save_the_csv/agcGainFILE____032.csv"
    # 输出的文件路径
    outfilename = "F:\dradar\signal_processing_code\data_output\my_data\save_the_csv/Subtracting_averageFILE____032.csv"
    # 输出图像的路径
    outimagename = "F:\dradar\signal_processing_code\data_output\my_data\save_the_csv/Subtracting_averageFILE____032.png"
    # 输入指定的头信息内容
    length_trace = 48 - 5.7
    Start_position = 0
    Scans_per_meter = 50

    # 设置窗口大小(大于1的一个整数)
    ntraces = 501
    # 进行背景抑制处理
    loc_data = Subtracting_average_2D(
        infilename,
        outfilename,
        outimagename,
        length_trace,
        Start_position,
        Scans_per_meter,
        ntraces,
    )

    x = np.array(loc_data["x"])  # 空间轴
    twtt = np.array(loc_data["twtt"])  # 时间轴
    if loc_data["error_sign"] == 0:
        # 处理后图像显示
        newdata = readcsv(
            "F:\dradar\signal_processing_code\data_output\my_data\save_the_csv/Subtracting_averageFILE____032.csv"
        )
        show_image(
            newdata, time_range=(twtt[0], twtt[-1]), distance_range=(x[0], x[-1])
        )
    else:
        print(loc_data)


if __name__ == "__main__":
    from read_file_data import readcsv, show_image

    main()
