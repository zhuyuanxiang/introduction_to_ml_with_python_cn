# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   pydata-book
@File       :   tools.py
@Version    :   v0.1
@Time       :   2019-12-21 17:58
@License    :   (C)Copyright 2018-2019, zYx.Tom
@Reference  :   《利用 Python 进行数据分析，Wes McKinney》
@Desc       :   常用的工具函数
@理解：
"""
import config
import matplotlib.pyplot as plt


def beep_end():
    # 运行结束的提醒
    import winsound
    winsound.Beep(600, 500)
    pass


def show_figures():
    # 运行结束前显示存在的图形
    if len(plt.get_fignums()) != 0:
        plt.show()
    pass


def show_title(message):
    # 输出运行模块的标题
    print('-' * 5, '>' + message + '<', '-' * 5)
    pass
