# -*- encoding: utf-8 -*-    
"""
@Author     :   zYx.Tom
@Contact    :   526614962@qq.com
@site       :   https://zhuyuanxiang.github.io
---------------------------
@Software   :   PyCharm
@Project    :   introduction_to_ml_with_python
@File       :   config.py
@Version    :   v0.1
@Time       :   2020-08-09 17:18
@License    :   (C)Copyright 2018-2020, zYx.Tom
@Reference  :   
@Desc       :   配置文件
@理解：
"""
from preamble import *

# to make this notebook's output stable across runs
seed = 42
np.random.seed(seed)

tmp_path = '../temp/'

__all__ = ['np', 'mglearn', 'plt', 'pd', 'seed', 'tmp_path']
