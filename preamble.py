import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from IPython.display import set_matplotlib_formats
from cycler import cycler

import mglearn
import mglearn.plot_helpers

set_matplotlib_formats('pdf', 'png')

# 不能正常显示，拷贝字体后需要删除"用户/.matplotlib/fontList.json"
# plt.rcParams.keys()
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['YaHei Consolas Hybrid']  # 用来正常显示中文标签
plt.rcParams['image.cmap'] = "viridis"
plt.rcParams['image.interpolation'] = "none"
plt.rcParams['legend.numpoints'] = 1
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['savefig.bbox'] = "tight"
plt.rcParams['savefig.dpi'] = 300
plt.rc('axes', prop_cycle=(
        cycler('color', mglearn.plot_helpers.cm_cycle.colors) +
        cycler('linestyle', ['-', '-', "--", (0, (3, 3)), (0, (1.5, 1.5))])))
# ----------------------------------------------------------------------
# mpl.rcParams['font.family'] = ['SimHei']
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# mpl.rcParams['font.sans-serif'] = ['YaHei Consolas Hybrid']
# mpl.rcParams['font.size'] = 9

# 设置数据显示的精确度为小数点后3位
np.set_printoptions(precision=3, suppress=True, threshold=np.inf, linewidth=200)

pd.set_option("display.max_columns", 8)
pd.set_option('precision', 2)

__all__ = ['np', 'mglearn', 'display', 'plt', 'pd']
