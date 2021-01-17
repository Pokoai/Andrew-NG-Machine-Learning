import sys
import random
import matplotlib
import numpy as np

matplotlib.use("Qt5Agg")
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QSizePolicy, QWidget
from numpy import arange, sin, pi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from RegulatedLogisticRegression import *


# 设置绘图类
class MyMplCanvas(FigureCanvas):
    """FigureCanvas的最终的父类其实是QWidget。"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # 配置中文显示
        plt.rcParams['font.family'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        # 生成画布
        self.fig = Figure(figsize=(width, height), dpi=dpi)  # 新建一个figure
        self.axes = self.fig.add_subplot(111)  # 建立一个子图，如果要建立复合图，可以在这里修改

        #self.axes.hold(False)  # 每次绘图的时候不保留上一次绘图的结果

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        '''定义FigureCanvas的尺寸策略，这部分的意思是设置FigureCanvas，使之尽可能的向外填充空间。'''
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    '''绘制静态图，可以在这里定义自己的绘图逻辑'''

    # def start_static_plot(self):
    #     self.fig.suptitle('测试静态图')
    #     t = arange(0.0, 3.0, 0.01)
    #     s = sin(2 * pi * t)
    #     self.axes.plot(t, s)
    #     self.axes.set_ylabel('静态图：Y轴')
    #     self.axes.set_xlabel('静态图：X轴')
    #     self.axes.grid(True)
    #     self.draw()

    # 画决策边界
    # def draw_boundary(power, lambdaRate):
    def start_static_plot(self, power, lambdaRate):
        density = 200
        threshhold = 2 * 10 ** (-3)

        theta = find_theta(power, lambdaRate)
        x, y, xx, yy, z = find_decision_boundary(density, power, theta, threshhold)

        df = pd.read_csv(path, header=None, names=['test1', 'test2', 'labels'])
        pos = df.loc[data2['labels'] == 1]
        neg = df.loc[data2['labels'] == 0]

        self.axes.scatter(pos.test1, pos.test2, s=50, c='b', marker='o', label='1')
        self.axes.scatter(neg.test1, neg.test2, s=50, c='g', marker='x', label='0')
        # self.axes.scatter(x, y, s=50, c='r', marker='.', label='Decision Boundary')
        # 画等高线图
        self.axes.contour(xx, yy, z, 0)

        self.axes.set_xlabel('Microchip Test1')
        self.axes.set_ylabel('Microchip Test2')

        self.draw()
        # # 清除上一次图像
        # self.axes.cla()
        #
        # density = 200
        # threshhold = 2 * 10 ** (-3)
        #
        # theta = find_theta(power, lambdaRate)
        #
        # x, y, xx, yy, z = find_decision_boundary(density, power, theta, threshhold)
        #
        # df = pd.read_csv(path, header=None, names=['test1', 'test2', 'labels'])
        # pos = df.loc[data2['labels'] == 1]
        # neg = df.loc[data2['labels'] == 0]
        #
        # self.axes.scatter(pos.test1, pos.test2, s=50, c='b', marker='o', label='1')
        # self.axes.scatter(neg.test1, neg.test2, s=50, c='g', marker='x', label='0')
        # self.axes.scatter(x, y, s=50, c='r', marker='.', label='Decision Boundary')
        # # # 画等高线图
        # # self.axes.contour(xx, yy, z, 0)
        #
        # self.axes.set_xlabel('Microchip Test1')
        # self.axes.set_ylabel('Microchip Test2')
        #
        # self.draw()


    '''启动绘制动态图'''

    def start_dynamic_plot(self, *args, **kwargs):
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_figure)  # 每隔一段时间就会触发一次update_figure函数。
        timer.start(1000)  # 触发的时间间隔为1秒。


    '''动态图的绘图逻辑可以在这里修改'''

    def update_figure(self):
        self.fig.suptitle('测试动态图')
        # 更新绘图数据
        l = [random.randint(0, 10) for i in range(4)]
        # 清除上一次图像
        self.axes.cla()
        # 绘图
        self.axes.plot([0, 1, 2, 3], l, 'r')
        self.axes.set_ylabel('动态图：Y轴')
        self.axes.set_xlabel('动态图：X轴')
        self.axes.grid(True)
        self.draw()

# 封装绘图类
class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)
        self.initUi()

    def initUi(self):
        self.layout = QVBoxLayout(self)
        self.mpl = MyMplCanvas(self, width=5, height=4, dpi=100)
        # self.mpl.start_static_plot(power=6, lambdaRate=1)
        # self.mpl.start_static_plot() #如果你想要初始化的时候就呈现静态图，请把这行注释去掉
        # self.mpl.start_dynamic_plot() # 如果你想要初始化的时候就呈现动态图，请把这行注释去掉
        self.mpl_ntb = NavigationToolbar(self.mpl, self)  # 添加完整的 toolbar

        self.layout.addWidget(self.mpl)
        self.layout.addWidget(self.mpl_ntb)

    # power = 6
    # lambdaRate = 1
# 测试
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = MatplotlibWidget()
    ui.mpl.start_static_plot(power=6, lambdaRate=1)  # 测试静态图效果
    # ui.mpl.start_dynamic_plot() # 测试动态图效果
    ui.show()
    sys.exit(app.exec_())