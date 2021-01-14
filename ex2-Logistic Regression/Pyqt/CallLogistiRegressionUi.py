import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from LogisticRegressionUi import Ui_Form

from RegulatedLogisticRegression import *

class MyMainWindow(QMainWindow, Ui_Form):
    # 初始化
    def __init__(self):
        super(MyMainWindow, self).__init__()
        self.setupUi(self)
        self.initUi()
        self.logisticWidget.setVisible(False)

    # 初始化窗口
    def initUi(self):
        scrollVal = self.lambdaHorizontalScrollBar.value()
        self.lambdaLabel.setText(str(scrollVal))

    # 槽函数
    def lambdaRateChange(self):
        print('* lambdaHorizontalScrollBar ---------')
        pos = self.lambdaHorizontalScrollBar.value()
        self.lambdaLabel.setText(str(pos))


    def updataLogisticRegression(self):
        power = self.powerSpinBox.value()
        lambdaRate = self.lambdaHorizontalScrollBar.value()
        return power, lambdaRate

    def showBound(self):
        power, lambdaRate = self.updataLogisticRegression()
        self.logisticWidget.setVisible(True)
        self.logisticWidget.mpl.start_static_plot(power, lambdaRate)

        val = accuracy(find_theta(power, lambdaRate), getX(power))
        self.accuracyLabel.setText(val)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    Win = MyMainWindow()
    Win.show()
    sys.exit(app.exec_())