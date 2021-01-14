# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'LogisticRegressionUi.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(740, 475)
        self.logisticWidget = MatplotlibWidget(Form)
        self.logisticWidget.setGeometry(QtCore.QRect(10, 70, 721, 401))
        self.logisticWidget.setObjectName("logisticWidget")
        self.powerSpinBox = QtWidgets.QSpinBox(Form)
        self.powerSpinBox.setGeometry(QtCore.QRect(140, 10, 121, 20))
        self.powerSpinBox.setMinimum(1)
        self.powerSpinBox.setMaximum(20)
        self.powerSpinBox.setObjectName("powerSpinBox")
        self.lambdaHorizontalScrollBar = QtWidgets.QScrollBar(Form)
        self.lambdaHorizontalScrollBar.setGeometry(QtCore.QRect(140, 40, 141, 17))
        self.lambdaHorizontalScrollBar.setMaximum(150)
        self.lambdaHorizontalScrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.lambdaHorizontalScrollBar.setObjectName("lambdaHorizontalScrollBar")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(70, 10, 30, 20))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(40, 40, 66, 16))
        self.label_2.setObjectName("label_2")
        self.lambdaLabel = QtWidgets.QLabel(Form)
        self.lambdaLabel.setGeometry(QtCore.QRect(340, 40, 40, 16))
        self.lambdaLabel.setText("")
        self.lambdaLabel.setObjectName("lambdaLabel")
        self.startButton = QtWidgets.QPushButton(Form)
        self.startButton.setGeometry(QtCore.QRect(650, 30, 75, 23))
        self.startButton.setObjectName("startButton")
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(330, 0, 80, 40))
        self.label_3.setObjectName("label_3")
        self.accuracyLabel = QtWidgets.QLabel(Form)
        self.accuracyLabel.setGeometry(QtCore.QRect(410, 0, 80, 40))
        self.accuracyLabel.setText("")
        self.accuracyLabel.setObjectName("accuracyLabel")

        self.retranslateUi(Form)
        self.powerSpinBox.valueChanged['int'].connect(Form.updataLogisticRegression)
        self.lambdaHorizontalScrollBar.valueChanged['int'].connect(Form.updataLogisticRegression)
        self.lambdaHorizontalScrollBar.valueChanged['int'].connect(Form.lambdaRateChange)
        self.startButton.clicked.connect(Form.showBound)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "Power"))
        self.label_2.setText(_translate("Form", "lambadaRate"))
        self.startButton.setText(_translate("Form", "决策边界"))
        self.label_3.setText(_translate("Form", "模型准确率："))

from MatplotlibWidget import MatplotlibWidget
