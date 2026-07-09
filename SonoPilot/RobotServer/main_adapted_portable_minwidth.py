# -*- coding: utf-8 -*-
"""
main_adapted_portable_minwidth.py
- 在便携版基础上，新增：左右分栏采用 QSplitter，三栏均设置最小宽度；超过后自适应伸缩
- 适用于窗口缩小时保证关键区域不被挤没；放大时按权重自适应
运行：python main_adapted_portable_minwidth.py
"""
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QObject
import uuid
# 可选导入 QtWebEngine（若失败将回退为原生气泡对话）
HAS_WEBENGINE = True
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    from PyQt5.QtWebChannel import QWebChannel
    from PyQt5.QtCore import QUrl
except Exception:
    HAS_WEBENGINE = False


# ----------------------------- 1) 原始 Ui_Form（来自用户提供代码，未改结构名） ----------------------------- #
class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1850, 1000)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        Form.setMinimumSize(QtCore.QSize(1200, 720))  # 调低窗口最小尺寸，结合栏位最小宽度更灵活
        Form.setMaximumSize(QtCore.QSize(1920, 1246))
        Form.setStyleSheet("background-color: rgb(8, 37, 62);")
        self.horizontalLayout_22 = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout_22.setObjectName("horizontalLayout_22")
        self.verticalLayout_15 = QtWidgets.QVBoxLayout()
        self.verticalLayout_15.setObjectName("verticalLayout_15")
        self.horizontalLayout_20 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_20.setObjectName("horizontalLayout_20")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_20.addItem(spacerItem)
        self.label_19 = QtWidgets.QLabel(Form)
        font = QtGui.QFont()
        font.setPointSize(30)
        font.setBold(True)
        font.setWeight(75)
        self.label_19.setFont(font)
        self.label_19.setStyleSheet("color: rgb(238, 238, 236);")
        self.label_19.setObjectName("label_19")
        self.horizontalLayout_20.addWidget(self.label_19)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_20.addItem(spacerItem1)
        self.verticalLayout_15.addLayout(self.horizontalLayout_20)
        self.line_6 = QtWidgets.QFrame(Form)
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.verticalLayout_15.addWidget(self.line_6)

        # 该布局原本容纳左/中/右三大块，我们稍后会用 QSplitter 替换
        self.horizontalLayout_21 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_21.setObjectName("horizontalLayout_21")

        # 左侧（相机/检测/超声/分割 + 压力扭矩）
        self.widget_5 = QtWidgets.QWidget(Form)
        self.widget_5.setMinimumSize(QtCore.QSize(0, 0))
        self.widget_5.setMaximumSize(QtCore.QSize(720, 982))
        self.widget_5.setObjectName("widget_5")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.widget_5)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.widget_6 = QtWidgets.QWidget(self.widget_5)
        self.widget_6.setStyleSheet("background-color: rgb(38, 68, 93);")
        self.widget_6.setObjectName("widget_6")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget_6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_RGB = QtWidgets.QLabel(self.widget_6)
        self.label_RGB.setMinimumSize(QtCore.QSize(328, 243))
        self.label_RGB.setStyleSheet("")
        self.label_RGB.setText("")
        self.label_RGB.setObjectName("label_RGB")
        self.horizontalLayout.addWidget(self.label_RGB)
        self.horizontalLayout_12.addWidget(self.widget_6)
        self.widget_7 = QtWidgets.QWidget(self.widget_5)
        self.widget_7.setMinimumSize(QtCore.QSize(346, 261))
        self.widget_7.setStyleSheet("background-color: rgb(38, 68, 93);")
        self.widget_7.setObjectName("widget_7")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.widget_7)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_20 = QtWidgets.QLabel(self.widget_7)
        self.label_20.setText("")
        self.label_20.setObjectName("label_20")
        self.horizontalLayout_3.addWidget(self.label_20)
        self.horizontalLayout_4.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_12.addWidget(self.widget_7)
        self.verticalLayout_2.addLayout(self.horizontalLayout_12)
        self.line = QtWidgets.QFrame(self.widget_5)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_2.addWidget(self.line)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_US = QtWidgets.QLabel(self.widget_5)
        self.label_US.setMinimumSize(QtCore.QSize(346, 261))
        self.label_US.setStyleSheet("background-color: rgb(38, 68, 93);")
        self.label_US.setText("")
        self.label_US.setObjectName("label_US")
        self.horizontalLayout_2.addWidget(self.label_US)
        self.label_seg = QtWidgets.QLabel(self.widget_5)
        self.label_seg.setMinimumSize(QtCore.QSize(346, 261))
        self.label_seg.setStyleSheet("background-color: rgb(38, 68, 93);")
        self.label_seg.setText("")
        self.label_seg.setObjectName("label_seg")
        self.horizontalLayout_2.addWidget(self.label_seg)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.line_2 = QtWidgets.QFrame(self.widget_5)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout_2.addWidget(self.line_2)
        self.widget = QtWidgets.QWidget(self.widget_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setMinimumSize(QtCore.QSize(700, 200))
        self.widget.setMaximumSize(QtCore.QSize(16777215, 200))
        self.widget.setStyleSheet("background-color: rgb(38, 68, 93);")
        self.widget.setObjectName("widget")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.horizontalLayout_6.addLayout(self.horizontalLayout_10)
        self.verticalLayout_2.addWidget(self.widget)
        self.widget_2 = QtWidgets.QWidget(self.widget_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_2.sizePolicy().hasHeightForWidth())
        self.widget_2.setSizePolicy(sizePolicy)
        self.widget_2.setMinimumSize(QtCore.QSize(700, 200))
        self.widget_2.setMaximumSize(QtCore.QSize(16777215, 200))
        self.widget_2.setStyleSheet("background-color: rgb(38, 68, 93);")
        self.widget_2.setObjectName("widget_2")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.widget_2)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.horizontalLayout_9.addLayout(self.horizontalLayout_11)
        self.verticalLayout_2.addWidget(self.widget_2)
        self.verticalLayout_5.addLayout(self.verticalLayout_2)

        # 分隔线（原布局内）
        self.line_3 = QtWidgets.QFrame(Form)
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")

        # 中部容器（将放置聊天）
        self.widget_4 = QtWidgets.QWidget(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_4.sizePolicy().hasHeightForWidth())
        self.widget_4.setSizePolicy(sizePolicy)
        self.widget_4.setMaximumSize(QtCore.QSize(852, 982))
        self.widget_4.setObjectName("widget_4")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.widget_4)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.widget_8 = QtWidgets.QWidget(self.widget_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.widget_8.setSizePolicy(sizePolicy)
        self.widget_8.setMinimumSize(QtCore.QSize(800, 300))
        self.widget_8.setStyleSheet("background-color: rgb(38, 68, 93);")
        self.widget_8.setObjectName("widget_8")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout(self.widget_8)
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.horizontalLayout_14.addLayout(self.horizontalLayout_13)
        self.verticalLayout_9.addWidget(self.widget_8)

        # 该行原有报告/表格，后续被隐藏
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.widget_10 = QtWidgets.QWidget(self.widget_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        self.widget_10.setSizePolicy(sizePolicy)
        self.widget_10.setObjectName("widget_10")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.widget_10)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.tableWidget = QtWidgets.QTableWidget(self.widget_10)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.MinimumExpanding)
        self.tableWidget.setSizePolicy(sizePolicy)
        self.tableWidget.setMinimumSize(QtCore.QSize(256, 636))
        self.tableWidget.setStyleSheet("QTableView { background-color: rgb(38, 68, 93); border: none; }")
        self.tableWidget.setRowCount(1)
        self.tableWidget.setColumnCount(1)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.horizontalHeader().setVisible(False)
        self.tableWidget.verticalHeader().setVisible(False)
        self.verticalLayout_8.addWidget(self.tableWidget)
        self.horizontalLayout_5.addWidget(self.widget_10)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem2)
        self.widget_9 = QtWidgets.QWidget(self.widget_4)
        self.widget_9.setObjectName("widget_9")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.widget_9)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout_6.addItem(spacerItem3)
        self.label_PDF = QtWidgets.QLabel(self.widget_9)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        self.label_PDF.setSizePolicy(sizePolicy)
        self.label_PDF.setMinimumSize(QtCore.QSize(422, 634))
        self.label_PDF.setMaximumSize(QtCore.QSize(422, 16777215))
        self.label_PDF.setStyleSheet("")
        self.label_PDF.setText("")
        self.label_PDF.setObjectName("label_PDF")
        self.verticalLayout_6.addWidget(self.label_PDF)
        self.verticalLayout_7.addLayout(self.verticalLayout_6)
        self.horizontalLayout_5.addWidget(self.widget_9)
        self.verticalLayout_9.addLayout(self.horizontalLayout_5)
        self.verticalLayout_10.addLayout(self.verticalLayout_9)

        # 右侧（患者信息 + 控制）
        self.widget_3 = QtWidgets.QWidget(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        self.widget_3.setSizePolicy(sizePolicy)
        self.widget_3.setObjectName("widget_3")
        self.verticalLayout_14 = QtWidgets.QVBoxLayout(self.widget_3)
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.verticalLayout_13 = QtWidgets.QVBoxLayout()
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_18.addItem(spacerItem4)
        self.label_12 = QtWidgets.QLabel(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_12.setFont(font)
        self.label_12.setStyleSheet("color: rgb(238, 238, 236);")
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_18.addWidget(self.label_12)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_18.addItem(spacerItem5)
        self.verticalLayout_11.addLayout(self.horizontalLayout_18)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label_13 = QtWidgets.QLabel(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_13.setFont(font)
        self.label_13.setStyleSheet("color: rgb(238, 238, 236);")
        self.label_13.setObjectName("label_13")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_13)
        self.label_14 = QtWidgets.QLabel(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_14.setFont(font)
        self.label_14.setStyleSheet("color: rgb(238, 238, 236);")
        self.label_14.setObjectName("label_14")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_14)
        self.label_15 = QtWidgets.QLabel(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_15.setFont(font)
        self.label_15.setStyleSheet("color: rgb(238, 238, 236);")
        self.label_15.setObjectName("label_15")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_15)
        self.label_16 = QtWidgets.QLabel(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_16.setFont(font)
        self.label_16.setStyleSheet("color: rgb(238, 238, 236);")
        self.label_16.setObjectName("label_16")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_16)
        self.label_17 = QtWidgets.QLabel(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_17.setFont(font)
        self.label_17.setStyleSheet("color: rgb(238, 238, 236);")
        self.label_17.setObjectName("label_17")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_17)
        self.label_18 = QtWidgets.QLabel(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_18.setFont(font)
        self.label_18.setStyleSheet("color: rgb(238, 238, 236);")
        self.label_18.setObjectName("label_18")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_18)
        self.comboBox = QtWidgets.QComboBox(self.widget_3)
        self.comboBox.setStyleSheet("color: rgb(238, 238, 236);\n"
"background-color: rgb(38, 68, 93);")
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.comboBox)
        self.lineEdit = QtWidgets.QLineEdit(self.widget_3)
        self.lineEdit.setStyleSheet("color: rgb(238, 238, 236);\n"
"background-color: rgb(38, 68, 93);")
        self.lineEdit.setObjectName("lineEdit")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.widget_3)
        self.lineEdit_2.setStyleSheet("color: rgb(238, 238, 236);\n"
"background-color: rgb(38, 68, 93);")
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_2)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.widget_3)
        self.lineEdit_3.setStyleSheet("color: rgb(238, 238, 236);\n"
"background-color: rgb(38, 68, 93);")
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lineEdit_3)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.widget_3)
        self.lineEdit_4.setStyleSheet("color: rgb(238, 238, 236);\n"
"background-color: rgb(38, 68, 93);")
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.lineEdit_4)
        self.lineEdit_5 = QtWidgets.QLineEdit(self.widget_3)
        self.lineEdit_5.setStyleSheet("color: rgb(238, 238, 236);\n"
"background-color: rgb(38, 68, 93);")
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.lineEdit_5)
        self.verticalLayout_11.addLayout(self.formLayout)
        self.verticalLayout_13.addLayout(self.verticalLayout_11)
        self.verticalLayout_12 = QtWidgets.QVBoxLayout()
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_15.addItem(spacerItem6)
        self.label_2 = QtWidgets.QLabel(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color: rgb(238, 238, 236);")
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_15.addWidget(self.label_2)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_15.addItem(spacerItem7)
        self.label_11 = QtWidgets.QLabel(self.widget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Preferred)
        self.label_11.setSizePolicy(sizePolicy)
        self.label_11.setText("")
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_15.addWidget(self.label_11)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_15.addItem(spacerItem8)
        self.verticalLayout_12.addLayout(self.horizontalLayout_15)
        self.line_5 = QtWidgets.QFrame(self.widget_3)
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.verticalLayout_12.addWidget(self.line_5)
        self.pushButton = QtWidgets.QPushButton(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("color: rgb(238, 238, 236);")
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_12.addWidget(self.pushButton)
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setStyleSheet("color: rgb(238, 238, 236);")
        self.pushButton_5.setObjectName("pushButton_5")
        self.verticalLayout_4.addWidget(self.pushButton_5)
        self.pushButton_8 = QtWidgets.QPushButton(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_8.setFont(font)
        self.pushButton_8.setStyleSheet("color: rgb(238, 238, 236);")
        self.pushButton_8.setObjectName("pushButton_8")
        self.verticalLayout_4.addWidget(self.pushButton_8)
        self.pushButton_9 = QtWidgets.QPushButton(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_9.setFont(font)
        self.pushButton_9.setStyleSheet("color: rgb(238, 238, 236);")
        self.pushButton_9.setObjectName("pushButton_9")
        self.verticalLayout_4.addWidget(self.pushButton_9)
        self.horizontalLayout_17.addLayout(self.verticalLayout_4)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.radioButton = QtWidgets.QRadioButton(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.radioButton.setFont(font)
        self.radioButton.setStyleSheet("color: rgb(238, 238, 236);")
        self.radioButton.setChecked(True)
        self.radioButton.setObjectName("radioButton")
        self.verticalLayout.addWidget(self.radioButton)
        self.radioButton_2 = QtWidgets.QRadioButton(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.radioButton_2.setFont(font)
        self.radioButton_2.setStyleSheet("color: rgb(238, 238, 236);")
        self.radioButton_2.setObjectName("radioButton_2")
        self.verticalLayout.addWidget(self.radioButton_2)
        self.radioButton_3 = QtWidgets.QRadioButton(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.radioButton_3.setFont(font)
        self.radioButton_3.setStyleSheet("color: rgb(238, 238, 236);")
        self.radioButton_3.setObjectName("radioButton_3")
        self.verticalLayout.addWidget(self.radioButton_3)
        self.horizontalLayout_17.addLayout(self.verticalLayout)
        self.verticalLayout_12.addLayout(self.horizontalLayout_17)
        self.pushButton_6 = QtWidgets.QPushButton(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_6.setFont(font)
        self.pushButton_6.setStyleSheet("color: rgb(238, 238, 236);")
        self.pushButton_6.setObjectName("pushButton_6")
        self.verticalLayout_12.addWidget(self.pushButton_6)
        self.horizontalLayout_19 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_19.setObjectName("horizontalLayout_19")
        self.pushButton_2 = QtWidgets.QPushButton(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet("color: rgb(238, 238, 236);")
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_19.addWidget(self.pushButton_2)
        self.pushButton_7 = QtWidgets.QPushButton(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_7.setFont(font)
        self.pushButton_7.setStyleSheet("color: rgb(238, 238, 236);")
        self.pushButton_7.setObjectName("pushButton_7")
        self.horizontalLayout_19.addWidget(self.pushButton_7)
        self.verticalLayout_12.addLayout(self.horizontalLayout_19)
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_16.addItem(spacerItem9)
        self.label = QtWidgets.QLabel(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet("color: rgb(238, 238, 236);")
        self.label.setObjectName("label")
        self.horizontalLayout_16.addWidget(self.label)
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_16.addItem(spacerItem10)
        self.verticalLayout_12.addLayout(self.horizontalLayout_16)
        self.pushButton_3 = QtWidgets.QPushButton(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setStyleSheet("color: rgb(238, 238, 236);")
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout_12.addWidget(self.pushButton_3)
        self.progressBar = QtWidgets.QProgressBar(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.progressBar.setFont(font)
        self.progressBar.setStyleSheet("color: rgb(238, 238, 236);")
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.verticalLayout_12.addWidget(self.progressBar)
        self.pushButton_4 = QtWidgets.QPushButton(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setStyleSheet("color: rgb(238, 238, 236);")
        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout_12.addWidget(self.pushButton_4)
        self.progressBar_2 = QtWidgets.QProgressBar(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.progressBar_2.setFont(font)
        self.progressBar_2.setStyleSheet("color: rgb(238, 238, 236);")
        self.progressBar_2.setProperty("value", 0)
        self.progressBar_2.setObjectName("progressBar_2")
        self.verticalLayout_12.addWidget(self.progressBar_2)
        self.verticalLayout_13.addLayout(self.verticalLayout_12)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        spacerItem11 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem11)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        spacerItem12 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem12)
        self.label_4 = QtWidgets.QLabel(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("color: rgb(238, 238, 236);")
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_7.addWidget(self.label_4)
        self.label_7 = QtWidgets.QLabel(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setStyleSheet("color: rgb(164, 0, 0);")
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_7.addWidget(self.label_7)
        spacerItem13 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem13)
        self.verticalLayout_3.addLayout(self.horizontalLayout_7)
        self.dial = QtWidgets.QDial(self.widget_3)
        self.dial.setMinimumSize(QtCore.QSize(100, 100))
        self.dial.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.dial.setMaximum(100)
        self.dial.setSingleStep(1)
        self.dial.setProperty("value", 20)
        self.dial.setOrientation(QtCore.Qt.Vertical)
        self.dial.setNotchesVisible(True)
        self.dial.setObjectName("dial")
        self.verticalLayout_3.addWidget(self.dial)
        self.horizontalLayout_8.addLayout(self.verticalLayout_3)
        spacerItem14 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem14)
        self.verticalLayout_13.addLayout(self.horizontalLayout_8)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_3 = QtWidgets.QLabel(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("color: rgb(238, 238, 236);")
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1)
        self.horizontalSlider = QtWidgets.QSlider(self.widget_3)
        self.horizontalSlider.setMaximum(300)
        self.horizontalSlider.setProperty("value", 117)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.gridLayout.addWidget(self.horizontalSlider, 0, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.widget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        self.label_8.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setStyleSheet("color: rgb(164, 0, 0);")
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 0, 2, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("color: rgb(238, 238, 236);")
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 1, 0, 1, 1)
        self.horizontalSlider_2 = QtWidgets.QSlider(self.widget_3)
        self.horizontalSlider_2.setProperty("value", 7)
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.gridLayout.addWidget(self.horizontalSlider_2, 1, 1, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setStyleSheet("color: rgb(164, 0, 0);")
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 1, 2, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.widget_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setStyleSheet("color: rgb(238, 238, 236);")
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 2, 0, 1, 1)
        self.horizontalSlider_3 = QtWidgets.QSlider(self.widget_3)
        self.horizontalSlider_3.setMaximum(10)
        self.horizontalSlider_3.setProperty("value", 5)
        self.horizontalSlider_3.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_3.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.horizontalSlider_3.setObjectName("horizontalSlider_3")
        self.gridLayout.addWidget(self.horizontalSlider_3, 2, 1, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.widget_3)
        self.label_10.setMinimumSize(QtCore.QSize(50, 0))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_10.setFont(font)
        self.label_10.setStyleSheet("color: rgb(164, 0, 0);")
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 2, 2, 1, 1)
        self.verticalLayout_13.addLayout(self.gridLayout)
        self.textEdit = QtWidgets.QTextEdit(self.widget_3)
        self.textEdit.setStyleSheet("background-color: rgb(38, 68, 93);\n"
"color: rgb(238, 238, 236);")
        self.textEdit.setReadOnly(True)
        self.textEdit.setObjectName("textEdit")
        self.verticalLayout_13.addWidget(self.textEdit)
        self.verticalLayout_14.addLayout(self.verticalLayout_13)

        # 将三大块先放进占位容器（我们稍后用 splitter 重新组织）
        self.horizontalLayout_21.addWidget(self.widget_5)
        self.horizontalLayout_21.addWidget(self.line_3)
        self.horizontalLayout_21.addWidget(self.widget_4)
        self.horizontalLayout_21.addWidget(self.widget_3)

        self.verticalLayout_15.addLayout(self.horizontalLayout_21)
        self.horizontalLayout_22.addLayout(self.verticalLayout_15)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "IS-MAN"))
        self.label_19.setText(_translate("Form", "IS-MAN"))
        self.label_12.setText(_translate("Form", "患者信息"))
        self.label_13.setText(_translate("Form", "姓名："))
        self.label_14.setText(_translate("Form", "ID号："))
        self.label_15.setText(_translate("Form", "性别："))
        self.label_16.setText(_translate("Form", "年龄："))
        self.label_17.setText(_translate("Form", "临床诊断："))
        self.label_18.setText(_translate("Form", "病例描述："))
        self.comboBox.setItemText(0, _translate("Form", "男"))
        self.comboBox.setItemText(1, _translate("Form", "不清楚"))
        self.comboBox.setItemText(2, _translate("Form", "女"))
        self.lineEdit.setText(_translate("Form", "小小"))
        self.lineEdit_2.setText(_translate("Form", "A000000"))
        self.lineEdit_3.setText(_translate("Form", "24"))
        self.lineEdit_4.setText(_translate("Form", "甲状腺结节"))
        self.lineEdit_5.setText(_translate("Form", "体检"))
        self.label_2.setText(_translate("Form", "状态灯"))
        self.pushButton.setText(_translate("Form", "（1）启动自检"))
        self.pushButton_5.setText(_translate("Form", "（2-1）关键点预览"))
        self.pushButton_8.setText(_translate("Form", "（2-2）关键点发布"))
        self.pushButton_9.setText(_translate("Form", "（2-3）关键点清除"))
        self.radioButton.setText(_translate("Form", "方法1：喉结明显"))
        self.radioButton_2.setText(_translate("Form", "方法2: 下颌明显"))
        self.radioButton_3.setText(_translate("Form", "方法3: 不明显"))
        self.pushButton_6.setText(_translate("Form", "（3）压力调零"))
        self.pushButton_2.setText(_translate("Form", "（4）自动扫查"))
        self.pushButton_7.setText(_translate("Form", "（4-1）中断扫查"))
        self.label.setText(_translate("Form", "结果生成"))
        self.pushButton_3.setText(_translate("Form", "（5）3D重建"))
        self.pushButton_4.setText(_translate("Form", "（6）AI分析"))
        self.label_4.setText(_translate("Form", "探头压力"))
        self.label_7.setText(_translate("Form", "2N"))
        self.label_3.setText(_translate("Form", "P比例"))
        self.label_8.setText(_translate("Form", "1.17"))
        self.label_5.setText(_translate("Form", "I积分"))
        self.label_9.setText(_translate("Form", "0.07"))
        self.label_6.setText(_translate("Form", "D微分"))
        self.label_10.setText(_translate("Form", "5"))


# ----------------------------- 2) 对话桥（两种实现复用） ----------------------------- #
class ChatBridge(QObject):
    assistantMessage = pyqtSignal(str)
    systemMessage = pyqtSignal(str)

    @pyqtSlot(str)
    def onUserMessage(self, text: str):
        # TODO: 接入真实 LLM/Agent（HTTP/gRPC/SDK等）
        reply = f"【助手】已收到：{text}。如需实际推理，请接入后端接口。"
        self.assistantMessage.emit(reply)


# ----------------------------- 3A) WebEngine 聊天部件 ----------------------------- #
class _JSCoalescer(QtCore.QObject):
    def __init__(self, view, interval_ms=30, parent=None):
        super().__init__(parent)
        self.view = view
        self.buf = []
        self.t = QtCore.QTimer(self)
        self.t.setInterval(interval_ms)
        self.t.timeout.connect(self.flush)

    def enqueue(self, script: str):
        self.buf.append(script)
        if not self.t.isActive():
            self.t.start()

    @QtCore.pyqtSlot()
    def flush(self):
        if not self.buf: 
            self.t.stop(); 
            return
        # 合并为一次调用，减少跨进程次数
        merged = ";\n".join(self.buf) + ";"
        self.buf.clear()
        self.view.page().runJavaScript(merged)
        # 若还在持续写入，定时器保持运行；否则 timeout 后自动停


class WebChatWidget(QtWidgets.QWidget):
    readyChanged = pyqtSignal(bool)
    def __init__(self, parent=None):
        super().__init__(parent)
        if not HAS_WEBENGINE:
            raise RuntimeError("QtWebEngine 不可用")
        self._coalescer = _JSCoalescer(self.view, interval_ms=30, parent=self)
        self._ready = False
        self._js_queue = []  # 等待页面加载完成后执行的 JS 队列

        v = QtWidgets.QVBoxLayout(self); v.setContentsMargins(0,0,0,0); v.setSpacing(0)
        self.view = QWebEngineView(self); self.view.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        v.addWidget(self.view)
        self.view.loadFinished.connect(self._on_load_finished)

        # WebChannel（仅用于从网页向 Python 传入 onUserMessage）
        self.channel = QWebChannel(self.view.page()); self.bridge = ChatBridge()
        self.channel.registerObject('pyBridge', self.bridge); self.view.page().setWebChannel(self.channel)
        # 将 Python → 对话 UI 的信号改为调用 JS（而非依赖网页内订阅），避免时序问题
        self.bridge.assistantMessage.connect(self.send_assistant_text)
        self.bridge.systemMessage.connect(self.send_system)

        self.view.setHtml(self._html(), QUrl("qrc:///"))

        # self.view.setAttribute(Qt.WA_OpaquePaintEvent, True)
        # self.view.setAttribute(Qt.WA_NoSystemBackground, True)
        # try:
        #     self.view.page().setBackgroundColor(QtGui.QColor("#0b1733"))
        # except Exception:
        #     pass


    def _on_load_finished(self, ok: bool):
        self._ready = True
        # 执行排队 JS
        # while self._js_queue:
        #     js = self._js_queue.pop(0)
        #     self.view.page().runJavaScript(js)
        while self._js_queue:
            self._coalescer.enqueue(self._js_queue.pop(0))
        self.readyChanged.emit(True)

        


    # ---- 统一接口（全部走 JS+队列） ----
    # def _js(self, script: str):
    #     if self._ready:
    #         self.view.page().runJavaScript(script)
    #     else:
    #         self._js_queue.append(script)
    def _js(self, script: str):
        if self._ready:
            self._coalescer.enqueue(script)
        else:
            self._js_queue.append(script)


    def send_user_text(self, text: str):
        self._js(f"add('user', {repr(text)})")

    def send_assistant_text(self, text: str):
        self._js(f"add('assistant', {repr(text)})")

    def send_system(self, text: str):
        self._js(f"addSys({repr(text)})")

    def _file_to_data_url(self, path_or_b64: str) -> str:
        import os, base64, mimetypes
        if os.path.exists(path_or_b64):
            mime = mimetypes.guess_type(path_or_b64)[0] or "image/png"
            data = open(path_or_b64, "rb").read()
            b64 = base64.b64encode(data).decode("ascii")
            return f"data:{mime};base64,{b64}"
        return f"data:image/png;base64,{path_or_b64}"

    def send_user_image(self, path_or_b64: str):
        url = self._file_to_data_url(path_or_b64)
        self._js(f"addImage('user', {repr(url)})")

    def send_assistant_image(self, path_or_b64: str):
        url = self._file_to_data_url(path_or_b64)
        self._js(f"addImage('assistant', {repr(url)})")

    def start_assistant_stream(self) -> str:
        mid = f"a_{uuid.uuid4().hex[:8]}"
        self._js(f"startStream('assistant', {repr(mid)})")
        return mid

    def start_user_stream(self) -> str:
        mid = f"u_{uuid.uuid4().hex[:8]}"
        self._js(f"startStream('user', {repr(mid)})")
        return mid

    def append_stream(self, message_id: str, chunk: str) -> None:
        # chunk 可能包含引号等，使用 repr 由 JS 端原样读入
        self._js(f"appendStream({repr(message_id)}, {repr(chunk)})")

    def finish_stream(self, message_id: str) -> None:
        self._js(f"finishStream({repr(message_id)})")

    @staticmethod
    def _html()->str:
        return r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>智能对话</title>
<style>
  :root{--bg:#0b1733;--bd:rgba(65,234,212,.25);--fg:#e5eef8;--muted:#9ab;}
  html,body{height:100%;margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial;}
  .chat{display:flex;flex-direction:column;height:100%;min-height:0}
  .scroll{flex:1;min-height:0;overflow:auto;padding:14px}
  .row{display:flex;margin:8px 0}.row.user{justify-content:flex-end}.row.assistant{justify-content:flex-start}
  .bubble{max-width:72%;padding:10px 12px;border-radius:14px;border:1px solid var(--bd);background:rgba(255,255,255,.05);line-height:1.5;word-break:break-word;white-space:pre-wrap}
  .user .bubble{background:rgba(65,234,212,.12);border-color:rgba(65,234,212,.45)}
  .system{color:var(--muted);text-align:center;font-size:12px;margin:6px 0}
  form{display:flex;gap:8px;padding:10px;border-top:1px solid var(--bd);background:rgba(255,255,255,.03)}
  input{flex:1;min-width:0;padding:10px;border-radius:10px;border:1px solid var(--bd);background:#0e2248;color:var(--fg)}
  button{padding:10px 14px;border-radius:10px;border:1px solid rgba(65,234,212,.45);background:rgba(65,234,212,.12);color:var(--fg);cursor:pointer}
  img.msg{max-width:100%;border-radius:10px;display:block}
  .cursor{opacity:.6;margin-left:2px}
</style>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script></head>
<body><div class="chat"><div id="scroll" class="scroll"></div>
<form id="form"><input id="input" type="text" placeholder="请输入指令…" autocomplete="off"/><button>发送</button></form></div>
<script>
  // 普通追加：文本 / 系统提示 / 图片
  function add(role,text){
    const s=document.getElementById('scroll');
    const r=document.createElement('div'); r.className='row '+role;
    const b=document.createElement('div'); b.className='bubble'; b.textContent=text;
    r.appendChild(b); s.appendChild(r); s.scrollTop=s.scrollHeight;
  }
  function addSys(text){
    const s=document.getElementById('scroll');
    const d=document.createElement('div'); d.className='system'; d.textContent=text;
    s.appendChild(d); s.scrollTop=s.scrollHeight;
  }
  function addImage(role,dataUrl){
    const s=document.getElementById('scroll');
    const r=document.createElement('div'); r.className='row '+role;
    const b=document.createElement('div'); b.className='bubble';
    const img=document.createElement('img'); img.className='msg'; img.src=dataUrl;
    b.appendChild(img); r.appendChild(b); s.appendChild(r); s.scrollTop=s.scrollHeight;
  }

  // —— 流式输出支持 —— //
  // 使用 __streamStore 记录“同一条消息”的气泡与光标；Python 端以 message_id 关联
  const __streamStore = {}; // mid -> {bubble, cursor}
  function startStream(role, mid){
    const s=document.getElementById('scroll');
    const r=document.createElement('div'); r.className='row '+role;
    const b=document.createElement('div'); b.className='bubble';
    r.appendChild(b); s.appendChild(r);
    // 光标（可选）
    const cur=document.createElement('span'); cur.className='cursor'; cur.id='cur_'+mid; cur.textContent='▍';
    b.appendChild(cur);
    __streamStore[mid]={bubble:b,cursor:cur};
    s.scrollTop=s.scrollHeight;
  }
  function appendStream(mid, text){
    const rec=__streamStore[mid]; if(!rec) return;
    const t=document.createTextNode(text);
    // 插入到光标之前
    rec.bubble.insertBefore(t, rec.cursor||null);
    const s=document.getElementById('scroll'); s.scrollTop=s.scrollHeight;
  }
  function finishStream(mid){
    const rec=__streamStore[mid]; if(!rec) return;
    if(rec.cursor && rec.cursor.parentNode){ rec.cursor.parentNode.removeChild(rec.cursor); }
    delete __streamStore[mid];
  }

  // WebChannel：仅负责把“网页输入”传给 Python
  new QWebChannel(qt.webChannelTransport,function(channel){
    const pyBridge = channel.objects.pyBridge;
    document.getElementById('form').addEventListener('submit',function(e){
      e.preventDefault();
      const i=document.getElementById('input'); const t=(i.value||'').trim(); if(!t) return;
      add('user',t); i.value='';
      if(pyBridge&&pyBridge.onUserMessage) pyBridge.onUserMessage(t);
    });
  });


</script></body></html>"""



# ----------------------------- 3B) 原生聊天（降级实现） ----------------------------- #
class Bubble(QtWidgets.QFrame):
    def __init__(self, widget: QtWidgets.QWidget, align_right: bool = False, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        layout = QtWidgets.QHBoxLayout(self); layout.setContentsMargins(0,0,0,0); layout.setSpacing(0)
        if align_right:
            layout.addStretch(1); layout.addWidget(widget, 0)
        else:
            layout.addWidget(widget, 0); layout.addStretch(1)


class NativeChatWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bridge = ChatBridge()
        # 用于流式输出时记录 message_id -> QLabel
        self._stream_labels = {}   # ★★★ 加在这里 ★★★
        root = QtWidgets.QVBoxLayout(self); root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        # 滚动区
        self.scroll = QtWidgets.QScrollArea(self); self.scroll.setWidgetResizable(True); self.scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        root.addWidget(self.scroll, 1)
        self.container = QtWidgets.QWidget(self.scroll)
        self.vbox = QtWidgets.QVBoxLayout(self.container); self.vbox.setContentsMargins(8,8,8,8); self.vbox.setSpacing(8)
        self.vbox.addStretch(1); self.scroll.setWidget(self.container)

        # 输入区
        in_row = QtWidgets.QHBoxLayout()
        self.input = QtWidgets.QLineEdit(self); self.btn = QtWidgets.QPushButton("发送", self)
        in_row.addWidget(self.input, 1); in_row.addWidget(self.btn, 0); root.addLayout(in_row, 0)

        # 信号连接
        self.btn.clicked.connect(self._on_submit); self.input.returnPressed.connect(self._on_submit)
        self.bridge.assistantMessage.connect(lambda t: self._append_assistant_text(t))
        self.bridge.systemMessage.connect(lambda t: self._append_system(t))

        self._append_system("智能对话（原生）已就绪。")

    # ----- 内部追加方法 -----
    def _append_system(self, text: str):
        lab = QtWidgets.QLabel(text, self.container); lab.setAlignment(Qt.AlignCenter); lab.setStyleSheet("color:#9ab;font-size:12px;")
        self.vbox.insertWidget(self.vbox.count()-1, lab); QtCore.QTimer.singleShot(0, self._scroll_to_bottom)

    def _append_user_text(self, text: str):
        lab = QtWidgets.QLabel(text, self.container); lab.setWordWrap(True)
        lab.setStyleSheet("QLabel{background:rgba(65,234,212,.12);border:1px solid rgba(65,234,212,.45);border-radius:12px;padding:10px 12px;color:#e5eef8;}")
        self.vbox.insertWidget(self.vbox.count()-1, Bubble(lab, align_right=True, parent=self.container)); QtCore.QTimer.singleShot(0, self._scroll_to_bottom)

    def _append_assistant_text(self, text: str):
        lab = QtWidgets.QLabel(text, self.container); lab.setWordWrap(True)
        lab.setStyleSheet("QLabel{background:rgba(255,255,255,.05);border:1px solid rgba(65,234,212,.25);border-radius:12px;padding:10px 12px;color:#e5eef8;}")
        self.vbox.insertWidget(self.vbox.count()-1, Bubble(lab, align_right=False, parent=self.container)); QtCore.QTimer.singleShot(0, self._scroll_to_bottom)

    def _append_image(self, path_or_b64: str, align_right: bool):
        pix = self._to_pixmap(path_or_b64)
        lab = QtWidgets.QLabel(self.container); lab.setPixmap(pix.scaledToWidth(360, Qt.SmoothTransformation))
        self.vbox.insertWidget(self.vbox.count()-1, Bubble(lab, align_right=align_right, parent=self.container)); QtCore.QTimer.singleShot(0, self._scroll_to_bottom)

    def _to_pixmap(self, path_or_b64: str) -> QtGui.QPixmap:
        import os, base64
        if os.path.exists(path_or_b64):
            return QtGui.QPixmap(path_or_b64)
        # base64（无前缀）
        data = base64.b64decode(path_or_b64)
        img = QtGui.QImage.fromData(data)
        return QtGui.QPixmap.fromImage(img)

    def _scroll_to_bottom(self):
        sb = self.scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_submit(self):
        t = (self.input.text() or "").strip()
        if not t: return
        self.input.clear(); self._append_user_text(t); self.bridge.onUserMessage(t)

    def start_assistant_stream(self) -> str:
        mid = f"a_{uuid.uuid4().hex[:8]}"
        lab = QtWidgets.QLabel("", self.container); lab.setWordWrap(True)
        lab.setStyleSheet("QLabel{background:rgba(255,255,255,.05);border:1px solid rgba(65,234,212,.25);border-radius:12px;padding:10px 12px;color:#e5eef8;}")
        # 放入左侧（助手侧）
        self.vbox.insertWidget(self.vbox.count()-1, Bubble(lab, align_right=False, parent=self.container))
        self._stream_labels[mid] = lab
        QtCore.QTimer.singleShot(0, self._scroll_to_bottom)
        return mid

    def start_user_stream(self) -> str:
        mid = f"u_{uuid.uuid4().hex[:8]}"
        lab = QtWidgets.QLabel("", self.container); lab.setWordWrap(True)
        lab.setStyleSheet("QLabel{background:rgba(65,234,212,.12);border:1px solid rgba(65,234,212,.45);border-radius:12px;padding:10px 12px;color:#e5eef8;}")
        # 放入右侧（用户侧）
        self.vbox.insertWidget(self.vbox.count()-1, Bubble(lab, align_right=True, parent=self.container))
        self._stream_labels[mid] = lab
        QtCore.QTimer.singleShot(0, self._scroll_to_bottom)
        return mid

    def append_stream(self, message_id: str, chunk: str) -> None:
        lab = self._stream_labels.get(message_id)
        if not lab: return
        lab.setText(lab.text() + chunk)
        QtCore.QTimer.singleShot(0, self._scroll_to_bottom)

    def finish_stream(self, message_id: str) -> None:
        self._stream_labels.pop(message_id, None)

    # ---- 统一外部接口 ----
    def send_user_text(self, text: str): self._append_user_text(text)
    def send_assistant_text(self, text: str): self._append_assistant_text(text)
    def send_user_image(self, path_or_b64: str): self._append_image(path_or_b64, align_right=True)
    def send_assistant_image(self, path_or_b64: str): self._append_image(path_or_b64, align_right=False)
    def send_system(self, text: str): self._append_system(text)

# class StreamWorker(QtCore.QObject):
#     chunk = QtCore.pyqtSignal(str)   # 发出 token/片段
#     finished = QtCore.pyqtSignal()

#     def __init__(self, prompt: str, parent=None):
#         super().__init__(parent)
#         self.prompt = prompt

#     @QtCore.pyqtSlot()
#     def run(self):
#         # 这里替换为您的后端流式接口，比如 SSE/gRPC 流
#         import time
#         fake_tokens = ["已", "开", "始", "规", "划", "路", "径", "，", "请", "保", "持", "稳", "定", "。"]
#         for t in fake_tokens:
#             self.chunk.emit(t)
#             time.sleep(0.05)
#         self.finished.emit()

class StreamWorker(QtCore.QObject):
    """
    简单的流式演示 Worker：
    - 将传入的 reply_text 按字符切片，每 30ms 发一个 token
    - 结束时发出 finished 信号
    """
    chunk = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, reply_text: str, interval_ms: int = 30, parent=None):
        super().__init__(parent)
        self.reply_text = reply_text
        self.interval_ms = interval_ms

    @QtCore.pyqtSlot()
    def run(self):
        from time import sleep
        for ch in self.reply_text:
            self.chunk.emit(ch)
            sleep(self.interval_ms / 1000.0)
        self.finished.emit()


# ----------------------------- 4) 适配主窗口（含 QSplitter 与最小宽度约束） ----------------------------- #
class AdaptedWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # 统一风格
        self._improve_style()

        # 在中部替换为聊天
        self._mount_chat_widget()

        # 用 QSplitter 重组左右三栏，并设置最小宽度与伸缩权重
        self._rebuild_h_splitter()

        # 右侧按钮事件 → 同步系统提示
        self._wire_buttons()

    def _improve_style(self):
        self.setWindowTitle("基于深度学习的甲状腺自动扫查与分析系统（最小宽度自适应版）")
        self.setStyleSheet("""
            QWidget { color: #e5eef8; font-size: 14px; }
            QLabel#label_RGB, QLabel#label_20, QLabel#label_US, QLabel#label_seg, QLabel#label_PDF {
                background: #26445d; border: 1px solid rgba(255,255,255,.06); border-radius: 8px;
            }
            QProgressBar { background: #0e2248; border: 1px solid rgba(255,255,255,.12); border-radius: 6px; height: 10px; }
            QProgressBar::chunk { background: #41ead4; border-radius: 6px; }
            QPushButton { background: rgba(65,234,212,.10); border: 1px solid rgba(65,234,212,.35); border-radius: 8px; padding: 6px 10px; }
            QPushButton:hover { background: rgba(65,234,212,.20); }
            QFrame[frameShape="4"] { border-top: 1px solid rgba(65,234,212,.25); } /* HLine */
        """)

    def _clear_layout(self, layout: QtWidgets.QLayout):
        if not layout:
            return
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
            else:
                self._clear_layout(item.layout())

    def _mount_chat_widget(self):
        # 隐藏 3D/报告容器
        if hasattr(self.ui, 'widget_10'):
            self.ui.widget_10.hide()
        if hasattr(self.ui, 'widget_9'):
            self.ui.widget_9.hide()

        # 清空 widget_8 内部并挂载聊天
        self._clear_layout(self.ui.horizontalLayout_14)
        if HAS_WEBENGINE:
            try:
                self.chat = WebChatWidget(self.ui.widget_8)
            except Exception:
                self.chat = NativeChatWidget(self.ui.widget_8)
        else:
            self.chat = NativeChatWidget(self.ui.widget_8)
        self.ui.horizontalLayout_14.addWidget(self.chat)
        # self.chat.send_system("3D/报告视图已替换为智能对话。")

    def _rebuild_h_splitter(self):
        # 取得原三块
        left = self.ui.widget_5
        center = self.ui.widget_4
        right = self.ui.widget_3

        # 清空最外层横向布局并用 splitter 替换
        host_layout = self.ui.horizontalLayout_21
        self._clear_layout(host_layout)

        splitter = QtWidgets.QSplitter(Qt.Horizontal, self)
        splitter.setObjectName("mainHSplitter")
        splitter.setHandleWidth(6)
        splitter.setStyleSheet("QSplitter::handle { background: rgba(255,255,255,.08); }")

        # 设置三栏最小宽度（可按需调整）
        left.setMinimumWidth(520)
        center.setMinimumWidth(500)   # 对话区域略宽，防止输入区挤压
        right.setMinimumWidth(400)

        # 让三栏可扩展
        for w in (left, center, right):
            w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(right)

        # 设置伸缩因子（总和无所谓，仅表示相对比例）
        splitter.setStretchFactor(0, 3)  # 左
        splitter.setStretchFactor(1, 6)  # 中（聊天）
        splitter.setStretchFactor(2, 2)  # 右

        host_layout.addWidget(splitter)

    def _wire_buttons(self):
        # if hasattr(self.ui, 'pushButton'):
        #     self.ui.pushButton.clicked.connect(lambda: self.chat.send_system("（1）启动自检：正在检查传感器与执行器状态…"))
        # if hasattr(self.ui, 'pushButton_2'):
        #     self.ui.pushButton_2.clicked.connect(lambda: self.chat.send_system("（4）自动扫查：流程已启动，请保持患者体位稳定。"))
        # if hasattr(self.ui, 'pushButton_7'):
        #     self.ui.pushButton_7.clicked.connect(lambda: self.chat.send_system("（4-1）中断扫查：正在安全回撤…"))
        # if hasattr(self.ui, 'pushButton_3'):
        #     self.ui.pushButton_3.clicked.connect(lambda: self.chat.send_system("（5）3D 重建：功能已由智能对话替代。"))
        # if hasattr(self.ui, 'pushButton_4'):
        #     self.ui.pushButton_4.clicked.connect(lambda: self.chat.send_system("（6）AI 分析：请在对话中直接下达分析指令。"))
        # if hasattr(self.ui, 'pushButton_5'):
        #     self.ui.pushButton_5.clicked.connect(lambda: self.chat.send_system("（2-1）关键点预览：已更新预览。"))
        # if hasattr(self.ui, 'pushButton_8'):
        #     self.ui.pushButton_8.clicked.connect(lambda: self.chat.send_system("（2-2）关键点发布：已发布关键点。"))
        # if hasattr(self.ui, 'pushButton_9'):
        #     self.ui.pushButton_9.clicked.connect(lambda: self.chat.send_system("（2-3）关键点清除：已清除关键点。"))
        # if hasattr(self.ui, 'pushButton_6'):
        #     self.ui.pushButton_6.clicked.connect(lambda: self.chat.send_system("（3）压力调零：请轻放探头，系统将重新标定压力零点。"))
        pass

    def start_streaming_reply(self, user_text: str):
        # 1) 先把用户的一条普通消息显示（或使用 start_user_stream 流式显示）
        self.chat.send_user_text(user_text)

        # 2) 为“助手”开启一条流式消息
        mid = self.chat.start_assistant_stream()

        # 3) 后端流式线程
        self._thread = QtCore.QThread(self)
        self._worker = StreamWorker(reply_text=user_text)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.chunk.connect(lambda s: self.chat.append_stream(mid, s))
        self._worker.finished.connect(lambda: self.chat.finish_stream(mid))
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()



# # ----------------------------- 5) 入口 ----------------------------- #
# if __name__ == "__main__":
#     import sys
#     app = QtWidgets.QApplication(sys.argv)
#     w = AdaptedWindow()
#     w.resize(1600, 900)

#     # 你现有的静态消息演示
#     w.chat.send_system('MedAI实验室')
#     w.chat.send_user_image(r'/home/usai/auto_RUSS/R_UI/keypoint_detection/person.png')
#     w.chat.send_user_text('这是什么？')
#     w.chat.send_assistant_image(r'/home/usai/auto_RUSS/R_UI/keypoint_detection/person.png')
#     w.chat.send_assistant_text('a women')

#     # ========================  流式输出测试  ========================
#     # 1) 在对话框中创建一条“助手气泡”，返回 message_id
#     mid = w.chat.start_assistant_stream()

#     # 2) 启动后台线程，逐字把字符串“流式”塞入这条气泡
#     reply_demo = "甲状腺实质回声增粗，血流正常。左叶大小5.8cm×2.8cm×2.7cm。左叶占位病变：多个，直径0.4cm-3.6cm，其中最大结节为多个融合，混合回声，境界清楚，血流丰富。右叶大小6.2cm×3.1cm×2.7cm，占位病变：多个，直径0.5cm-4.8cm，其中较大结节为多个融合，混合回声，较大结节伴液化，境界清楚，血流丰富。峡部厚0.7cm，占位病变：位于中下极，单个，大小1.3cm×0.9cm，混合回声，边界清楚，血流丰富。"
#     thread = QtCore.QThread(w)
#     worker = StreamWorker(reply_text=reply_demo, interval_ms=30)
#     worker.moveToThread(thread)

#     # 连接信号 → UI 追加 token / 结束收尾
#     worker.chunk.connect(lambda s: w.chat.append_stream(mid, s))
#     def _finish():
#         w.chat.finish_stream(mid)
#         thread.quit()
#     worker.finished.connect(_finish)

#     # 线程生命周期收尾
#     thread.started.connect(worker.run)
#     thread.finished.connect(worker.deleteLater)
#     thread.start()
#     # ======================  流式输出测试结束  ======================

#     w.show()
#     sys.exit(app.exec_())
