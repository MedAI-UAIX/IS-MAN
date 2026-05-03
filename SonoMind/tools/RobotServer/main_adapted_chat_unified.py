# -*- coding: utf-8 -*-
"""
main_adapted_chat_unified.py
在用户原工程基础上，完成以下增强：
1) 中间区域替换为“智能对话”（WebEngine 优先，失败则原生气泡）并彻底防止被遮挡/挤压：
   - 采用水平 QSplitter（三栏最小宽度 + 自适应 + 不可折叠 + 初始 setSizes）。
   - 取消中栏 widget_4 的最大尺寸限制；聊天容器使用 Expanding 策略、零边距。
   - 完全隐藏并回收原报告行（widget_10 / widget_9）。
2) 统一对话接口（外部统一调用，无需区分 Web/原生）：
   - send_user_text(text)
   - send_user_image(path_or_b64)
   - send_assistant_text(text)
   - send_assistant_image(path_or_b64)
   - send_system(text)
3) 保留左侧：相机视频(label_RGB)、检测结果(label_20)、超声(label_US)、分割(label_seg)，及右侧表单与按钮；
   按钮点击会在聊天中显示系统提示（可作为流程日志）。

运行：python main_adapted_chat_unified.py
"""
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QObject

# 可选：QtWebEngine（若缺失将自动回退到原生聊天）
HAS_WEBENGINE = True
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    from PyQt5.QtWebChannel import QWebChannel
    from PyQt5.QtCore import QUrl
except Exception:
    HAS_WEBENGINE = False


# ----------------------------- 1) 原始 Ui_Form（变量名与用户一致） ----------------------------- #
class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1850, 1000)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        Form.setMinimumSize(QtCore.QSize(1200, 720))
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
        font.setPointSize(30); font.setBold(True); font.setWeight(75)
        self.label_19.setFont(font)
        self.label_19.setStyleSheet("color: rgb(238, 238, 236);")
        self.label_19.setObjectName("label_19")
        self.horizontalLayout_20.addWidget(self.label_19)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_20.addItem(spacerItem1)
        self.verticalLayout_15.addLayout(self.horizontalLayout_20)
        self.line_6 = QtWidgets.QFrame(Form); self.line_6.setFrameShape(QtWidgets.QFrame.HLine); self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.verticalLayout_15.addWidget(self.line_6)

        # 容纳左/中/右三大块（稍后用 QSplitter 重建）
        self.horizontalLayout_21 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_21.setObjectName("horizontalLayout_21")

        # 左侧：相机/检测/超声/分割 + 压力扭矩
        self.widget_5 = QtWidgets.QWidget(Form)
        self.widget_5.setObjectName("widget_5")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.widget_5)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.widget_6 = QtWidgets.QWidget(self.widget_5); self.widget_6.setStyleSheet("background-color: rgb(38, 68, 93);")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget_6)
        self.label_RGB = QtWidgets.QLabel(self.widget_6); self.label_RGB.setMinimumSize(QtCore.QSize(328, 243)); self.label_RGB.setText("")
        self.horizontalLayout.addWidget(self.label_RGB); self.horizontalLayout_12.addWidget(self.widget_6)
        self.widget_7 = QtWidgets.QWidget(self.widget_5); self.widget_7.setMinimumSize(QtCore.QSize(346, 261)); self.widget_7.setStyleSheet("background-color: rgb(38, 68, 93);")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.widget_7); self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.label_20 = QtWidgets.QLabel(self.widget_7); self.label_20.setText(""); self.horizontalLayout_3.addWidget(self.label_20); self.horizontalLayout_4.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_12.addWidget(self.widget_7)
        self.verticalLayout_2.addLayout(self.horizontalLayout_12)
        self.line = QtWidgets.QFrame(self.widget_5); self.line.setFrameShape(QtWidgets.QFrame.HLine); self.line.setFrameShadow(QtWidgets.QFrame.Sunken); self.verticalLayout_2.addWidget(self.line)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.label_US = QtWidgets.QLabel(self.widget_5); self.label_US.setMinimumSize(QtCore.QSize(346, 261)); self.label_US.setStyleSheet("background-color: rgb(38, 68, 93);"); self.label_US.setText("")
        self.horizontalLayout_2.addWidget(self.label_US)
        self.label_seg = QtWidgets.QLabel(self.widget_5); self.label_seg.setMinimumSize(QtCore.QSize(346, 261)); self.label_seg.setStyleSheet("background-color: rgb(38, 68, 93);"); self.label_seg.setText("")
        self.horizontalLayout_2.addWidget(self.label_seg); self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.line_2 = QtWidgets.QFrame(self.widget_5); self.line_2.setFrameShape(QtWidgets.QFrame.HLine); self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken); self.verticalLayout_2.addWidget(self.line_2)
        self.widget = QtWidgets.QWidget(self.widget_5); self.widget.setMinimumSize(QtCore.QSize(700, 200)); self.widget.setMaximumSize(QtCore.QSize(16777215, 200)); self.widget.setStyleSheet("background-color: rgb(38, 68, 93);")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.widget); self.horizontalLayout_10 = QtWidgets.QHBoxLayout(); self.horizontalLayout_6.addLayout(self.horizontalLayout_10); self.verticalLayout_2.addWidget(self.widget)
        self.widget_2 = QtWidgets.QWidget(self.widget_5); self.widget_2.setMinimumSize(QtCore.QSize(700, 200)); self.widget_2.setMaximumSize(QtCore.QSize(16777215, 200)); self.widget_2.setStyleSheet("background-color: rgb(38, 68, 93);")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.widget_2); self.horizontalLayout_11 = QtWidgets.QHBoxLayout(); self.horizontalLayout_9.addLayout(self.horizontalLayout_11); self.verticalLayout_2.addWidget(self.widget_2)
        self.verticalLayout_5.addLayout(self.verticalLayout_2)

        # 中部容器（聊天放置区）
        self.widget_4 = QtWidgets.QWidget(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        self.widget_4.setSizePolicy(sizePolicy)
        self.widget_4.setMaximumSize(QtCore.QSize(852, 982))  # 将在 AdaptedWindow 中解除限制
        self.widget_4.setObjectName("widget_4")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.widget_4)
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.widget_8 = QtWidgets.QWidget(self.widget_4); self.widget_8.setMinimumSize(QtCore.QSize(800, 300)); self.widget_8.setStyleSheet("background-color: rgb(38, 68, 93);")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout(self.widget_8); self.horizontalLayout_13 = QtWidgets.QHBoxLayout(); self.horizontalLayout_14.addLayout(self.horizontalLayout_13)
        self.verticalLayout_9.addWidget(self.widget_8)
        # 下部原报告区（默认隐藏）
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.widget_10 = QtWidgets.QWidget(self.widget_4)  # 表格容器
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.widget_10)
        self.tableWidget = QtWidgets.QTableWidget(self.widget_10); self.tableWidget.setRowCount(1); self.tableWidget.setColumnCount(1); self.tableWidget.horizontalHeader().setVisible(False); self.tableWidget.verticalHeader().setVisible(False)
        self.verticalLayout_8.addWidget(self.tableWidget); self.horizontalLayout_5.addWidget(self.widget_10)
        self.widget_9 = QtWidgets.QWidget(self.widget_4)   # PDF 容器
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.widget_9); self.verticalLayout_6 = QtWidgets.QVBoxLayout(); self.label_PDF = QtWidgets.QLabel(self.widget_9); self.verticalLayout_6.addWidget(self.label_PDF); self.verticalLayout_7.addLayout(self.verticalLayout_6)
        self.horizontalLayout_5.addWidget(self.widget_9)
        self.verticalLayout_9.addLayout(self.horizontalLayout_5)
        self.verticalLayout_10.addLayout(self.verticalLayout_9)

        # 右侧：患者信息 + 控制（保持与原名一致）
        self.widget_3 = QtWidgets.QWidget(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        self.widget_3.setSizePolicy(sizePolicy)
        self.verticalLayout_14 = QtWidgets.QVBoxLayout(self.widget_3)
        self.verticalLayout_13 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout()
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum); self.horizontalLayout_18.addItem(spacerItem4)
        self.label_12 = QtWidgets.QLabel(self.widget_3); font = QtGui.QFont(); font.setBold(True); self.label_12.setFont(font); self.label_12.setStyleSheet("color: rgb(238, 238, 236);")
        self.horizontalLayout_18.addWidget(self.label_12); spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum); self.horizontalLayout_18.addItem(spacerItem5)
        self.verticalLayout_11.addLayout(self.horizontalLayout_18)
        self.formLayout = QtWidgets.QFormLayout()
        self.label_13 = QtWidgets.QLabel(self.widget_3); self.label_14 = QtWidgets.QLabel(self.widget_3); self.label_15 = QtWidgets.QLabel(self.widget_3); self.label_16 = QtWidgets.QLabel(self.widget_3); self.label_17 = QtWidgets.QLabel(self.widget_3); self.label_18 = QtWidgets.QLabel(self.widget_3)
        self.comboBox = QtWidgets.QComboBox(self.widget_3); self.comboBox.addItem(""); self.comboBox.addItem(""); self.comboBox.addItem("")
        self.lineEdit = QtWidgets.QLineEdit(self.widget_3); self.lineEdit_2 = QtWidgets.QLineEdit(self.widget_3); self.lineEdit_3 = QtWidgets.QLineEdit(self.widget_3); self.lineEdit_4 = QtWidgets.QLineEdit(self.widget_3); self.lineEdit_5 = QtWidgets.QLineEdit(self.widget_3)
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_13); self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_14); self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_15); self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_16); self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_17); self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_18)
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.comboBox); self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit); self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_2); self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lineEdit_3); self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.lineEdit_4); self.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.lineEdit_5)
        self.verticalLayout_11.addLayout(self.formLayout); self.verticalLayout_13.addLayout(self.verticalLayout_11)
        self.verticalLayout_12 = QtWidgets.QVBoxLayout()
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum); self.horizontalLayout_15.addItem(spacerItem6)
        self.label_2 = QtWidgets.QLabel(self.widget_3); font = QtGui.QFont(); font.setBold(True); self.label_2.setFont(font); self.label_2.setStyleSheet("color: rgb(238, 238, 236);"); self.horizontalLayout_15.addWidget(self.label_2)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum); self.horizontalLayout_15.addItem(spacerItem7)
        self.label_11 = QtWidgets.QLabel(self.widget_3); self.label_11.setText(""); self.horizontalLayout_15.addWidget(self.label_11)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum); self.horizontalLayout_15.addItem(spacerItem8)
        self.verticalLayout_12.addLayout(self.horizontalLayout_15)
        self.line_5 = QtWidgets.QFrame(self.widget_3); self.line_5.setFrameShape(QtWidgets.QFrame.HLine); self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken); self.verticalLayout_12.addWidget(self.line_5)
        # 按钮与控件（名称保持原样）
        def mk_btn(name): b = QtWidgets.QPushButton(self.widget_3); b.setObjectName(name); self.verticalLayout_12.addWidget(b); return b
        self.pushButton = mk_btn("pushButton")
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout(); self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.pushButton_5 = QtWidgets.QPushButton(self.widget_3); self.verticalLayout_4.addWidget(self.pushButton_5)
        self.pushButton_8 = QtWidgets.QPushButton(self.widget_3); self.verticalLayout_4.addWidget(self.pushButton_8)
        self.pushButton_9 = QtWidgets.QPushButton(self.widget_3); self.verticalLayout_4.addWidget(self.pushButton_9)
        self.horizontalLayout_17.addLayout(self.verticalLayout_4)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.radioButton = QtWidgets.QRadioButton(self.widget_3); self.radioButton.setChecked(True); self.verticalLayout.addWidget(self.radioButton)
        self.radioButton_2 = QtWidgets.QRadioButton(self.widget_3); self.verticalLayout.addWidget(self.radioButton_2)
        self.radioButton_3 = QtWidgets.QRadioButton(self.widget_3); self.verticalLayout.addWidget(self.radioButton_3)
        self.horizontalLayout_17.addLayout(self.verticalLayout); self.verticalLayout_12.addLayout(self.horizontalLayout_17)
        self.pushButton_6 = mk_btn("pushButton_6")
        self.horizontalLayout_19 = QtWidgets.QHBoxLayout(); self.pushButton_2 = QtWidgets.QPushButton(self.widget_3); self.horizontalLayout_19.addWidget(self.pushButton_2); self.pushButton_7 = QtWidgets.QPushButton(self.widget_3); self.horizontalLayout_19.addWidget(self.pushButton_7); self.verticalLayout_12.addLayout(self.horizontalLayout_19)
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout(); spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum); self.horizontalLayout_16.addItem(spacerItem9)
        self.label = QtWidgets.QLabel(self.widget_3); self.horizontalLayout_16.addWidget(self.label)
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum); self.horizontalLayout_16.addItem(spacerItem10); self.verticalLayout_12.addLayout(self.horizontalLayout_16)
        self.pushButton_3 = mk_btn("pushButton_3"); self.progressBar = QtWidgets.QProgressBar(self.widget_3); self.progressBar.setProperty("value", 0); self.verticalLayout_12.addWidget(self.progressBar)
        self.pushButton_4 = mk_btn("pushButton_4"); self.progressBar_2 = QtWidgets.QProgressBar(self.widget_3); self.progressBar_2.setProperty("value", 0); self.verticalLayout_12.addWidget(self.progressBar_2)
        self.verticalLayout_13.addLayout(self.verticalLayout_12)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        spacerItem11 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum); self.horizontalLayout_8.addItem(spacerItem11)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(); spacerItem12 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum); self.horizontalLayout_7.addItem(spacerItem12)
        self.label_4 = QtWidgets.QLabel(self.widget_3); self.horizontalLayout_7.addWidget(self.label_4); self.label_7 = QtWidgets.QLabel(self.widget_3); self.horizontalLayout_7.addWidget(self.label_7)
        spacerItem13 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum); self.horizontalLayout_7.addItem(spacerItem13)
        self.verticalLayout_3.addLayout(self.horizontalLayout_7)
        self.dial = QtWidgets.QDial(self.widget_3); self.dial.setMinimumSize(QtCore.QSize(100, 100)); self.dial.setLayoutDirection(Qt.RightToLeft); self.dial.setMaximum(100); self.dial.setSingleStep(1); self.dial.setProperty("value", 20); self.dial.setOrientation(Qt.Vertical); self.dial.setNotchesVisible(True)
        self.verticalLayout_3.addWidget(self.dial); self.horizontalLayout_8.addLayout(self.verticalLayout_3)
        spacerItem14 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum); self.horizontalLayout_8.addItem(spacerItem14)
        self.verticalLayout_13.addLayout(self.horizontalLayout_8)
        self.gridLayout = QtWidgets.QGridLayout()
        self.label_3 = QtWidgets.QLabel(self.widget_3); self.horizontalSlider = QtWidgets.QSlider(self.widget_3); self.label_8 = QtWidgets.QLabel(self.widget_3); self.label_5 = QtWidgets.QLabel(self.widget_3); self.horizontalSlider_2 = QtWidgets.QSlider(self.widget_3); self.label_9 = QtWidgets.QLabel(self.widget_3); self.label_6 = QtWidgets.QLabel(self.widget_3); self.horizontalSlider_3 = QtWidgets.QSlider(self.widget_3); self.label_10 = QtWidgets.QLabel(self.widget_3); self.label_10.setMinimumSize(QtCore.QSize(50, 0))
        self.horizontalSlider.setMaximum(300); self.horizontalSlider.setProperty("value", 117); self.horizontalSlider.setOrientation(Qt.Horizontal); self.horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.horizontalSlider_2.setProperty("value", 7); self.horizontalSlider_2.setOrientation(Qt.Horizontal); self.horizontalSlider_2.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.horizontalSlider_3.setMaximum(10); self.horizontalSlider_3.setProperty("value", 5); self.horizontalSlider_3.setOrientation(Qt.Horizontal); self.horizontalSlider_3.setTickPosition(QtWidgets.QSlider.TicksAbove)
        self.gridLayout.addWidget(self.label_3, 0, 0, 1, 1); self.gridLayout.addWidget(self.horizontalSlider, 0, 1, 1, 1); self.gridLayout.addWidget(self.label_8, 0, 2, 1, 1); self.gridLayout.addWidget(self.label_5, 1, 0, 1, 1); self.gridLayout.addWidget(self.horizontalSlider_2, 1, 1, 1, 1); self.gridLayout.addWidget(self.label_9, 1, 2, 1, 1); self.gridLayout.addWidget(self.label_6, 2, 0, 1, 1); self.gridLayout.addWidget(self.horizontalSlider_3, 2, 1, 1, 1); self.gridLayout.addWidget(self.label_10, 2, 2, 1, 1)
        self.verticalLayout_13.addLayout(self.gridLayout)
        self.textEdit = QtWidgets.QTextEdit(self.widget_3); self.textEdit.setReadOnly(True); self.verticalLayout_13.addWidget(self.textEdit)
        self.verticalLayout_14.addLayout(self.verticalLayout_13)

        # 初始三块放入占位布局
        self.horizontalLayout_21.addWidget(self.widget_5)
        self.horizontalLayout_21.addWidget(self.widget_4)
        self.horizontalLayout_21.addWidget(self.widget_3)

        self.verticalLayout_15.addLayout(self.horizontalLayout_21)
        self.horizontalLayout_22.addLayout(self.verticalLayout_15)

        self.retranslateUi(Form); QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _t = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_t("Form", "基于深度学习的甲状腺自主扫查与分析系统"))
        self.label_19.setText(_t("Form", "基于深度学习的甲状腺自动扫查与分析系统"))
        self.label_12.setText(_t("Form", "患者信息"))
        self.label_13.setText(_t("Form", "姓名："))
        self.label_14.setText(_t("Form", "ID号："))
        self.label_15.setText(_t("Form", "性别："))
        self.label_16.setText(_t("Form", "年龄："))
        self.label_17.setText(_t("Form", "临床诊断："))
        self.label_18.setText(_t("Form", "病例描述："))
        self.comboBox.setItemText(0, _t("Form", "男"))
        self.comboBox.setItemText(1, _t("Form", "不清楚"))
        self.comboBox.setItemText(2, _t("Form", "女"))
        self.lineEdit.setText(_t("Form", "小小"))
        self.lineEdit_2.setText(_t("Form", "A000000"))
        self.lineEdit_3.setText(_t("Form", "24"))
        self.lineEdit_4.setText(_t("Form", "甲状腺结节"))
        self.lineEdit_5.setText(_t("Form", "体检"))
        for w,t in [(self.label_2,"状态灯"),(self.pushButton,"（1）启动自检"),(self.pushButton_5,"（2-1）关键点预览"),(self.pushButton_8,"（2-2）关键点发布"),
                    (self.pushButton_9,"（2-3）关键点清除"),(self.pushButton_6,"（3）压力调零"),(self.pushButton_2,"（4）自动扫查"),
                    (self.pushButton_7,"（4-1）中断扫查"),(self.label,"结果生成"),(self.pushButton_3,"（5）3D重建"),(self.pushButton_4,"（6）AI分析"),
                    (self.label_4,"探头压力"),(self.label_7,"2N"),(self.label_3,"P比例"),(self.label_8,"1.17"),(self.label_5,"I积分"),(self.label_9,"0.07"),(self.label_6,"D微分"),(self.label_10,"5")]:
            w.setText(_t("Form", t))


# ----------------------------- 2) 对话桥（两实现复用） ----------------------------- #
class ChatBridge(QObject):
    assistantMessage = pyqtSignal(str)
    systemMessage = pyqtSignal(str)

    @pyqtSlot(str)
    def onUserMessage(self, text: str):
        # TODO: 接入真实 LLM/Agent（HTTP/gRPC/SDK）
        reply = f"【助手】已收到：{text}"
        self.assistantMessage.emit(reply)


# ----------------------------- 3A) WebEngine 聊天部件 ----------------------------- #
class WebChatWidget(QtWidgets.QWidget):
    readyChanged = pyqtSignal(bool)
    def __init__(self, parent=None):
        super().__init__(parent)
        if not HAS_WEBENGINE:
            raise RuntimeError("QtWebEngine 不可用")
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

    def _on_load_finished(self, ok: bool):
        self._ready = True
        # 执行排队 JS
        while self._js_queue:
            js = self._js_queue.pop(0)
            self.view.page().runJavaScript(js)
        self.readyChanged.emit(True)

    # ---- 统一接口（全部走 JS+队列） ----
    def _js(self, script: str):
        if self._ready:
            self.view.page().runJavaScript(script)
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
  .bubble{max-width:72%;padding:10px 12px;border-radius:14px;border:1px solid var(--bd);background:rgba(255,255,255,.05);line-height:1.5}
  .user .bubble{background:rgba(65,234,212,.12);border-color:rgba(65,234,212,.45)}
  .system{color:var(--muted);text-align:center;font-size:12px;margin:6px 0}
  form{display:flex;gap:8px;padding:10px;border-top:1px solid var(--bd);background:rgba(255,255,255,.03)}
  input{flex:1;min-width:0;padding:10px;border-radius:10px;border:1px solid var(--bd);background:#0e2248;color:var(--fg)}
  button{padding:10px 14px;border-radius:10px;border:1px solid rgba(65,234,212,.45);background:rgba(65,234,212,.12);color:var(--fg);cursor:pointer}
</style>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script></head>
<body><div class="chat"><div id="scroll" class="scroll"></div>
<form id="form"><input id="input" type="text" placeholder="请输入指令…" autocomplete="off"/><button>发送</button></form></div>
<script>
  function add(role,text){const s=document.getElementById('scroll');const r=document.createElement('div');r.className='row '+role;
    const b=document.createElement('div');b.className='bubble';b.textContent=text;r.appendChild(b);s.appendChild(r);s.scrollTop=s.scrollHeight;}
  function addSys(text){const s=document.getElementById('scroll');const d=document.createElement('div');d.className='system';d.textContent=text;s.appendChild(d);s.scrollTop=s.scrollHeight;}
  function addImage(role,dataUrl){const s=document.getElementById('scroll');const r=document.createElement('div');r.className='row '+role;
    const b=document.createElement('div');b.className='bubble';const img=document.createElement('img');img.src=dataUrl;img.style.maxWidth='72%';img.style.borderRadius='10px';
    b.appendChild(img);r.appendChild(b);s.appendChild(r);s.scrollTop=s.scrollHeight;}

  // WebChannel 仅用于把“网页输入”传给 Python
  new QWebChannel(qt.webChannelTransport,function(channel){
    const pyBridge = channel.objects.pyBridge;
    document.getElementById('form').addEventListener('submit',function(e){
      e.preventDefault();const i=document.getElementById('input');const t=(i.value||'').trim();if(!t)return;
      add('user',t); i.value=''; if(pyBridge&&pyBridge.onUserMessage) pyBridge.onUserMessage(t);
    });
  });
  addSys('智能对话（WebEngine）已就绪。')
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

    # ---- 统一外部接口 ----
    def send_user_text(self, text: str): self._append_user_text(text)
    def send_assistant_text(self, text: str): self._append_assistant_text(text)
    def send_user_image(self, path_or_b64: str): self._append_image(path_or_b64, align_right=True)
    def send_assistant_image(self, path_or_b64: str): self._append_image(path_or_b64, align_right=False)
    def send_system(self, text: str): self._append_system(text)


# ----------------------------- 4) 适配主窗口（含 QSplitter 与最小宽度） ----------------------------- #
class AdaptedWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form(); self.ui.setupUi(self)
        self._style()
        self._mount_chat_widget()
        self._rebuild_splitter()
        self._wire_buttons()

    def _style(self):
        self.setWindowTitle("基于深度学习的甲状腺自动扫查与分析系统（统一聊天接口版）")
        self.setStyleSheet("""
            QWidget { color: #e5eef8; font-size: 14px; }
            QLabel#label_RGB, QLabel#label_20, QLabel#label_US, QLabel#label_seg, QLabel#label_PDF {
                background:#26445d; border:1px solid rgba(255,255,255,.06); border-radius:8px;
            }
            QProgressBar { background:#0e2248; border:1px solid rgba(255,255,255,.12); border-radius:6px; height:10px; }
            QProgressBar::chunk { background:#41ead4; border-radius:6px; }
            QPushButton { background:rgba(65,234,212,.10); border:1px solid rgba(65,234,212,.35); border-radius:8px; padding:6px 10px; }
            QPushButton:hover { background:rgba(65,234,212,.20); }
            QFrame[frameShape="4"] { border-top:1px solid rgba(65,234,212,.25); }
        """)

    def _clear_layout(self, layout: QtWidgets.QLayout):
        if not layout: return
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w: w.setParent(None)
            else: self._clear_layout(item.layout())

    def _mount_chat_widget(self):
        # 彻底隐藏原报告区并释放空间
        for wname in ("widget_10", "widget_9"):
            w = getattr(self.ui, wname, None)
            if w is not None:
                w.hide()
        self._clear_layout(self.ui.horizontalLayout_5)

        # 清空聊天安放容器并安装聊天组件
        self._clear_layout(self.ui.horizontalLayout_14)
        if HAS_WEBENGINE:
            try:
                self.chat = WebChatWidget(self.ui.widget_8)
            except Exception:
                self.chat = NativeChatWidget(self.ui.widget_8)
        else:
            self.chat = NativeChatWidget(self.ui.widget_8)
        self.ui.horizontalLayout_14.addWidget(self.chat)
        self.chat.send_system("3D/报告区已替换为智能对话。")

        # 确保中部容器可扩展
        self.ui.widget_4.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.ui.widget_4.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.ui.widget_8.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.ui.verticalLayout_9.setStretch(0, 1)

    def _rebuild_splitter(self):
        left, center, right = self.ui.widget_5, self.ui.widget_4, self.ui.widget_3
        host = self.ui.horizontalLayout_21; self._clear_layout(host)

        self.splitter = QtWidgets.QSplitter(Qt.Horizontal, self)
        self.splitter.setHandleWidth(6)
        self.splitter.setStyleSheet("QSplitter::handle{background:rgba(255,255,255,.08);}")

        # 最小宽度（可根据实际再调）
        left.setMinimumWidth(520)
        center.setMinimumWidth(560)
        right.setMinimumWidth(420)

        for w in (left, center, right):
            w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.splitter.addWidget(left); self.splitter.addWidget(center); self.splitter.addWidget(right)
        for i in range(3):
            self.splitter.setCollapsible(i, False)

        # 相对分配比例：中间更多
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 5)
        self.splitter.setStretchFactor(2, 3)

        host.addWidget(self.splitter)
        # show 之后设置像素宽度，确保初始可见良好
        QtCore.QTimer.singleShot(0, lambda: self.splitter.setSizes([520, 760, 420]))

    def _wire_buttons(self):
        # 将右侧按钮联动到聊天区的系统提示，便于流程日志化
        sigs = []
        if hasattr(self.ui, 'pushButton'):   # （1）启动自检
            self.ui.pushButton.clicked.connect(lambda: self.chat.send_system("（1）启动自检：正在检查传感器与执行器状态…"))
        if hasattr(self.ui, 'pushButton_5'):
            self.ui.pushButton_5.clicked.connect(lambda: self.chat.send_system("（2-1）关键点预览：已更新预览。"))
        if hasattr(self.ui, 'pushButton_8'):
            self.ui.pushButton_8.clicked.connect(lambda: self.chat.send_system("（2-2）关键点发布：已发布关键点。"))
        if hasattr(self.ui, 'pushButton_9'):
            self.ui.pushButton_9.clicked.connect(lambda: self.chat.send_system("（2-3）关键点清除：已清除关键点。"))
        if hasattr(self.ui, 'pushButton_6'):
            self.ui.pushButton_6.clicked.connect(lambda: self.chat.send_system("（3）压力调零：请轻放探头，系统将重新标定压力零点。"))
        if hasattr(self.ui, 'pushButton_2'):
            self.ui.pushButton_2.clicked.connect(lambda: self.chat.send_system("（4）自动扫查：流程已启动，请保持患者体位稳定。"))
        if hasattr(self.ui, 'pushButton_7'):
            self.ui.pushButton_7.clicked.connect(lambda: self.chat.send_system("（4-1）中断扫查：正在安全回撤…"))
        if hasattr(self.ui, 'pushButton_3'):
            self.ui.pushButton_3.clicked.connect(lambda: self.chat.send_system("（5）3D 重建：功能已由智能对话替代。"))
        if hasattr(self.ui, 'pushButton_4'):
            self.ui.pushButton_4.clicked.connect(lambda: self.chat.send_system("（6）AI 分析：请在对话中直接下达分析指令。"))


# ----------------------------- 5) 入口 ----------------------------- #
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = AdaptedWindow()
    w.resize(1600, 900)
    w.chat.send_system('MedAI实验室')
    w.chat.send_user_image(r'/home/usai/auto_RUSS/R_UI/keypoint_detection/person.png')
    w.chat.send_user_text('这是什么？')
    w.chat.send_assistant_image(r'/home/usai/auto_RUSS/R_UI/keypoint_detection/person.png')
    w.chat.send_assistant_text('a women')

    w.show()
    sys.exit(app.exec_())
