# -*- coding: UTF-8 –*-
import tensorflow as tf
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
from PIL import Image
import numpy as np
import shutil
class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('.\\images\\logo.png'))
        self.setWindowTitle('垃圾智能分类系统')
        self.model = tf.keras.models.load_model(".\\models\\mobilenet_changedata_epoch30.h5")  # todo 修改为自己的模型路径
        self.to_predict_name = "./images/box.png"
        self.class_names = ['其他垃圾_PE塑料袋','其他垃圾_串串竹签','其他垃圾_卫生纸','其他垃圾_口罩','其他垃圾_餐盒',
                            '厨余垃圾_果皮','厨余垃圾_残渣剩饭','厨余垃圾_肉类','厨余垃圾_蔬菜','厨余垃圾_面包',
                            '可回收物_不锈钢制品','可回收物_塑料制品','可回收物_布制品','可回收物_果冻杯','可回收物_泡沫板',
                            '可回收物_玻璃制品','可回收物_盒子','可回收物_纸制品','可回收物_金属制品','可回收物_铝制用品',
                            '有害垃圾_温度计','有害垃圾_灯', '有害垃圾_电池','有害垃圾_胶水','有害垃圾_药片']
        self.resize(900, 700)
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        font = QFont('宋体', 15)

        left_widget = QWidget()
        left_layout = QVBoxLayout()
        img_title = QLabel("分类系统")
        img_title.setFont(font)
        img_title.setAlignment(Qt.AlignCenter)
        self.img_label = QLabel()
        img_init = cv2.imread(self.to_predict_name)
        h, w, c = img_init.shape
        scale = 400 / h
        img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
        cv2.imwrite("images\\show.png", img_show)
        img_init = cv2.resize(img_init, (224, 224))
        cv2.imwrite('images\\target.png', img_init)
        self.img_label.setPixmap(QPixmap("images/show.png"))
        left_layout.addWidget(img_title)
        left_layout.addWidget(self.img_label, 1, Qt.AlignCenter)
        # left_layout.setAlignment(Qt.AlignCenter)
        left_widget.setLayout(left_layout)

        right_widget = QWidget()
        right_layout = QVBoxLayout()
        btn_change = QPushButton("上传图片")
        btn_change.clicked.connect(self.change_img)
        btn_change.setFont(font)
        btn_predict = QPushButton(" 开始识别 ")
        btn_predict.setFont(font)
        btn_predict.clicked.connect(self.predict_img)

        label_result = QLabel('垃圾种类')
        self.result = QLabel("等待识别")
        label_result.setFont(QFont('宋体', 16))
        self.result.setFont(QFont('宋体', 24))

        label_result_f = QLabel('物品名称')
        self.result_f = QLabel("等待识别")

        self.label_info = QTextEdit()
        self.label_info.setFont(QFont('宋体', 12))

        label_result_f.setFont(QFont('宋体', 16))
        self.result_f.setFont(QFont('宋体', 24))

        right_layout.addStretch()
        right_layout.addWidget(label_result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(label_result_f, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.result_f, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(self.label_info, 0, Qt.AlignCenter)
        right_layout.addStretch()
        right_layout.addWidget(btn_change)
        right_layout.addWidget(btn_predict)
        right_layout.addStretch()
        right_widget.setLayout(right_layout)

        # 关于页面
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('欢迎使用垃圾分类系统')
        about_title.setFont(QFont('宋体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('images/about.png'))
        about_img.setAlignment(Qt.AlignCenter)
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        about_widget.setLayout(about_layout)

        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget)
        main_widget.setLayout(main_layout)
        self.addTab(main_widget, '主页')
        self.addTab(about_widget, '关于')
        self.setTabIcon(0, QIcon('images/主页面.png'))
        self.setTabIcon(1, QIcon('images/关于.png'))

    # 上传图片
    def change_img(self):
        openfile_name = QFileDialog.getOpenFileName(self, 'chose files', '', 'Image files(*.jpg *.png *jpeg)')
        img_name = openfile_name[0]
        if img_name == '':
            pass
        else:
            target_image_name = "./images/tmp_single" + img_name.split(".")[-1]
            shutil.copy(img_name, target_image_name)
            self.to_predict_name = target_image_name
            img_init = cv2.imread(self.to_predict_name)
            h, w, c = img_init.shape
            scale = 400 / h
            img_show = cv2.resize(img_init, (0, 0), fx=scale, fy=scale)
            cv2.imwrite("./images/show.png", img_show)
            img_init = cv2.resize(img_init, (224, 224))
            cv2.imwrite('./images/target.png', img_init)
            self.img_label.setPixmap(QPixmap("./images/show.png"))

    # 预测图片
    def predict_img(self):
        img = Image.open('./images/target.png')
        img = np.asarray(img)
        outputs = self.model.predict(img.reshape(1, 224, 224, 3))
        result_index = int(np.argmax(outputs))
        result = self.class_names[result_index]
        names = result.split("_")
        # print(result)
        if names[0] == "厨余垃圾":
            self.label_info.setText(
                "厨余垃圾是指居民日常生活及食品加工、饮食服务、单位供餐等活动中产生的垃圾，包括丢弃不用的菜叶、剩菜、剩饭、果皮、蛋壳、茶渣、骨头等。由于厨余垃圾含有极高的水分与有机物，很容易腐坏，产生恶臭。经过妥善处理和加工，可转化为新的资源，高有机物含量的特点使其经过严格处理后可作为肥料、饲料，也可产生沼气用作燃料或发电，油脂部分则可用于制备生物燃料。")
        if names[0] == "有害垃圾":
            self.label_info.setText(
                "有害垃圾指对人体健康或者自然环境造成直接或者潜在危害的生活废弃物。常见的有害垃圾包括废灯管、废油漆、杀虫剂、废弃化妆品、过期药品、废电池、废灯泡、废水银温度计等，有害垃圾需按照特殊正确的方法安全处理，一般需要经过特殊的处理之后才可以进行焚烧，堆肥，填埋处理")
        if names[0] == "可回收物":
            self.label_info.setText(
                " 根据《城市生活垃圾分类及其评价标准》行业标准以及参考德国垃圾分类法，可回收物是指适宜回收循环使用和资源利用的废物。主要包括：纸类，塑料，金属，玻璃，织物等。主要的处理方式有：1.垃圾再生法；2.垃圾焚烧法；3.垃圾堆肥法；4.垃圾生物降解法。")
        if names[0] == "其他垃圾":
            self.label_info.setText(
                "其他垃圾指危害比较小，没有再次利用的价值的垃圾，其他垃圾包括砖瓦陶瓷、渣土、卫生间废纸、瓷器碎片、动物排泄物、一次性用品等难以回收的废弃物。一般都采取填埋、焚烧、卫生分解等方法处理，部分还可以使用生物分解的方法解决")
        self.result.setText(names[0])
        self.result_f.setText(names[1])

    # 窗口关闭事件
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     '退出',
                                     "是否要退出程序？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    x = MainWindow()
    x.show()
    sys.exit(app.exec_())
