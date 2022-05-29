import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QDesktopWidget, QApplication, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap

import cv2
import numpy as np
import pandas as pd
#from PIL import Image
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
#from sklearn.decomposition import PCA
#from image_split import imgSplit


input_dir = 'logo_test/'
test_image = ''


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.resize(1200, 800)
        self.center()
        self.setWindowTitle('Logo Retrieval')
        self.show()
        self.pred_images = []
        self.image_paths = []
        self.knn = NearestNeighbors()
        #self.pca = PCA(n_components = 128)

        model = tf.keras.models.load_model('model/ResNet50V2Logos.h5')
        model = tf.keras.models.Sequential(model.layers[:-2])
        
        model = tf.keras.Sequential([
                model,
                tf.keras.layers.GlobalMaxPooling2D()
        ])
        
        self.model = model
        self.loadData()

    def initUI(self):
        self.label_0 = QLabel(self)
        self.label_1 = QLabel(self)
        self.label_2 = QLabel(self)
        self.label_3 = QLabel(self)
        self.label_4 = QLabel(self)
        self.label_5 = QLabel(self)
        self.label_dist1 = QLabel(self)
        self.label_dist2 = QLabel(self)
        self.label_dist3 = QLabel(self)
        self.label_dist4 = QLabel(self)
        self.label_dist5 = QLabel(self)
        self.label_rate = QLabel(self)

        self.setLabel(0, 400, 400, 'Test image')
        self.setLabel(1, 200, 150, 'Top 1')
        self.setLabel(2, 200, 150, 'Top 2')
        self.setLabel(3, 200, 150, 'Top 3')
        self.setLabel(4, 200, 150, 'Top 4')
        self.setLabel(5, 200, 150, 'Top 5')
        self.label_dist1.setText('Top1 distance: N/A')
        self.label_dist2.setText('Top2 distance: N/A')
        self.label_dist3.setText('Top3 distance: N/A')
        self.label_dist4.setText('Top4 distance: N/A')
        self.label_dist5.setText('Top5 distance: N/A')
        self.label_rate.setText('Top5 Accuracy rate: N/A')

        btn_select = QPushButton('Select')
        btn_search = QPushButton('Search')

        btn_select.clicked.connect(self.onClickedBtnSelect)
        btn_search.clicked.connect(self.onClickedBtnSearch)

        hbox_btn = QHBoxLayout()
        hbox_btn.addWidget(btn_select)
        hbox_btn.addWidget(btn_search)

        vbox_left = QVBoxLayout()
        vbox_left.addStretch(1)
        vbox_left.addWidget(self.label_0)
        vbox_left.addLayout(hbox_btn)
        vbox_left.addStretch(1)

        vbox_right = QVBoxLayout()
        vbox_right.addStretch(1)
        vbox_right.addWidget(self.label_1)
        vbox_right.addWidget(self.label_dist1)
        vbox_right.addWidget(self.label_2)
        vbox_right.addWidget(self.label_dist2)
        vbox_right.addWidget(self.label_3)
        vbox_right.addWidget(self.label_dist3)
        vbox_right.addWidget(self.label_4)
        vbox_right.addWidget(self.label_dist4)
        vbox_right.addWidget(self.label_5)
        vbox_right.addWidget(self.label_dist5)
        vbox_right.addWidget(self.label_rate)
        vbox_right.addStretch(1)

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addLayout(vbox_left)
        hbox.addStretch(1)
        hbox.addLayout(vbox_right)
        hbox.addStretch(1)

        self.setLayout(hbox)

    def center(self):
        qr = self.frameGeometry()  # Get window
        cp = QDesktopWidget().availableGeometry().center()  # Get screen center
        qr.moveCenter(cp)  # Show on the center
        self.move(qr.topLeft())

    def setLabel(self, id, width, height, text):
        label = self.label_0
        if id == 1:
            label = self.label_1
        elif id == 2:
            label = self.label_2
        elif id == 3:
            label = self.label_3
        elif id == 4:
            label = self.label_4
        elif id == 5:
            label = self.label_5
        label.setText(text)
        label.setFixedSize(width, height)
        label.setStyleSheet("QLabel{background:white;}")
        label.setAlignment(Qt.AlignCenter)
        label.setScaledContents(True)

    def onClickedBtnSelect(self):
        image_file, _ = QFileDialog.getOpenFileName(self, 'Open file', input_dir, 'Image files (*.jpg *.gif *.png *.jpeg)')
        self.label_0.setPixmap(QPixmap(image_file))
        global test_image
        test_image = image_file
        print(test_image)

    def onClickedBtnSearch(self):
        if test_image == '':
            return
        images = []
        
        image_np = cv2.imread(test_image, 0)
        #image_np = cv2.medianBlur(image_np, 5)
        #_, image_np = cv2.threshold(image_np, 127, 255, cv2.THRESH_TOZERO)
        #image_np = cv2.adaptiveThreshold(image_np, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
        image_np = cv2.resize(cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR), (224, 224))
        
        #image_np = imgSplit(cv2.resize(cv2.imread(test_image), (224, 224)))
        #image_np = cv2.resize(cv2.imread(test_image), (224, 224))
        images.append(image_np)
        images = np.array(images) / 255

        img_features = self.model.predict(images)
        #img_features = np.concatenate((self.pred_images, img_features))
        #img_features = self.pca.fit_transform(img_features)

        #self.knn.fit(img_features)
        predList = self.knn.kneighbors([img_features[0]], 5, return_distance = False)[0]
        distList = self.knn.kneighbors([img_features[0]], 5, return_distance = True)[0]
        print(predList)

        self.label_1.setPixmap(QPixmap(self.image_paths[predList[0]].replace("\\", "/")))
        self.label_2.setPixmap(QPixmap(self.image_paths[predList[1]].replace("\\", "/")))
        self.label_3.setPixmap(QPixmap(self.image_paths[predList[2]].replace("\\", "/")))
        self.label_4.setPixmap(QPixmap(self.image_paths[predList[3]].replace("\\", "/")))
        self.label_5.setPixmap(QPixmap(self.image_paths[predList[4]].replace("\\", "/")))
        self.label_dist1.setText('Top1 distance: ' + str(distList[0][0]))
        self.label_dist2.setText('Top2 distance: ' + str(distList[0][1]))
        self.label_dist3.setText('Top3 distance: ' + str(distList[0][2]))
        self.label_dist4.setText('Top4 distance: ' + str(distList[0][3]))
        self.label_dist5.setText('Top5 distance: ' + str(distList[0][4]))

        if len(self.image_paths) == 102:
            top5s = []
            for i in predList:
                top5s.append(list(self.image_paths[i])[10:13])
            if list(self.image_paths[predList[0]])[10:13] in top5s:
                self.label_rate.setText('Top5 Accuracy rate: 1.000')
            else:
                self.label_rate.setText('Top5 Accuracy rate: 0.000')

        # plt.imshow(image_np)
        # plt.show()

    def loadData(self):
        self.pred_images = pd.read_csv('preprocessed/pred_data_all_rn50v2.csv').values
        self.image_paths = pd.read_csv('preprocessed/img_paths_all.csv').values.flatten()
        self.knn.fit(self.pred_images)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
