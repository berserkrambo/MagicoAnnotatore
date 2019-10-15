import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QLabel, QProgressBar, QVBoxLayout, \
    QHBoxLayout, QLabel, QGroupBox, QRadioButton
from PyQt5 import QtCore, uic
from PyQt5.QtGui import QIcon, QPixmap, QImage, QPainter, QPen, QColor
import threading
import cv2
import time
import numpy as np

from utils import VideoLoader, Annotator, readColors, hls2rgb, re_scale_coords


class App(QWidget):
    def __init__(self):
        # super(threading.Thread, self).__init__()
        super(QWidget, self).__init__()
        self.vido_loader = None
        self.bbox_selected = None
        self.current_img = None
        self.n_classes = 6
        self.colors_class_dict = readColors()
        self.initPosClick()
        self.initUI()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Q:
            if self.vido_loader is not None:
                self.annotator.save_all()
            self.deleteLater()
        elif event.key() == QtCore.Qt.Key_N:
            if self.vido_loader is not None:
                self.current_img = self.get_next_img()
                if self.current_img is not None:
                    self.load_pix_from_buff()
                    self.annotator.save_current()
                    self.annotator.current_frame += 1
                else:
                    print("saving....")
                    self.annotator.save_all()
                    self.reset_video()
        elif event.key() == QtCore.Qt.Key_Backslash:
            if self.vido_loader is not None and len(self.annotator.frame_objs) > 0:
                keys = list(self.annotator.frame_objs)
                if self.bbox_selected is None:
                    self.bbox_selected = 0
                else:
                    self.bbox_selected += 1
                    if self.bbox_selected >= keys.__len__():
                        self.bbox_selected = 0
                self.drawBoxbyClass(tobg=True)
        elif event.key() == QtCore.Qt.Key_Escape:
            self.bbox_selected = None
            self.drawBoxbyClass()
        elif event.key() == QtCore.Qt.Key_X:
            if self.vido_loader is not None and len(self.annotator.frame_objs) > 0 and self.bbox_selected is not None:
                k = list(self.annotator.frame_objs)[self.bbox_selected]
                self.annotator.frame_objs.pop(k)
                self.bbox_selected = None
                self.load_pix_from_buff()
                self.drawBoxbyClass()

        event.accept()

    def reset_video(self):
        self.current_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.load_pix_from_buff()
        self.pbar.setValue(0)
        self.vido_loader = None
        self.setCLassGroupVisibility(False)

    def get_next_img(self):
        img = self.vido_loader.get_next_img()
        if img is None:
            return None
        self.pbar.setValue(self.annotator.current_frame / self.annotator.total_frames * 100)
        self.annotator.current_frame += 1
        return img

    def load_pix(self):
        img = cv2.cvtColor(cv2.imread('/home/rgasparini/Pictures/giova.png'), cv2.COLOR_BGR2RGB)
        qscale = 1280 / img.shape[1]
        img = cv2.resize(img, (0, 0), fx=qscale, fy=qscale)
        height, width, channel = img.shape
        bytesPerLine = channel * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.pixmap_image = QPixmap.fromImage(qImg)
        self.label.setPixmap(self.pixmap_image)
        self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.label.update()

    def load_pix_from_buff(self):
        height, width, channel = self.current_img.shape
        bytesPerLine = channel * width
        qImg = QImage(self.current_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.pixmap_image = QPixmap.fromImage(qImg)
        self.label.setPixmap(self.pixmap_image)
        self.label.update()

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        video_file, _ = QFileDialog.getOpenFileName(self, "Selezionare un file video", "",
                                                    "MP4 video files (*.mp4);;AVI video files (*.avi)", options=options)
        if video_file is not "":
            self.vido_loader = VideoLoader(video_file)
            self.annotator = Annotator(self.vido_loader.tot_frames,
                                       video_file.replace(".avi", ".txt").replace(".mp4", ".txt"),
                                       self.vido_loader.meta)
            while not self.vido_loader.batch_loaded:
                time.sleep(0.5)
            self.current_img = self.get_next_img()
            self.load_pix_from_buff()
            self.setCLassGroupVisibility(True)

    def initPosClick(self):
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
        self.click = 0

    def getCheckedClass(self):
        for n, r in enumerate(self.radioclass_List):
            if r.isChecked():
                return n

    def drawBoxbyClass(self, tobg=None):
        # create painter instance with pixmap
        painterInstance = QPainter(self.pixmap_image)

        for k, v in self.annotator.frame_objs.items():
            h,l,s = self.colors_class_dict[v]
            if tobg and k == list(self.annotator.frame_objs)[self.bbox_selected]:
                s = 80
            # set rectangle color and thickness
            r, g, b = hls2rgb((h,l,s))

            penRectangle = QPen(QColor(r, g, b))
            penRectangle.setWidth(3)

            # draw rectangle on painter
            painterInstance.setPen(penRectangle)
            # x1, y1, x2, y2 = k[1] * self.annotator.scale_factor, k[2] * self.annotator.scale_factor, k[3] * self.annotator.scale_factor, k[4] * self.annotator.scale_factor
            x1, y1, x2, y2 = re_scale_coords(list(k[1:5]), self.vido_loader.meta)
            painterInstance.drawRect(x1, y1, x2 - x1, y2 - y1)

            # set pixmap onto the label widget
            self.label.setPixmap(self.pixmap_image)
            self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

        self.label.update()

    def getPos(self, event):
        if self.vido_loader is None:
            return
        x = event.pos().x()
        y = event.pos().y()
        self.click += 1
        if self.click == 1:
            self.x1 = x
            self.y1 = y
        elif self.click == 2:
            self.x2 = x
            self.y2 = y
            self.click = 0
            self.x1, self.x2 = min(self.x1, self.x2), max(self.x1, self.x2)
            self.y1, self.y2 = min(self.y1, self.y2), max(self.y1, self.y2)

            self.annotator.update_obj([self.annotator.current_frame, self.x1, self.y1, self.x2, self.y2],
                                      self.getCheckedClass())
            print(f"frame {self.annotator.current_frame}, pos {self.x1},{self.y1} : {self.x2},{self.y2}")

            self.drawBoxbyClass()

    def setCLassGroupVisibility(self, val=False):
        self.class_group_label.setVisible(val)
        self.class_group_label.update()
        for rdi_i, rdi in enumerate(self.radioclass_List):
            rdi.setVisible(val)
            r,g,b = hls2rgb(self.colors_class_dict[rdi_i])
            rdi.setStyleSheet(f'color: {QColor(r,g,b).name()}')
            rdi.update()

    def initUI(self):
        self.title = 'Magico Annotatore Ferroviario'
        self.left = 10
        self.top = 10
        self.width = 1800
        self.height = 850

        self.hlay0 = QHBoxLayout(self)

        # load button
        self.loadButton = QPushButton()
        self.loadButton.setText("Carica Video")
        self.loadButton.setToolTip('Clicca per caricare un video')
        self.loadButton.setMaximumWidth(200)
        self.loadButton.clicked.connect(self.openFileNameDialog)

        # instructions label
        self.instrLabel = QLabel()
        self.instrLabel.move(1350, 120)
        self.instrLabel.setStyleSheet('color: red')
        self.instrLabel.setText("Istruzioni:\n "
                                "1: premere 'carica video'\n"
                                "per aprire un video ed aspettare\n"
                                "che venga caricato il primo frame\n"
                                "2: Se ci sono oggetti presenti\n"
                                " annotarli cliccando con il mouse\n"
                                "frame successivo\n\n"
                                "Funzionalità:\n"
                                "'n' avanti di un frame\n"
                                "'\\' scorre le annotazioni\n"
                                "'x' elimina l'annotazione\n"
                                "'Esc' deseleziona l'annotazione\n"
                                "'q' salva ed esce")

        # image label
        self.label = QLabel()
        self.label.setMaximumSize(1280, 720)
        self.label.setMinimumSize(1280, 720)
        self.label.mousePressEvent = self.getPos

        # video annotation progress bar
        self.pbar = QProgressBar()
        # self.pbar.move(100, 730)
        self.pbar.setRange(0, 100)
        self.pbar.setMinimumSize(500, 10)
        self.pbar.setMaximumWidth(1280)

        # classes
        self.radioclass_List = []
        for rdi in range(self.n_classes):
            self.radioclass_List.append(QRadioButton(f"Classe {rdi}"))
        self.radioclass_List[0].setChecked(True)

        # layouts to put all together
        self.l_vlay0 = QVBoxLayout(self)
        self.r_vlay0 = QVBoxLayout(self)
        self.l_vlay0.addWidget(self.label)
        self.l_vlay0.addWidget(self.pbar)
        self.r_vlay0.addWidget(self.instrLabel)
        self.r_vlay0.addWidget(self.loadButton)
        self.r_vlay0.addSpacing(30)
        self.class_group_label = QLabel("Classi:")
        self.r_vlay0.addWidget(self.class_group_label)
        for rdi in self.radioclass_List:
            self.r_vlay0.addWidget(rdi)
        self.setCLassGroupVisibility(False)

        self.r_vlay0.setAlignment(QtCore.Qt.AlignTop)

        self.hlay0.addLayout(self.l_vlay0)
        self.hlay0.addLayout(self.r_vlay0)

        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setMaximumSize(1600, 800)
        self.setMinimumSize(1600, 800)

        self.reset_video()
        self.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
