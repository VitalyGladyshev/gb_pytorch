from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtPrintSupport import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *

import os
import sys
import time

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

import res_img
from camera_window import Ui_MainWindow

import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# For webcam input:
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

arm_marks = ['WRIST',
             'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
             'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
             'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
             'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
             'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']

cam_index_list = []
for i in range(6):
    cam_tmp = cv2.VideoCapture(i)
    if cam_tmp.isOpened():
        cam_index_list.append(i)
        cam_tmp.release()

cam = 0
if cam_index_list:
    cam = cv2.VideoCapture(cam_index_list[0])
    cam.set(cv2.CAP_PROP_FPS, 24)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)


# Класс главного окна приложения
class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        if not cam_index_list:
            print(f"cam is not opened: {cam}")
            qDebug("No camera")

        self.save_path = ""

        # Set the default camera.
        self.select_camera(0)

        # Setup tools
        self.ui.camera_toolbar.setIconSize(QSize(14, 14))
        # camera_toolbar = QToolBar("Camera")
        # camera_toolbar.setIconSize(QSize(14, 14))
        # self.addToolBar(camera_toolbar)

        img_fld = QIcon(":/img/images/blue-folder-horizontal-open.png")
        img_cam = QIcon(":/img/images/camera-black.png")

        photo_action = QAction(img_cam, "Скриншот...", self)
        # photo_action = QAction(QIcon(os.path.join('images', 'camera-black.png')), "Take photo...", self)
        photo_action.setStatusTip("Сделать скриншот")
        photo_action.triggered.connect(self.take_photo)
        self.ui.camera_toolbar.addAction(photo_action)

        change_folder_action = QAction(img_fld, "Папка для скриншотов...", self)
        # change_folder_action = QAction(QIcon(os.path.join('images', 'blue-folder-horizontal-open.png')),
        #                                "Change save location...", self)
        change_folder_action.setStatusTip("Изменить папку для скриншотов")
        change_folder_action.triggered.connect(self.change_folder)
        self.ui.camera_toolbar.addAction(change_folder_action)

        camera_selector = QComboBox()
        if cam_index_list:
            for ind in cam_index_list:
                if cam.isOpened():
                    camera_selector.addItem(f"Камера {ind+1}")
        else:
            camera_selector.addItem("Нет камеры")
        camera_selector.currentIndexChanged.connect(self.select_camera)
        self.ui.camera_toolbar.addWidget(camera_selector)

        self.setWindowTitle("Распознавание жестов")
        self.show()

    def select_camera(self, ind):
        global cam, cam_index_list
        if cam_index_list:
            cam.release()
            cam = cv2.VideoCapture(cam_index_list[ind])
            cam.set(cv2.CAP_PROP_FPS, 24)
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.save_seq = 0

    def take_photo(self):
        if cam_index_list:
            timestamp = time.strftime("%d-%b-%Y-%H_%M_%S")
            ret, frame = cam.read()
            cv2.imwrite(os.path.join(self.save_path, f"camera-{self.save_seq}-{timestamp}.jpg"), frame)
            self.save_seq += 1

    def change_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Папка для скриншотов", "")
        if path:
            self.save_path = path
            self.save_seq = 0

    # Показываем кадр в окне
    def show_frame_slot(self, frame):
        self.ui.label_for_cam.setPixmap(frame)


# Класс детектирования и обработки лица с веб-камеры
class FaceAndHandDetector(QThread):
    frame_update_signal = pyqtSignal(QPixmap)

    def __init__(self):
        QThread.__init__(self)

        self.frame = 0
        self.mtcnn = MTCNN()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # print(self.device)
        self.frame_counter = 0
        self.prev_frame_counter = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.fps_count)
        self.timer.start(1000)

    # Функция рисования прямоугольника лица
    def draw_face(self, frame, boxes, probs):   # , landmarks
        try:
            cnt = 0
            for box, prob in zip(boxes, probs):   # , ld , landmarks
                cnt += 1
                print(f"Лицо {cnt} box: {box} prob: {prob:.4f}")
                # Рисуем обрамляющий прямоугольник лица на кадре
                cv2.rectangle(frame,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (0, 0, 255),
                              thickness=2)
        except Exception as e:
            print('Error in _draw')
            print(f'error : {e}')
        return frame

    # Функция рисования прямоугольников рук
    def draw_hand(self, frame, hand_landmarks):
        # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        max_x = max_y = 0
        min_x = min_y = 65535
        for mark in hand_landmarks.landmark:
            if mark.x > max_x:
                max_x = mark.x
            if mark.x < min_x:
                min_x = mark.x
            if mark.y > max_y:
                max_y = mark.y
            if mark.y < min_y:
                min_y = mark.y
        max_x = round(max_x * IMAGE_WIDTH) + 15
        min_x = round(min_x * IMAGE_WIDTH) - 15
        max_y = round(max_y * IMAGE_HEIGHT) + 15
        min_y = round(min_y * IMAGE_HEIGHT) - 15
        print(f"\tmax_x: {max_x} min_x: {min_x} max_y: {max_y} min_y: {min_y}")
        # Рисуем обрамляющий прямоугольник руки на кадре
        cv2.rectangle(frame,
                      (min_x, min_y),
                      (max_x, max_y),
                      (0, 255, 0),
                      thickness=2)
        return frame, max_x, min_x, max_y, min_y

    def fps_count(self):
        self.prev_frame_counter, self.frame_counter = self.frame_counter, 0
        # self.frame_counter = 0

    # Определение наличия рук в кадре
    def hand_detection_mp(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # cv2.flip(frame, 1)
        frame.flags.writeable = False
        results = hands.process(frame)
        frame.flags.writeable = True

        if results.multi_hand_landmarks:
            count = 0
            for hand_landmarks in results.multi_hand_landmarks:
                count += 1
                print(f"Рука {count}")
                print(
                    f'\tIndex finger tip coordinates: ('
                    f'x: {round(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * IMAGE_WIDTH)}, '
                    f'y: {round(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * IMAGE_HEIGHT)})'
                )
                for num, mark in enumerate(hand_landmarks.landmark):
                    print(f"\tМетка {arm_marks[num]}"
                          f"- x: {round(mark.x * IMAGE_WIDTH)}, y: {round(mark.y * IMAGE_HEIGHT)}")

        return results

    # Функция в которой будет происходить процесс считывания и обработки каждого кадра
    def run(self):
        # Заходим в бесконечный цикл
        while True:
            if cam_index_list:
                # Считываем каждый новый кадр - frame
                # ret - логическая переменая. Смысл - считали ли мы кадр с потока или нет
                ret, self.frame = cam.read()
                try:
                    # детектируем расположение лица на кадре, вероятности на сколько это лицо
                    boxes, probs = self.mtcnn.detect(self.frame, landmarks=False)   # , landmarks

                    if boxes is not None:
                        # Рисуем на кадре
                        self.frame = self.draw_face(self.frame, boxes, probs)   # , landmarks
                        # Ищем руки
                        hand_detect_rez = self.hand_detection_mp(self.frame)
                        if hand_detect_rez.multi_hand_landmarks:
                            for hand_landmarks in hand_detect_rez.multi_hand_landmarks:
                                self.frame, max_x, min_x, max_y, min_y = self.draw_hand(self.frame, hand_landmarks)

                except Exception as e:
                    print(f'Error {e} in run')

                # пишем в кадре число FPS
                cv2.putText(self.frame,
                            f"FPS: {self.prev_frame_counter}",
                            (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

                self.frame_counter += 1
                rgb_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                convert_to_qt_format = QImage(rgb_image.data,
                                              rgb_image.shape[1],
                                              rgb_image.shape[0],
                                              QImage.Format_RGB888)
                convert_to_qt_format = QPixmap.fromImage(convert_to_qt_format)
                pixmap = QPixmap(convert_to_qt_format)
                self.frame_update_signal.emit(pixmap)   # cv2.imshow(self.label, self.frame)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    app.setApplicationName("Распознавание жестов")
    window = MainWindow()
    window.show()

    face_hand_detect = FaceAndHandDetector()
    face_hand_detect.frame_update_signal.connect(window.show_frame_slot)
    face_hand_detect.start()

    sys.exit(app.exec_())
