import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, 
    QVBoxLayout, QWidget, QHBoxLayout, QFrame, QProgressBar, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder


class EmotionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Facial Expression Recognition")
        self.setGeometry(100, 100, 900, 600)
        self.setStyleSheet("""
            QWidget { background-color: #f0f0f0; }
            QLabel#title { font-size: 20px; font-weight: bold; color: #0078d7; }
            QFrame#card { background-color: #e6f3ff; border: 1px solid #0078d7; border-radius: 8px; }
            QLabel#cardTitle { font-size: 12px; font-weight: bold; color: #0078d7; }
            QLabel#cardText { font-size: 11px; color: #333333; }
            QPushButton { background-color: #0078d7; color: white; border-radius: 6px; padding: 8px 14px; }
            QPushButton:disabled { background-color: #999999; }
            QPushButton:hover { background-color: #005a9e; }
        """)

        # Load model (adjust path to your PC)
        self.model = load_model(r"D:\college\Ai\AS2\final\1\multi_output_model.h5")

        # Encoders
        self.emotion_encoder = LabelEncoder()
        self.gender_encoder = LabelEncoder()
        self.age_encoder = LabelEncoder()
        self.emotion_encoder.fit(['angry','disgust','fear','happy','neutral','sad','surprise'])
        self.gender_encoder.fit(['male','female'])
        num_age_classes = self.model.outputs[2].shape[-1]   # third output = age
        self.age_encoder.fit([str(i) for i in range(num_age_classes)])


        # Webcam
        self.capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Layout
        main_layout = QVBoxLayout()
        self.title_label = QLabel("Facial Expression Recognition")
        self.title_label.setObjectName("title")
        self.title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.title_label)

        content_layout = QHBoxLayout()

        # Image panel
        self.image_label = QLabel("No image")
        self.image_label.setFixedSize(500, 400)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #dddddd; border: 2px dashed #aaaaaa;")
        content_layout.addWidget(self.image_label)

        # Prediction cards
        pred_layout = QVBoxLayout()
        self.emotion_label = self._make_card("Emotion")
        self.age_label = self._make_card("Age")
        self.gender_label = self._make_card("Gender")
        pred_layout.addWidget(self.emotion_label["frame"])
        pred_layout.addWidget(self.age_label["frame"])
        pred_layout.addWidget(self.gender_label["frame"])
        pred_layout.addStretch()
        content_layout.addLayout(pred_layout)

        main_layout.addLayout(content_layout)

        # Buttons + progress
        btn_layout = QHBoxLayout()
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)
        self.webcam_btn = QPushButton("Start Webcam")
        self.webcam_btn.clicked.connect(self.toggle_webcam)
        self.capture_btn = QPushButton("Capture")
        self.capture_btn.setEnabled(False)
        self.capture_btn.clicked.connect(self.capture_prediction)
        btn_layout.addWidget(self.upload_btn)
        btn_layout.addWidget(self.webcam_btn)
        btn_layout.addWidget(self.capture_btn)
        main_layout.addLayout(btn_layout)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(10)
        main_layout.addWidget(self.progress)

        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 11px; color: #555555;")
        main_layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def _make_card(self, title):
        frame = QFrame()
        frame.setObjectName("card")
        layout = QVBoxLayout()
        lbl_title = QLabel(title)
        lbl_title.setObjectName("cardTitle")
        lbl_text = QLabel("Not predicted")
        lbl_text.setObjectName("cardText")
        lbl_text.setWordWrap(True)
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_text)
        frame.setLayout(layout)
        return {"frame": frame, "text": lbl_text}

    def preprocess_image(self, image):
        image = cv2.resize(image, (48, 48))
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        return image

    def predict(self, image):
        self.progress.setRange(0, 0)
        processed = self.preprocess_image(image)
        preds = self.model.predict(processed)
        emotion_pred, gender_pred, age_pred = preds
        emotion = self.emotion_encoder.inverse_transform([np.argmax(emotion_pred)])[0]
        gender = self.gender_encoder.inverse_transform([np.argmax(gender_pred)])[0]
        age = self.age_encoder.inverse_transform([np.argmax(age_pred)])[0]
        self.progress.setRange(0, 1)
        return emotion, np.max(emotion_pred)*100, gender, np.max(gender_pred)*100, age, np.max(age_pred)*100

    def upload_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            image = cv2.imread(path)
            if image is None:
                QMessageBox.warning(self, "Error", "Could not load image")
                return
            self.display_and_predict(image)

    def toggle_webcam(self):
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                QMessageBox.warning(self, "Error", "Webcam not available")
                self.capture = None
                return
            self.timer.start(30)
            self.webcam_btn.setText("Stop Webcam")
            self.capture_btn.setEnabled(True)
            self.upload_btn.setEnabled(False)
        else:
            self.stop_webcam()

    def stop_webcam(self):
        if self.capture:
            self.capture.release()
            self.capture = None
        self.timer.stop()
        self.webcam_btn.setText("Start Webcam")
        self.capture_btn.setEnabled(False)
        self.upload_btn.setEnabled(True)
        self.image_label.clear()
        self.image_label.setText("No image")

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            self.display_and_predict(frame)

    def display_and_predict(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio
        )
        self.image_label.setPixmap(pixmap)
        
        # Run prediction
        emotion, e_conf, gender, g_conf, age, a_conf = self.predict(frame)
        self.emotion_label["text"].setText(f"{emotion} ({e_conf:.1f}%)")
        self.age_label["text"].setText(f"{age} ({a_conf:.1f}%)")
        self.gender_label["text"].setText(f"{gender} ({g_conf:.1f}%)")


    def capture_prediction(self):
        self.status_label.setText("Prediction captured ")

    def closeEvent(self, event):
        self.stop_webcam()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionUI()
    window.show()
    sys.exit(app.exec_())
