import cv2
from FileProcessor.ImageProcess import face_extract, image2tensor
from FileProcessor.VideoProcess import extract_frames_from_video
from ModelLoader import *
from networks.mesonet import Meso4


def detect_deepfakes(image):
    """
    检测图像是否为伪造
    """
    tensor = image2tensor(image)
    model = MesoLoder(model=Meso4, model_path="models/model.pth")
    prediction = model.model.forward(tensor)
    prediction = prediction.item()
    return prediction


def annotate_image(image, definition=100):
    """
    检测图像并标注
    :param image: 图像
    :param definition: 人脸识别清晰度阈值
    """
    faces = face_extract(image, definition=definition)
    for face, (x, y, w, h) in faces:
        probability = detect_deepfakes(face)
        label = "Fake" if probability > 0.5 else "Real"
        color = (0, 0, 255) if label == "Fake" else (0, 255, 0)

        # 在原图上标记人脸
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"{label} {probability:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def annotate_video(cap, interval, definition=100):
    frames = extract_frames_from_video(cap, interval)
    for frame in frames:
        yield annotate_image(frame, definition=definition)
