import cv2


def extract_frames_from_video(cap, interval):
    """
    从视频中提取指定间隔的帧
    :param cap: 视频对象
    :param interval: 帧间隔
    :return: 帧列表
    """
    frame_count = 0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % (int(cap.get(cv2.CAP_PROP_FPS) * interval)) == 0:
            frames.append(frame)

    cap.release()
    return frames


if __name__ == "__main__":
    pass
