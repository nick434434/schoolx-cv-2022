import numpy as np
import cv2


def get_all_info(video_file, opencv_only=True):
    # Gets all basic info about the video
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    if not ret:
        raise FileNotFoundError(f"Could not read frames from {video_file}: probably path does not exist")
    else:
        resolution = frame.shape[1], frame.shape[0]
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        ex = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_array = ex & 0XFF, (ex & 0XFF00) >> 8, (ex & 0XFF0000) >> 16, (ex & 0XFF000000) >> 24
        codec = "".join([chr(elem) for elem in codec_array])
        bitrate = cap.get(cv2.CAP_PROP_BITRATE)

    bitrate_per_pixel = bitrate / resolution[0] / resolution[1]
    w, h = resolution
    gcd = np.gcd(w, h)
    aspect_ratio = (w // gcd, h // gcd)

    return {
        "name": video_file,
        "resolution": resolution,
        "aspect ratio": aspect_ratio,
        "fps": fps,
        "frame count": frame_count,
        "codec": codec,
        "bitrate per pixel": bitrate_per_pixel,
    }
