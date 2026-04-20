import os

import cv2


def main() -> None:
    rtsp_url = os.getenv("CAMERA_RTSP_URL", "").strip()
    if not rtsp_url:
        raise ValueError("Definí CAMERA_RTSP_URL para probar conectividad RTSP")

    print("Probando conexión RTSP...")
    cap = cv2.VideoCapture(rtsp_url)
    ok, _ = cap.read()
    print("abrió:", cap.isOpened(), "frame:", ok)
    cap.release()


if __name__ == "__main__":
    main()
