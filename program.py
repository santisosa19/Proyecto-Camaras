import cv2

usuario = "admin"
clave = "abcd1234"
ip = "181.90.13.25"

urls = [
    "rtsp://admin:abcd1234@181.90.13.25:8554/Streaming/Channels/101"
]

for url in urls:
    print("Probando:", url.replace(clave, "****"))
    cap = cv2.VideoCapture(url)
    ok, frame = cap.read()
    print("abrió:", cap.isOpened(), "frame:", ok)
    cap.release()