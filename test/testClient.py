import cv2
from flask import Flask, Response
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # small model for real-time on Pi


cap = cv2.VideoCapture("rtsp://192.168.1.82:8080/h264.sdp")

app = Flask(__name__)

if not cap.isOpened():
    print("Failed to open IP stream")
    exit()


def gen_frames():
    while True:
        
        ret, frame = cap.read()
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)  # convert to 3-channel RGB
        results = model(frame)[0]  # returns predictions for this frame
        annotated_frame = results.plot()
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run server on all interfaces so other devices can connect
    app.run(host='0.0.0.0', port=5000)