# realtime_onnx_opencv.py
import cv2
import numpy as np
import time

ONNX_PATH = "best.onnx"
class_names = ["scissor","grasper","suction", ...]  # fill with classes from data.yaml

# Load model
net = cv2.dnn.readNetFromONNX(ONNX_PATH)
# Optionally: set preferable backend (DNN_BACKEND_OPENCV) or CUDA if built with CUDA
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    H, W = frame.shape[:2]
    # Preprocess: ultralytics normalizes to 0..1 and resizes with letterbox usually
    inp = cv2.resize(frame, (640,640))
    blob = cv2.dnn.blobFromImage(inp, 1/255.0, (640,640), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()  # output shape depends on model export (xywh + conf + class scores)
    # WARNING: post-processing depends on exact ONNX output format. Ultralytics ONNX returns
    # a tensor of shape (N, 85) for COCO-like models (if export was configured). You must parse boxes, apply NMS.
    # Below is a sample parsing assuming [x, y, w, h, conf, cls1, cls2, ...] per row:
    boxes = []
    confidences = []
    class_ids = []
    for det in preds[0]:
        scores = det[5:]
        class_id = np.argmax(scores)
        conf = float(scores[class_id] * det[4])
        if conf > 0.25:
            cx,cy,w,h = det[0:4]
            # map to original image size
            x1 = int((cx - w/2) * W/640)
            y1 = int((cy - h/2) * H/640)
            x2 = int((cx + w/2) * W/640)
            y2 = int((cy + h/2) * H/640)
            boxes.append([x1,y1,x2-x1,y2-y1])
            confidences.append(conf)
            class_ids.append(class_id)
    # NMS
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
    if len(idxs) > 0:
        for i in idxs.flatten():
            x,y,w,h = boxes[i]
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            label = f"{class_names[class_ids[i]]}:{confidences[i]:.2f}"
            cv2.putText(frame, label, (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("ONNX Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
