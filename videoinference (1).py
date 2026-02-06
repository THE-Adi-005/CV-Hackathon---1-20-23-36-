from ultralytics import YOLO
import cv2
import time
import os

# Load YOLOv10 pretrained model
model = YOLO("yolov10n.pt")

video_path = "videos/video2.mp4"
output_path = "outputs2/video2_output_yolo10.mp4"

os.makedirs("outputs2", exist_ok=True)

cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()
    out.write(annotated_frame)

cap.release()
out.release()

end_time = time.time()

print("✅ YOLOv10 video inference completed")
print(f"⏱️ Total processing time: {end_time - start_time:.2f} seconds")
print(f"📁 Output saved at: {output_path}")
import cv2, torch, torchvision
from torchvision.transforms import functional as F

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

cap = cv2.VideoCapture("videos/drone_test.mp4")
w, h = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    "videos/drone_test_fasterrcnn_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    t = F.to_tensor(frame)
    with torch.no_grad():
        preds = model([t])[0]

    for box in preds["boxes"]:
        x1,y1,x2,y2 = map(int, box.tolist())
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    out.write(frame)

cap.release()
out.release()
print("✅ Faster R-CNN video inference done")
import cv2, torch, torchvision
from torchvision.transforms import functional as F

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

cap = cv2.VideoCapture("videos/drone_test.mp4")
w, h = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    "videos/drone_test_maskrcnn_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    t = F.to_tensor(frame)
    with torch.no_grad():
        preds = model([t])[0]

    for box in preds["boxes"]:
        x1,y1,x2,y2 = map(int, box.tolist())
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

    out.write(frame)

cap.release()
out.release()
print("✅ Mask R-CNN video inference done")
from ultralytics import YOLO
import cv2

model = YOLO("rtdetr-l.pt")  # or rtdetr-s.pt if available

cap = cv2.VideoCapture("videos/drone_test.mp4")
w, h = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    "videos/drone_test_rtdetr_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    res = model(frame)[0]
    out.write(res.plot())

cap.release()
out.release()
print("✅ RT-DETR video inference done")