from ultralytics import YOLO
import cv2
# Load a YOLOv8n PyTorch model
#model = YOLO("yolov8n.pt")

# Export the model
#model.export(format="engine",int8 = True)  # creates 'yolov8n.engine'

# Load the exported TensorRT model
trt_model = YOLO("yolov8n.engine", task='detect')

# Run inference
results = trt_model("https://ultralytics.com/images/bus.jpg")
img = cv2.imread("bus.jpg")
annotated_img = img.copy()
print(results)


for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int,box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = trt_model.names[cls]
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0,255,0), 2)
        label = f"{class_name} {conf:.2f}"
        cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.imwrite("annot_img.jpg",annotated_img)
