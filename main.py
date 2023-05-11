from ultralytics import YOLO
import time

model = YOLO('yolo_custom.pt')
#
# model.train(data = 'data_custom.yaml', batch=8, imgsz=640, epochs=150, workers=1)

start_time = time.perf_counter()
model.predict(source="1.jpg", show=True, save=True, conf=0.5, line_thickness=1, hide_labels=True, hide_conf=True)
end_time = time.perf_counter()
elapsed_time_ms = (end_time - start_time) * 1000
print(f"Время распознавания: {elapsed_time_ms:.2f} мс")

# yolo task=detect mode=train epochs=100 data=data_custom.yaml model=yolov8m.pt imgsz=640 batch=8
# yolo task=detect mode=predict model=yolov8m_custom.pt show=True conf=0.5 source=1.jpg line_thickness=1 hide_labels=True hide_conf=True
# pip3 install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# In flask_uploads.py
# Change
# from werkzeug import secure_filename,FileStorage
# to
# from werkzeug.utils import secure_filename
# from werkzeug.datastructures import  FileStorage