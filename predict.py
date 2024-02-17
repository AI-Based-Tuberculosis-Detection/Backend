from ultralytics import YOLO
import os
import shutil

MODEL_PATH = 'best.pt'
# Load the YOLOv8 model
model = YOLO(MODEL_PATH)
# Make prediction
image_path = 'C:/Users/vikas/mig/fastapi/example.png'
results_list = model.predict(image_path, save=True, imgsz=512, conf=0.25)
results = results_list[0]

output_dir = results.save_dir
filename = os.path.basename(image_path)
shutil.move(output_dir, 'C:/Users/vikas/mig/fastapi/')
output_image = 'C:/Users/vikas/mig/fastapi/predict/'+filename
# shutil.rmtree('C:/Users/vikas/mig/fastapi/predict')
print(output_image)