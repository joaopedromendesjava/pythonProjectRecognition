from ultralytics import YOLO
from roboflow import Roboflow
import os

model = YOLO('yolov8n.pt')


#rf = Roboflow(api_key="xYNvXQODHkMKwvKxiVTC")
#project = rf.workspace("joo-pedro-2tyj4").project("yolov8_images")
#dataset = project.version(1).download("yolov8")


try:
    os.system('yolo task=detect mode=train model=yolov8n.pt data="C:/Users/joao pedro/PycharmProjects/pythonProject/YoloV8_Images-1/data.yaml" epochs=20 imgsz=640')
    print("Finalizado")
except Exception as ex:
    print("ERRO: ", ex)
    raise

