





from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import json
from ultralytics import YOLO

app = FastAPI(title="YOLO11N Inference")

# YOLO 모델은 자체적으로 전처리를 수행합니다. 그래서
# transforms를 할 필요니다.
model = YOLO("../models/yolo11n.pt")


# @app.post("/yolo")
# async def predict(file: UploadFile=File(...)):
    
#     image = Image.open(file.file)
    
#     results = model(image)
    
#     detections = []
    
#     for result in results:
#         if not result.boxes:
#             continue

#         for box in result.boxes:
#             x1, y1, x2, y2 = [
#                 round(x) for x in box.xyxy[0].tolist
#             ]

#             confidence = round(box.conf[0].item(), 2)
#             class_id = int(box.cls[0].item())
#             class_name = result.names[class_id]

#             detections.append({
#                 "class_id": class_id,
#                 "class_name": class_name,
#                 "confidence": confidence,
#                 "bbox": [x1, y1, x2, y2]
#             })

        
#     return {"detection" : "detection"}
   


   