from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from ultralytics import YOLO
import uvicorn

app = FastAPI()

# Enable CORS
origins = ["http://localhost", "http://localhost:3000","*"]  # Add your frontend URL(s) here

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize YOLO model
MODEL_PATH = 'best.pt'
model = YOLO(MODEL_PATH)

def process_image(file_path):
    results_list = model.predict(file_path, save=True, imgsz=512, conf=0.25)
    results = results_list[0]
    os.remove(file_path)
    output_dir = results.save_dir
    filename = os.path.basename(file_path)
    extracted_folder = os.path.basename(output_dir)
    shutil.move(output_dir+'\\'+filename, 'C:/Users/vikas/mig/fastapi/output/')
    output_image = 'C:/Users/vikas/mig/fastapi/output/' + filename

    return output_image

@app.get("/")
async def root():

    return {"message": "TB Detection"} 

@app.post("/tbdetection/")
async def tb_detection(file: UploadFile = File(...)):

    try:
        # Save the uploaded file
        file_path = f'C:/Users/vikas/mig/fastapi/{file.filename}'
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Process the image using the YOLO model
        output_image_path = process_image(file_path)
        # Return the processed image to the frontend
        return StreamingResponse(open(output_image_path, "rb"), media_type="image/jpeg")

    except Exception as e:
        # Log the error
        print(f"Error processing image: {str(e)}")
        
        # Raise HTTP exception with a meaningful error message
        raise HTTPException(status_code=500, detail="Error processing image")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
