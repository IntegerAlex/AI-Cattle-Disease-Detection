# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf

# app = FastAPI()

# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # MODEL = tf.keras.models.load_model("../saved_models/5")
# # MODEL = tf.keras.models.load_model("../potatoes.h5")
# MODEL = tf.keras.models.load_model("../diseases.h5", compile=False)


# # CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
# CLASS_NAMES = ["FMD", "IBK", "LSD"]

# @app.get("/ping")
# async def ping():
#     return "Hello, I am alive"

# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image

# @app.post("/predict")
# async def predict(
#     file: UploadFile = File(...)
# ):
#     image = read_file_as_image(await file.read())
#     img_batch = np.expand_dims(image, 0)
    
#     predictions = MODEL.predict(img_batch)

#     predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#     confidence = np.max(predictions[0])
#     return {
#         'class': predicted_class,
#         'confidence': float(confidence)
#     }

# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)










from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# CORS settings
origins = [
        "https://ai-classification-diseases-cattle.vercel.app/",
        
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
MODEL = tf.keras.models.load_model("../diseases.h5", compile=False)

# Class labels
CLASS_NAMES = ["FMD", "IBK", "LSD"]

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

# Preprocessing function (now matches training script)
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")  # Ensure 3 channels
    image = image.resize((256, 256))  # Resize to match training
    img_array = tf.keras.preprocessing.image.img_to_array(image)  # Convert to NumPy array (no /255.0)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    img_batch = preprocess_image(image)  # Use new preprocessing function

    predictions = MODEL.predict(img_batch)
    
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {
        "class": predicted_class,
        "confidence": confidence,
        "all_confidences": {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)



