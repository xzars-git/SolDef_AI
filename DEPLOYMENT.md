# 🚀 Deployment Guide

Panduan lengkap untuk deploy model Casting Product Defect Detection (submersible pump impeller inspection) ke production.

---

## 📦 Model Export Options

### Option 1: Keras .h5 Format (Default)
```python
# Sudah otomatis saat training
model.save('qc_inspector_model.h5')
```

**Pros:**
- ✅ Easy to load
- ✅ Full model architecture
- ✅ Compatible dengan TensorFlow Python

**Cons:**
- ❌ Large file size (~9-10 MB)
- ❌ Python-only deployment

**Use Case:** Python backends, APIs, desktop apps

---

### Option 2: TensorFlow Lite (Mobile/Edge)
```python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('qc_inspector_model.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optimize size
tflite_model = converter.convert()

# Save
with open('qc_inspector_model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"TFLite model size: {len(tflite_model) / 1024:.2f} KB")
```

**Pros:**
- ✅ Small size (~2-3 MB)
- ✅ Fast inference
- ✅ Mobile-friendly (Android, iOS, Flutter)
- ✅ Edge device support

**Cons:**
- ❌ Limited ops support (rare)
- ❌ Requires TFLite runtime

**Use Case:** Mobile apps (Flutter, React Native), IoT devices, edge devices di factory floor, handheld inspection devices

---

### Option 3: SavedModel Format
```python
# Save as SavedModel
model.save('saved_model/')  # Creates folder with assets

# Load
loaded_model = tf.saved_model.load('saved_model/')
```

**Pros:**
- ✅ TensorFlow standard format
- ✅ Multi-language support (Python, C++, Java)
- ✅ TensorFlow Serving compatible

**Cons:**
- ❌ Folder structure (not single file)
- ❌ Larger than .h5

**Use Case:** Production servers, TensorFlow Serving, cloud deployments

---

### Option 4: ONNX Format (Cross-platform)
```python
import tf2onnx
import onnx

# Convert Keras to ONNX
model = tf.keras.models.load_model('qc_inspector_model.h5')
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
onnx.save(model_proto, "qc_inspector_model.onnx")
```

**Pros:**
- ✅ Cross-platform (PyTorch, TensorFlow, etc.)
- ✅ Optimized inference
- ✅ Wide deployment support

**Cons:**
- ❌ Requires additional libraries
- ❌ Conversion complexity

**Use Case:** Multi-framework environments, ONNX Runtime

---

## 🌐 Deployment Scenarios

### 1️⃣ Flask REST API (Python Backend)

**File Structure:**
```
api/
├── app.py
├── qc_inspector_model.h5
├── requirements.txt
└── test_image.jpg
```

**app.py:**
```python
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load model once at startup
model = tf.keras.models.load_model('qc_inspector_model.h5')
print("✅ Model loaded successfully!")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image from request
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Load & preprocess image
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array)[0][0]
        
        # Interpret result
        is_defect = prediction < 0.5
        result = "DEFECTIVE" if is_defect else "OK"
        confidence = (1 - prediction) * 100 if is_defect else prediction * 100
        
        return jsonify({
            'result': result,
            'confidence': float(confidence),
            'prediction_score': float(prediction),
            'product': 'submersible_pump_impeller',
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

**requirements.txt:**
```txt
flask==2.3.0
tensorflow==2.10.0
pillow==9.5.0
numpy==1.24.0
```

**Run API:**
```bash
pip install -r requirements.txt
python app.py
```

**Test API:**
```bash
# PowerShell
curl -X POST -F "file=@test_image.jpg" http://localhost:5000/predict

# Python test
import requests
response = requests.post(
    'http://localhost:5000/predict',
    files={'file': open('test_image.jpg', 'rb')}
)
print(response.json())
```

---

### 2️⃣ FastAPI (Modern Python Backend)

**app.py:**
```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = FastAPI(title="PCB Defect Detection API")

# Load model
model = tf.keras.models.load_model('qc_inspector_model.h5')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read & preprocess image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        img = img.resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array)[0][0]
        
        # Result
        is_defect = prediction < 0.5
        result = "CACAT" if is_defect else "LULUS"
        confidence = (1 - prediction) * 100 if is_defect else prediction * 100
        
        return {
            "result": result,
            "confidence": float(confidence),
            "prediction_score": float(prediction),
            "status": "success"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "status": "failed"}
        )

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}
```

**Run:**
```bash
pip install fastapi uvicorn
uvicorn app:app --host 0.0.0.0 --port 8000
```

**Access API Docs:** http://localhost:8000/docs

---

### 3️⃣ Flutter Mobile App (TFLite)

**Step 1: Convert to TFLite**
```python
import tensorflow as tf

model = tf.keras.models.load_model('qc_inspector_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('qc_inspector_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

**Step 2: Flutter Integration**

**pubspec.yaml:**
```yaml
dependencies:
  tflite_flutter: ^0.10.0
  image_picker: ^0.8.7
  
assets:
  - assets/qc_inspector_model.tflite
```

**main.dart:**
```dart
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'dart:io';

class PCBDetector {
  Interpreter? _interpreter;
  
  // Load model
  Future<void> loadModel() async {
    _interpreter = await Interpreter.fromAsset('assets/qc_inspector_model.tflite');
    print('✅ Model loaded successfully!');
  }
  
  // Predict
  Future<Map<String, dynamic>> predict(File imageFile) async {
    // Load & resize image
    img.Image? image = img.decodeImage(imageFile.readAsBytesSync());
    img.Image resized = img.copyResize(image!, width: 224, height: 224);
    
    // Convert to input tensor
    var input = List.generate(1, (i) =>
      List.generate(224, (y) =>
        List.generate(224, (x) {
          var pixel = resized.getPixel(x, y);
          return [
            img.getRed(pixel) / 255.0,
            img.getGreen(pixel) / 255.0,
            img.getBlue(pixel) / 255.0,
          ];
        })
      )
    );
    
    // Output tensor
    var output = List.filled(1, 0.0).reshape([1, 1]);
    
    // Run inference
    _interpreter!.run(input, output);
    
    // Interpret result
    double prediction = output[0][0];
    bool isDefect = prediction < 0.5;
    String result = isDefect ? "CACAT" : "LULUS";
    double confidence = isDefect ? (1 - prediction) * 100 : prediction * 100;
    
    return {
      'result': result,
      'confidence': confidence,
      'prediction_score': prediction,
    };
  }
}

// Usage
void main() async {
  var detector = PCBDetector();
  await detector.loadModel();
  
  File imageFile = File('path/to/image.jpg');
  var result = await detector.predict(imageFile);
  
  print('Result: ${result['result']}');
  print('Confidence: ${result['confidence'].toStringAsFixed(2)}%');
}
```

---

### 4️⃣ Docker Container (Cloud Deployment)

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model & app
COPY qc_inspector_model.h5 .
COPY app.py .

EXPOSE 5000

# Run API
CMD ["python", "app.py"]
```

**docker-compose.yml:**
```yaml
version: '3.8'
services:
  pcb-detector:
    build: .
    ports:
      - "5000:5000"
    environment:
      - TF_CPP_MIN_LOG_LEVEL=2
    restart: unless-stopped
```

**Build & Run:**
```bash
docker build -t pcb-detector:latest .
docker run -p 5000:5000 pcb-detector:latest

# Or with docker-compose
docker-compose up -d
```

---

### 5️⃣ TensorFlow Serving (Production-grade)

**Step 1: Export SavedModel**
```python
import tensorflow as tf

model = tf.keras.models.load_model('qc_inspector_model.h5')
model.save('saved_model/1/')  # Version 1
```

**Step 2: Run TensorFlow Serving**
```bash
docker run -p 8501:8501 \
  --mount type=bind,source=/path/to/saved_model,target=/models/pcb_detector \
  -e MODEL_NAME=pcb_detector \
  tensorflow/serving
```

**Step 3: Make Predictions**
```python
import requests
import json
import numpy as np
from tensorflow.keras.preprocessing import image

# Load & preprocess image
img = image.load_img('test.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0

# Prepare request
data = json.dumps({
    "signature_name": "serving_default",
    "instances": img_array.tolist()
})

# Send request
headers = {"content-type": "application/json"}
response = requests.post(
    'http://localhost:8501/v1/models/pcb_detector:predict',
    data=data,
    headers=headers
)

prediction = response.json()['predictions'][0][0]
result = "CACAT" if prediction < 0.5 else "LULUS"
print(f"Result: {result} ({prediction:.4f})")
```

---

## ☁️ Cloud Deployment Options

### AWS Lambda + API Gateway
```python
# lambda_function.py
import json
import base64
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

model = tf.keras.models.load_model('qc_inspector_model.h5')

def lambda_handler(event, context):
    # Decode base64 image
    image_data = base64.b64decode(event['body'])
    img = Image.open(io.BytesIO(image_data))
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array)[0][0]
    result = "CACAT" if prediction < 0.5 else "LULUS"
    confidence = (1 - prediction) * 100 if prediction < 0.5 else prediction * 100
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'result': result,
            'confidence': float(confidence)
        })
    }
```

### Google Cloud Run
```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/pcb-detector', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/pcb-detector']
  - name: 'gcr.io/cloud-builders/gcloud'
    args: ['run', 'deploy', 'pcb-detector', 
           '--image', 'gcr.io/$PROJECT_ID/pcb-detector',
           '--platform', 'managed',
           '--region', 'us-central1',
           '--allow-unauthenticated']
```

### Azure ML Endpoint
```python
# score.py for Azure ML
import json
import numpy as np
import tensorflow as tf
from io import BytesIO
from PIL import Image

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'qc_inspector_model.h5')
    model = tf.keras.models.load_model(model_path)

def run(raw_data):
    data = json.loads(raw_data)['data']
    image_data = base64.b64decode(data)
    img = Image.open(BytesIO(image_data))
    # ... preprocessing ...
    prediction = model.predict(img_array)[0][0]
    result = "CACAT" if prediction < 0.5 else "LULUS"
    return json.dumps({'result': result})
```

---

## 📊 Performance Optimization

### GPU Inference
```python
import tensorflow as tf

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# Load model
model = tf.keras.models.load_model('qc_inspector_model.h5')
```

### Batch Inference
```python
# Process multiple images at once
def batch_predict(image_paths, batch_size=32):
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        imgs = [preprocess_image(path) for path in batch]
        imgs = np.array(imgs)
        predictions = model.predict(imgs)
        results.extend(predictions)
    return results
```

### Model Quantization (TFLite)
```python
# Post-training quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # Float16 quantization
tflite_model = converter.convert()

# Size reduction: ~50% smaller, ~2x faster inference
```

---

## 🔒 Security Considerations

1. **API Authentication:**
```python
from flask import request

API_KEY = "your-secret-api-key"

@app.before_request
def verify_api_key():
    if request.endpoint != 'health':
        key = request.headers.get('X-API-Key')
        if key != API_KEY:
            return jsonify({'error': 'Unauthorized'}), 401
```

2. **Rate Limiting:**
```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    # ...
```

3. **Input Validation:**
```python
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
```

---

## 📈 Monitoring & Logging

```python
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = datetime.now()
    
    # ... prediction logic ...
    
    inference_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"Prediction: {result}, Confidence: {confidence:.2f}%, Time: {inference_time:.3f}s")
    
    return jsonify({
        'result': result,
        'confidence': confidence,
        'inference_time_ms': inference_time * 1000
    })
```

---

## 🎯 Next Steps

1. ✅ Choose deployment method (API, mobile, cloud)
2. ✅ Convert model to target format (.h5, .tflite, SavedModel)
3. ✅ Set up backend/infrastructure
4. ✅ Test thoroughly with various images
5. ✅ Monitor performance & accuracy
6. ✅ Implement logging & monitoring
7. ✅ Add authentication & rate limiting
8. ✅ Deploy to production!

---

**Ready for production deployment!** 🚀

Kembali ke [README.md](README.md) untuk dokumentasi lengkap.
