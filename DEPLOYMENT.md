# 🚀 Deployment & Next Steps

Panduan untuk evaluasi, deployment, dan integrasi model ke production.

---

## 📦 Output Training Files

Setelah training selesai, akan menghasilkan file-file berikut:

```
SolDef_AI/
├── qc_inspector_model.h5        # ⭐ Model final (gunakan ini untuk deployment)
├── best_model.h5                # Model checkpoint terbaik (backup)
├── training_history.json        # History metrics (loss, accuracy per epoch)
└── logs/                        # TensorBoard logs untuk visualisasi
    └── 20251112-143052/         # Folder per training session
```

---

## 1️⃣ Evaluasi Model dengan Test Data

### Cara 1: Testing dengan Single Image

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model('qc_inspector_model.h5')

# Test dengan gambar baru
img_path = 'test_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Prediksi
prediction = model.predict(img_array)
print(f"Prediction: {prediction[0][0]:.4f}")

# Interpretasi (sesuaikan dengan class_indices Anda)
if prediction[0][0] > 0.5:
    print("Hasil: Lulus QC ✅")
else:
    print("Hasil: Cacat Produksi ❌")
```

---

### Cara 2: Batch Testing dengan Test Dataset

```python
# evaluate.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load model
model = tf.keras.models.load_model('qc_inspector_model.h5')

# Prepare test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    'test_dataset',  # Buat folder terpisah untuk test
    target_size=(224, 224),
    batch_size=16,
    class_mode='binary',
    shuffle=False
)

# Evaluate
results = model.evaluate(test_gen)
print(f"Test Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1]:.4f}")

# Predictions
predictions = model.predict(test_gen)
y_pred = (predictions > 0.5).astype(int).flatten()
y_true = test_gen.classes

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=test_gen.class_indices.keys()))
```

---

### Cara 3: Visualisasi Training History

```python
import json
import matplotlib.pyplot as plt

# Load history
with open('training_history.json', 'r') as f:
    history = json.load(f)

# Plot accuracy & loss
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy
ax1.plot(history['accuracy'], label='Training')
ax1.plot(history['val_accuracy'], label='Validation')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Loss
ax2.plot(history['loss'], label='Training')
ax2.plot(history['val_loss'], label='Validation')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_plot.png', dpi=150)
plt.show()
```

---

### Cara 4: TensorBoard untuk Monitoring

```bash
# Jalankan TensorBoard
tensorboard --logdir=logs

# Buka browser: http://localhost:6006
```

---

## 2️⃣ Konversi ke TFLite untuk Mobile (Flutter)

### Basic Conversion

```python
# convert_tflite.py
import tensorflow as tf

# Load Keras model
model = tf.keras.models.load_model('qc_inspector_model.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save
with open('qc_inspector_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Model converted to TFLite!")
print(f"Size: {len(tflite_model) / 1024 / 1024:.2f} MB")
```

---

### Optimized Conversion (Smaller Size)

```python
# convert_tflite_optimized.py
import tensorflow as tf

model = tf.keras.models.load_model('qc_inspector_model.h5')

# Convert dengan optimization
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimization options
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]  # FP16 quantization

# Convert
tflite_model = converter.convert()

# Save
with open('qc_inspector_model_optimized.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Optimized model converted!")
print(f"Size: {len(tflite_model) / 1024 / 1024:.2f} MB")
```

**Comparison:**
- **Standard TFLite:** ~9-12 MB
- **Optimized (FP16):** ~5-6 MB (2x smaller, minimal accuracy loss)

---

## 3️⃣ Integrasi dengan Flutter

### Setup Flutter Project

**1. Tambahkan dependency di `pubspec.yaml`:**
```yaml
dependencies:
  flutter:
    sdk: flutter
  tflite_flutter: ^0.10.0
  image_picker: ^0.8.7+5
  image: ^4.0.17

flutter:
  assets:
    - assets/qc_inspector_model.tflite
```

**2. Copy model ke folder assets:**
```bash
mkdir assets
copy qc_inspector_model.tflite assets/
```

---

### Flutter Implementation

**pcb_inspector.dart:**
```dart
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'dart:io';

class PCBInspector {
  Interpreter? _interpreter;
  
  // Load model saat app start
  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/qc_inspector_model.tflite');
      print('✅ Model loaded successfully');
    } catch (e) {
      print('❌ Error loading model: $e');
    }
  }
  
  // Predict dari image path
  Future<Map<String, dynamic>> predict(String imagePath) async {
    if (_interpreter == null) {
      throw Exception('Model not loaded');
    }
    
    // 1. Load & preprocess image
    final imageData = File(imagePath).readAsBytesSync();
    img.Image? image = img.decodeImage(imageData);
    img.Image resized = img.copyResize(image!, width: 224, height: 224);
    
    // 2. Convert to float32 array (normalized)
    var input = List.generate(
      224,
      (y) => List.generate(
        224,
        (x) {
          var pixel = resized.getPixel(x, y);
          return [
            img.getRed(pixel) / 255.0,
            img.getGreen(pixel) / 255.0,
            img.getBlue(pixel) / 255.0,
          ];
        },
      ),
    );
    
    // 3. Run inference
    var output = List.filled(1, 0.0).reshape([1, 1]);
    _interpreter!.run([input], output);
    
    double prediction = output[0][0];
    
    // 4. Interpret result (sesuaikan dengan class_indices)
    String result;
    double confidence;
    
    if (prediction > 0.5) {
      result = "Lulus QC ✅";
      confidence = prediction;
    } else {
      result = "Cacat Produksi ❌";
      confidence = 1 - prediction;
    }
    
    return {
      'result': result,
      'confidence': confidence,
      'raw_prediction': prediction,
    };
  }
  
  // Dispose interpreter saat app close
  void dispose() {
    _interpreter?.close();
  }
}
```

---

### Flutter UI Example

**main.dart:**
```dart
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'pcb_inspector.dart';
import 'dart:io';

void main() => runApp(PCBInspectorApp());

class PCBInspectorApp extends StatefulWidget {
  @override
  _PCBInspectorAppState createState() => _PCBInspectorAppState();
}

class _PCBInspectorAppState extends State<PCBInspectorApp> {
  final PCBInspector _inspector = PCBInspector();
  final ImagePicker _picker = ImagePicker();
  
  File? _image;
  Map<String, dynamic>? _result;
  bool _isLoading = false;
  
  @override
  void initState() {
    super.initState();
    _inspector.loadModel();
  }
  
  Future<void> _pickImage() async {
    final XFile? pickedFile = await _picker.pickImage(
      source: ImageSource.gallery,
    );
    
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _isLoading = true;
      });
      
      // Predict
      final result = await _inspector.predict(pickedFile.path);
      
      setState(() {
        _result = result;
        _isLoading = false;
      });
    }
  }
  
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: Text('PCB Quality Inspector'),
        ),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (_image != null)
                Image.file(_image!, height: 300),
              SizedBox(height: 20),
              if (_isLoading)
                CircularProgressIndicator(),
              if (_result != null && !_isLoading)
                Column(
                  children: [
                    Text(
                      _result!['result'],
                      style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                    ),
                    Text(
                      'Confidence: ${(_result!['confidence'] * 100).toStringAsFixed(1)}%',
                      style: TextStyle(fontSize: 18),
                    ),
                  ],
                ),
              SizedBox(height: 20),
              ElevatedButton(
                onPressed: _pickImage,
                child: Text('Pick Image'),
              ),
            ],
          ),
        ),
      ),
    );
  }
  
  @override
  void dispose() {
    _inspector.dispose();
    super.dispose();
  }
}
```

---

## 4️⃣ Deploy ke Production (API Server)

### Flask API Server

**app.py:**
```python
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = tf.keras.models.load_model('qc_inspector_model.h5')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'loaded'})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Load image
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array)[0][0]
        
        # Result
        result = {
            'prediction': float(prediction),
            'class': 'lulus_qc' if prediction > 0.5 else 'cacat_produksi',
            'confidence': float(prediction if prediction > 0.5 else 1 - prediction),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Run server:**
```bash
pip install flask
python app.py
```

**Test dengan curl:**
```bash
curl -X POST -F "image=@test_image.jpg" http://localhost:5000/predict
```

---

## 5️⃣ Model Versioning & Metadata

**Save metadata untuk tracking:**
```python
import json
from datetime import datetime

metadata = {
    'model_version': '1.0.0',
    'training_date': datetime.now().isoformat(),
    'dataset_size': 428,
    'train_accuracy': 0.9523,
    'val_accuracy': 0.9176,
    'epochs_trained': 200,
    'class_indices': {'cacat_produksi': 0, 'lulus_qc': 1},
    'hyperparameters': {
        'batch_size': 16,
        'learning_rate': 0.001,
        'dropout': 0.3,
        'dense_units': 128
    },
    'architecture': 'MobileNetV2',
    'input_shape': [224, 224, 3]
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

---

## 6️⃣ Deployment Checklist

### Pre-Deployment ✅
- [ ] Training accuracy ≥ 90%
- [ ] Validation accuracy ≥ 90%
- [ ] Gap train/val < 10%
- [ ] Test dengan gambar baru
- [ ] Model converted to TFLite (jika mobile)
- [ ] Metadata tersimpan
- [ ] Backup model files

### Production Ready 🚀
- [ ] API endpoint tested
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Performance tested
- [ ] Security implemented (API key, HTTPS)
- [ ] Documentation lengkap

---

## 📚 Referensi & Resources

### Official Documentation
- [TensorFlow Keras Guide](https://www.tensorflow.org/guide/keras)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [TFLite Converter](https://www.tensorflow.org/lite/convert)
- [Flutter TFLite Plugin](https://pub.dev/packages/tflite_flutter)

### Tutorials
- [Image Classification with Transfer Learning](https://www.tensorflow.org/tutorials/images/classification)
- [Data Augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation)
- [TensorBoard Visualization](https://www.tensorflow.org/tensorboard/get_started)

### Community
- [TensorFlow Forum](https://discuss.tensorflow.org/)
- [Stack Overflow - TensorFlow](https://stackoverflow.com/questions/tagged/tensorflow)
- [GitHub - TensorFlow Issues](https://github.com/tensorflow/tensorflow/issues)

---

Kembali ke [README.md](README.md) | Lihat [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
