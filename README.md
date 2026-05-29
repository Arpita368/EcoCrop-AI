# 🌱 EcoCrop AI — Smart Crop Disease Detection & Market Insights Using AI

EcoCrop AI is an AI-powered web application designed to help farmers detect crop diseases from leaf images and access real-time agricultural market prices. The system combines **Deep Learning**, **Computer Vision**, and **Government Market Data Integration** to support smarter and more sustainable farming decisions.

The platform allows users to upload a crop leaf image, detects diseases using a trained CNN model, and provides treatment suggestions along with market price insights through the official Agmarknet portal.

---

## 🚀 Features

- 🔍 AI-based crop disease detection using CNN
- 📸 Upload leaf images for instant prediction
- 🧠 Deep Learning model trained on PlantVillage dataset
- 💊 Disease description and treatment recommendations
- 📈 Real-time crop market price access
- 🌐 Agmarknet integration for government-verified prices
- 📱 Responsive and user-friendly web interface
- ⚡ Fast predictions using TensorFlow/Keras
- 🧩 Modular and scalable architecture

---

# 🖼️ System Workflow

1. User uploads a crop leaf image
2. Flask backend preprocesses the image
3. CNN model predicts disease class
4. System displays:
   - Disease Name
   - Confidence Score
   - Symptoms
   - Treatment Suggestions
5. User can check live market prices through Agmarknet integration

---

# 🏗️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| AI Framework | TensorFlow, Keras |
| Dataset | PlantVillage Dataset |
| Model Type | Convolutional Neural Network (CNN) |
| Deployment | Flask Local Server |

---

# 🧠 AI Model Details

The disease detection system uses a **Convolutional Neural Network (CNN)** trained on the **PlantVillage dataset**.

## ✅ Model Capabilities

- Detects healthy and diseased crop leaves
- Supports crops like:
  - Tomato
  - Potato
  - Pepper/Bell Pepper
  - etc

## 🦠 Diseases Supported

- Early Blight
- Late Blight
- Bacterial Spot
- Leaf Mold
- Mosaic Virus
- Healthy Leaf Detection

## 📊 Model Performance

- Training Accuracy: ~96%
- Validation Accuracy: ~90%

---

# 📂 Project Structure

```bash
EcoCrop-AI/
│
├── static/
│   ├── uploads/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│   └── index.html
│
├── models/
│   └── plant_disease_model.h5
│
├── app.py
├── requirements.txt
├── README.md
└── dataset/
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/EcoCrop-AI.git
cd EcoCrop-AI
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/Mac

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run the Flask Application

```bash
python app.py
```

---

## 5️⃣ Open in Browser

```bash
http://127.0.0.1:5000
```

---

# 📦 Requirements

```txt
Flask
TensorFlow
Keras
NumPy
OpenCV
Pillow
Matplotlib
scikit-learn
```

---

# 🌾 Market Price Integration

EcoCrop AI integrates with the official **Agmarknet** portal to provide:

- Real-time crop prices
- Historical market data
- State-wise market insights
- Government-authenticated information

Users can select:
- Crop Name
- State
- Date Range

The system redirects to Agmarknet with pre-filled details.

---

# 📸 Frontend Features

- Leaf image preview
- Responsive UI
- Result popup/modal
- Smooth navigation
- Market price lookup form

---

# 🔬 Methodology

## 📥 Data Preprocessing

- Image resizing (64×64)
- Normalization
- Augmentation
  - Rotation
  - Flipping
  - Zooming

## 🧠 CNN Architecture

- Convolution Layers
- MaxPooling Layers
- Flatten Layer
- Dense Layers
- Softmax Output Layer

## 🏋️ Training

- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Epochs: 25–30

---

# 🎯 Objectives

- Detect crop diseases accurately using AI
- Help farmers make informed decisions
- Reduce agricultural losses
- Increase farming productivity
- Provide easy access to market prices
- Encourage smart farming practices

---

# 🌍 Real-World Impact

EcoCrop AI helps farmers by:

✅ Detecting diseases early  
✅ Reducing crop losses  
✅ Improving yield quality  
✅ Providing market awareness  
✅ Supporting sustainable agriculture  

---

# 🔮 Future Scope

- 📱 Android/iOS Mobile App
- 🌐 Multilingual Support
- 🎤 Voice-Based Assistance
- ☁️ Cloud Deployment
- 🌦️ Weather & Soil Integration
- 🤖 AI Farming Chatbot
- 📡 IoT Sensor Integration
- 📴 Offline Prediction Support

---

# 📚 Dataset

The model is trained using the **PlantVillage Dataset**.

---

# 👨‍💻 Authors

- Arpita Jitendra Sonparote
- Pranali Suresh Somwanshi

Department of Computer Science & Engineering  
Shri Guru Gobind Singhji Institute of Engineering & Technology, Nanded

---

# 📖 References

1. PlantVillage Dataset
2. TensorFlow Documentation
3. Keras Documentation
4. Agmarknet Official Website
5. Research Papers on Deep Learning in Agriculture

---

# 🤝 Contributing

Contributions are welcome!

Feel free to:
- Fork the repository
- Create a new branch
- Submit a pull request

---

# 📜 License

This project is developed for educational and research purposes.

---
