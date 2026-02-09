# 🌿 AI-Driven Crop Disease Prediction and Management System

Deep Learning • Explainable AI • Smart Agriculture

An end-to-end AI system that detects crop diseases from leaf images, explains predictions using Explainable AI, and provides automated treatment recommendations through a web application and chatbot.

This repository contains the implementation of our **Major Project & Research Paper**.

---

## 📌 Overview

Crop diseases significantly reduce agricultural productivity worldwide. Traditional disease detection is:

- Manual and time-consuming  
- Error-prone  
- Not scalable for large farms  

This project proposes an **AI-powered, interpretable, and deployable system** that bridges the gap between disease prediction and real-world farmer support.

The system integrates:
- Image enhancement
- Deep learning models
- Explainable AI (Grad-CAM)
- Treatment recommendation engine
- Farmer chatbot

---

## 🎯 Key Contributions

✔ Image enhancement using **CLAHE** to improve low-quality images  
✔ Comparison of **CNN, VGG16, MobileNetV2** architectures  
✔ Explainable AI using **Grad-CAM heatmaps**  
✔ Automated **treatment recommendation module**  
✔ **Flask web deployment + chatbot integration**  
✔ Tested on **14 crops & 38 diseases** using PlantVillage dataset  

---

## 🧠 System Workflow

1. Upload crop leaf image  
2. Image enhancement using CLAHE  
3. Disease classification using MobileNetV2  
4. Grad-CAM heatmap visualization  
5. Treatment recommendation generation  
6. Chatbot assistance for farmers  

---

## 📊 Dataset

**Dataset:** PlantVillage

- ~54,000 leaf images  
- 14 crop species  
- 38 disease classes  
- Images resized to **224×224**

### Data Augmentation
- Rotation  
- Zoom  
- Flipping  
- Shifting  

---

## 🤖 Model Comparison

| Model | Accuracy | Loss | Inference Time |
|------|----------|------|----------------|
| CNN | 93.4% | 0.21 | 75 ms |
| VGG16 | 95.7% | 0.15 | 120 ms |
| **MobileNetV2** | **97.8%** | **0.09** | **42 ms** |

🏆 **MobileNetV2 selected for deployment** due to best accuracy and speed.

---

## 🔍 Explainable AI (Grad-CAM)

To overcome the black-box nature of deep learning models, **Grad-CAM** is used to:

- Highlight diseased regions on leaf images
- Provide visual explanations of predictions
- Increase trust and transparency

---

## 🌱 Treatment Recommendation System

The system provides:

- Chemical treatment suggestions  
- Organic remedies  
- Prevention techniques  

This transforms disease prediction into **actionable agricultural guidance**.

---

## 💬 Farmer Chatbot

Integrated chatbot features:

- Answers disease-related questions  
- Provides prevention tips  
- Helps farmers understand symptoms  
- Offers real-time support  

---

## 🛠 Tech Stack

### AI / ML
- TensorFlow
- Keras
- OpenCV
- NumPy

### Backend
- Python
- Flask

### Frontend
- HTML
- CSS
- JavaScript

### Explainable AI
- Grad-CAM

---

## 📂 Project Structure

Major_Project/
│
├── app.py
├── model.py
├── requirements.txt
│
├── templates/
├── static/
│
└── README.md


---

## ▶️ How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/VINAY050805/Major_Project.git
cd Major_Project

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the application
python app.py

4️⃣ Open in browser
http://127.0.0.1:5000/

📦 Dataset & Model Download

Due to GitHub size limits, dataset and trained model are hosted externally.

👉 Dataset + Model: (Paste Google Drive Link Here)

📈 Results

Accuracy: 97.8%

High ROC-AUC (> 0.97)

Strong generalization performance

Suitable for real-time deployment

🔮 Future Enhancements

Expand to 50+ disease classes

Add soil & weather data integration

Mobile application deployment

Edge AI / on-device inference

Multilingual farmer support

👨‍💻 Authors

Vinay S

Dhakshath U K

Prajwal M

Chiranjith R S

Dr. Anil Kumar C J

ATME College of Engineering, Mysuru

⭐ Support

If you like this project, give it a ⭐ on GitHub!


Now push it:

```bash
git add README.md
git commit -m "Added final README"
git push