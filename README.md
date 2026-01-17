# Neonatal Cry Detection System

This project is a part of the **Smart Neonatal Incubator Monitoring System**.  
It uses **Deep Learning (CNN)** to automatically detect **infant crying** from audio signals and generates a **medical alert** through a web-based interface to assist doctors and caregivers in real time.

---

## 🩺 Project Motivation

In neonatal intensive care units (NICU), continuous monitoring of infants is critical.  
Crying is an important indicator of discomfort, hunger, pain, or distress in neonates.  
Manual monitoring is difficult and error-prone.

This system aims to:
- Automatically detect infant crying
- Reduce response time for medical staff
- Assist in smart and non-contact neonatal monitoring

---

## 🚀 Features

- 🎤 Real-time audio capture using microphone
- 🧠 Deep learning based cry vs non-cry classification
- 📊 Audio feature extraction using Mel Spectrograms
- 🌐 Flask-based medical alert web interface
- 🚨 Visual alert for doctors when crying is detected
- 💻 Runs on CPU (GPU/CUDA supported optionally)

---

## 🧠 Methodology

1. **Audio Input**  
   Infant audio is captured using a microphone or loaded from audio files.

2. **Preprocessing**  
   - Audio is resampled
   - Mel Spectrogram is generated
   - Converted to decibel scale

3. **Deep Learning Model**  
   - Convolutional Neural Network (CNN)
   - Binary classification: Cry / Non-Cry

4. **Decision Logic**  
   - Model outputs a probability score
   - Threshold-based classification
   - Alert generated if cry is detected

5. **Web Interface**  
   - Flask backend
   - HTML/CSS frontend
   - Displays medical alert status in real time

---

## 🗂 Dataset Used

The model is trained using publicly available Kaggle datasets:

### 🔹 Dataset 1: Infant Cry Dataset
- Source: Kaggle
- Link:  
  https://www.kaggle.com/datasets/sanmithasadhish/infant-cry-dataset
- Contains labeled cry and non-cry infant audio samples

### 🔹 Dataset 2: Baby Crying Sounds Dataset
- Source: Kaggle
- Link:  
  https://www.kaggle.com/datasets/mennaahmed23/baby-crying-sounds-dataset
- Includes diverse baby crying audio samples

> ⚠️ Note: Datasets are not included in this repository due to size constraints.

---

## 🛠 Technologies Used

- **Programming Language:** Python 3.10
- **Deep Learning:** TensorFlow / Keras
- **Audio Processing:** Librosa
- **Web Framework:** Flask
- **Frontend:** HTML, CSS
- **Others:** NumPy, SoundDevice

---

## 📁 Project Structure

