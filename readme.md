# 🐾 Animal Image Classifier

## 📌 Objective
Build a deep learning-based system that can classify images of animals into 15 categories using Convolutional Neural Networks (CNNs) and Transfer Learning.

> This project fulfills the internship requirement:
> **“Build a system that can identify the animal in a given image. Explore the data set and identify an appropriate solution.”**

---

## 🐶 Classes in Dataset
- Bear
- Bird
- Cat
- Cow
- Deer
- Dog
- Dolphin
- Elephant
- Giraffe
- Horse
- Kangaroo
- Lion
- Panda
- Tiger
- Zebra

---

## 🧠 Approach

### 🔍 1. Dataset
- Folder-based dataset with 15 subdirectories (one per class).
- Images are of shape **224x224x3**.

### ⚙️ 2. Model
- Used **MobileNetV2** from Keras Applications as the base model.
- Transfer Learning:
  - Base model was frozen.
  - Custom classifier head was added (Dense → ReLU → Softmax).
- Trained for 5 epochs with `ImageDataGenerator` and 20% validation split.

### 📈 3. Visualization
- Training and validation accuracy/loss are plotted using Matplotlib.

### 🧪 4. Prediction Script
- `predict.py` loads a new image and returns the predicted animal class.

---

## 🗂️ Project Structure
animal_classifier/
│
├── animals_dataset/ ← 15 animal folders
├── test_images/ ← images for testing prediction
├── main.py ← model training code
├── predict.py ← single-image prediction
├── animal_model.h5 ← saved model
└── README.md

---

## 🚀 How to Run

### 🔧 Install Dependencies
```bash
pip install tensorflow numpy matplotlib pillow
🏋️‍♂️ Train the Model
python main.py
🔍 Predict on a New Image
python predict.py
Output:
Predicted Animal: Tiger
✅ Outcome
Built a robust image classification model with ~15 categories using deep learning.

Achieved efficient training via Transfer Learning.

Built a functional system that accepts an image and returns the correct animal class.

📌 Future Scope
Add a Streamlit GUI for easier use

Deploy the model on the web (using Hugging Face or Vercel)

Fine-tune more layers of the base model for higher accuracy