# **Advanced Techniques in Deep Learning for Pancreatic Cancer Detection and Classification**

## **Team Details**
- **Gandikota Narendra**  
- **Katta Subbarao**  
- **Nallabothu Narendra**  

---

## **Introduction**
Pancreatic cancer is one of the deadliest cancers, often detected in later stages due to its subtle onset. Early detection can significantly improve survival rates. This project leverages **deep learning** techniques for accurate detection and classification of pancreatic cancer using **medical imaging**.  

We employ the **InceptionDense model**, which combines the strengths of **InceptionV3** and **DenseNet121** for **multi-scale feature extraction** and **efficient feature reuse**. This enhances tumor detection across varying shapes and sizes, ensuring **high diagnostic reliability**.

---

## **Dataset**
- **1,411 annotated CT images** sourced from **Kaggle**.  
- Includes **cancerous** and **non-cancerous** images, ensuring a **balanced dataset** for training and evaluation.  
- **Dataset Links:**  
  - [Kaggle Dataset](https://www.kaggle.com/datasets/jayaprakashpondy/pancreatic-ct-images)  
  - [Google Drive](https://drive.google.com/drive/folders/1hpjCAuNCcZUyYZsFjdti9MDulgSF0yeh)

---

## **Model Architecture**
The **InceptionDense model** integrates:  
✅ **InceptionV3**: Captures multi-scale features through parallel convolutional filters.  
✅ **DenseNet121**: Enhances learning through feature reuse and dense connectivity.  

### **Other Hybrid Models Evaluated:**
- **EfficientDense** *(EfficientNetB0 + DenseNet121)*
- **EfficientV3** *(EfficientNetB0 + InceptionV3)*
- **EfficientVGG** *(EfficientNetB0 + VGG16)*
- **VGG16V2** *(VGG16 + MobileNetV2)*
- **ResNetV2** *(ResNet50 + MobileNetV2)*

---

## **Preprocessing Techniques**
To optimize input for deep learning models, the following preprocessing steps were applied:
✅ **Image Resizing:** Standardized to **224x224 pixels**.  
✅ **Normalization:** Pixel values scaled to **[0,1]**.  
✅ **Label Encoding:** Binary labels assigned **(0: non-cancerous, 1: cancerous)**.  

---

## **Training and Evaluation**
🔹 **80% of the dataset** used for training, **20% for testing**.  
🔹 **Optimizer:** Adam  
🔹 **Loss Function:** Binary Cross-Entropy  
🔹 **Training Duration:** 10 epochs  
🔹 **Learning Rate:** 0.0001  

---

## **Performance Metrics**
| Model | Accuracy | Precision | Recall | F1-Score |
|--------|------------|------------|---------|------------|
| **InceptionDense** | **99.75%** | **99.47%** | **99.75%** | **99.73%** |
| **EfficientDense** | **100%** | **100%** | **100%** | **100%** |
| **EfficientV3** | **100%** | **100%** | **100%** | **100%** |
| **EfficientVGG** | **97.75%** | **99.47%** | **100%** | **99.73%** |
| **VGG16V2** | **72.98%** | **100%** | **42.78%** | **59.63%** |
| **ResNetV2** | **65.40%** | **57.72%** | **100%** | **73.19%** |

---

## **Key Findings**
✅ **InceptionDense outperformed all models**, achieving **99.75% accuracy**.  
✅ **EfficientDense and EfficientV3 reached 100% accuracy**, but may be overfitting.  
✅ **VGG16V2 and ResNetV2 had lower performance**, requiring further optimization.  

---

## **Results and Discussion**
📌 **Confusion Matrix Analysis** – Highlights misclassification trends and areas for improvement.  
📌 **Feature Fusion** – Enhances performance by combining different feature extraction methods.  
📌 **Challenges** – Handling **low-light images** and high **computational costs** for real-world deployment.  

---

## **Conclusion**
This project demonstrates that **deep learning models, particularly InceptionDense, can significantly improve pancreatic cancer detection and classification**. The study shows promising results for **early diagnosis**, which is crucial for **improving patient survival rates**.  

🔹 **Future Work:**  
- **Validation on larger datasets** for improved generalization.  
- **Extension to other imaging modalities** like **MRI and ultrasound** for broader clinical applicability.  

---

## **Deployment**
🚀 The model can be deployed using **Flask, FastAPI, or TensorFlow Serving**.  
💡 **Future plans:** Cloud-based deployment for **real-time analysis**.  
📌 **Deployment Link:** **  

---

## **Technologies Used**
💻 **Programming & Frameworks:** Python, TensorFlow, Keras  
📷 **Image Processing:** OpenCV  
📊 **Visualization:** Matplotlib, Seaborn  
📊 **Data Handling:** NumPy, Pandas  

---
