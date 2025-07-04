# Prodigy Info Tech - Machine Learning Internship Task 03  
# Implement a SVM to classify Images of Cats and Dogs  
This project is a simple image classifier built with Tkinter GUI that predicts whether an uploaded image is a cat or a dog  

## 🛠️ Tech Stack  
- Programming Language: Python  
- GUI Library: Tkinter  
- Image Processing: OpenCV  
- Machine Learning: Scikit-learn (SVM, PCA, StandardScaler)  
- Feature Extraction: HOG (from scikit-image)  
- Image Handling: Pillow (PIL)  
## 📦 Dataset  
### 👉 Dogs vs. Cats - Kaggle Dataset  
The project uses the Dogs vs Cats dataset from Kaggle:  

25,000 labeled images (12,500 cats, 12,500 dogs)  
Used only 2000 images per class for training due to local compute limitations  
Images were resized to 128×128 grayscale for feature consistency  
## 📈 Model Performance  
- Metric	Score  
- Accuracy	85–90%  
- Classes	Cat (0), Dog (1)  
- Input Size	128 x 128 grayscale  
- Feature Size	8100 (HOG) ➝ PCA ➝ SVM  
✅ Performance may vary based on dataset size and image quality.  

## 🖼️ GUI Feature  
Used Tkinter to build a graphical user interface allowing users to:  
Upload an image  
Predict whether the image is of a cat or a dog 

Special thanks to Prodigy Info Tech for the internship opportunity  
