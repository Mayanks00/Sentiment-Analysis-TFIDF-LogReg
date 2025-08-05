## 📄 Internship Project Details
**PROJECT_NAME** : Sentiment-Analysis-TFIDF-LogReg  
**COMPANY** : CODETECH IT SOLUTIONS  
**NAME** : MAYANK PRATAP SINGH  
**INTERN_ID** : CT04DH1775  
**DOMAIN** : MACHINE LEARNING  
**DURATION** : 4 WEEKS  
**MENTOR** : Neela Santhosh Kumar  



# 💬 Sentiment Analysis on Amazon Product Reviews  
### Using TF-IDF Vectorization & Logistic Regression



## 📌 Project Overview
This project performs **sentiment analysis** on **Amazon product reviews**, classifying them as **Positive** or **Negative** using **Natural Language Processing (NLP)** and **Machine Learning**.  

We leverage:  
- **TF-IDF (Term Frequency-Inverse Document Frequency)** → to convert reviews into numeric features  
- **Logistic Regression** → to classify sentiments  

The trained model and vectorizer are **saved and reusable**, making the project **deployment-ready**.



## 🎯 Objectives
- Clean and preprocess raw text reviews  
- Convert text to **numeric vectors using TF-IDF**  
- Train a **Logistic Regression model** for binary classification  
- Evaluate performance using **Accuracy, Precision, Recall, and F1-score**  
- **Save and reload the model** for future predictions  
- Create **sample predictions CSV** for demonstration purposes


## 🌐 Real-World Applications
- 🛍 **E-commerce** → Analyze product reviews to improve customer satisfaction  
- 🎥 **Media & Entertainment** → Understand audience feedback for movies and shows  
- 🤖 **Chatbots** → Respond differently to positive/negative user messages  
- 📊 **Business Intelligence** → Monitor market sentiment in real-time  



## 🧠 Project Workflow

1️⃣ **Import Libraries** → Setup Python environment for NLP & ML  
2️⃣ **Upload Dataset** → Load Amazon review `.bz2` files  
3️⃣ **Parse Data** → Extract labels (`Positive`/`Negative`) and review text  
4️⃣ **Text Cleaning** → Remove punctuation, lowercase, and clean spaces  
5️⃣ **Feature Extraction (TF-IDF)** → Convert text into numerical vectors  
6️⃣ **Train Model (Logistic Regression)** → Binary sentiment classification  
7️⃣ **Evaluate Model** → Accuracy and classification report  
8️⃣ **Custom Predictions** → Test the model on any review text  
9️⃣ **Save Model & Vectorizer** → Export `.pkl` and `.json` files  
🔟 **Deployment Ready** → Load model without retraining  



## 📂 Project Structure-

Sentiment-Analysis-TFIDF-LogReg/

│
├── data/

│ ├── train.ft.txt.bz2

│ └── test.ft.txt.bz2
│
├── model/

│ ├── sentiment_model.pkl

│ ├── tfidf_vectorizer.pkl

│ └── tfidf_vocab.json

| |___field_classifier.pkl

├── notebooks/

│ └── Sentiment_Analysis_TFIDF_LogReg.ipynb
│
├── README.md

├── requirements.txt

└── .gitignore



## 📦 Installation & Setup


# Clone the repository-
git clone https://github.com/<Mayanks00>/Sentiment-Analysis-TFIDF-LogReg.git

# Navigate into the project folder-

cd Sentiment-Analysis-TFIDF-LogReg

# Install required Python packages-

pip install -r requirements.txt


▶️ Usage

Open the Jupyter notebook or run in Google Colab

Load the saved model and vectorizer

Run predictions on custom reviews or sample data


Example:


import joblib

# Load model and vectorizer
model = joblib.load("model/sentiment_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Predict sentiment
review = ["This product is amazing!"]
prediction = model.predict(vectorizer.transform(review))
print("Sentiment:", "Positive" if prediction[0] == 1 else "Negative")


📄 License
This project is developed as part of CODETECH IT SOLUTIONS Internship and is for educational and portfolio purposes only.






