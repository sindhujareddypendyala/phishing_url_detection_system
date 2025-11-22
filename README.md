🔗🕵️‍♀️Phishing URL Detection System

A machine learning–based web application built using Python, Streamlit, Scikit-learn, Pandas, NumPy, and RandomForestClassifier to classify URLs as Legitimate or Phishing.

🔗🚀 Project Overview

The Phishing URL Detection System analyzes URLs and predicts whether they are safe or malicious.
It extracts key features from URLs such as domain length, presence of suspicious keywords, number of dots, HTTPS usage, and more.
The system uses a trained RandomForestClassifier model and provides results through a simple Streamlit UI.

🔗🧠 Features

✔️ Detects phishing URLs with Machine Learning

✔️ Extracts URL-based features automatically

✔️ Simple and clean Streamlit interface

✔️ Fast and real-time predictions

✔️ Model trained using RandomForestClassifier

✔️ Includes pre-trained model saved using joblib

🔗🛠️ Technologies Used

Python

Streamlit

Scikit-learn

Pandas, NumPy

Joblib

RandomForestClassifier

urllib

🔗📂 Project Structure
phishing_url_detection_system/
│── model/
│     └── rf_model.pkl
│── app.py
│── feature_extraction.py
│── requirements.txt
│── README.md

🔗🧪 How It Works

User enters a URL into the Streamlit app

The system extracts ML features from the URL

The trained RandomForest model predicts

The result is shown as Phishing / Safe URL

🔗▶️ How to Run the Project
1. Clone the Repository
git clone <your-repo-link>

2. Install Dependencies
pip install -r requirements.txt

3. Run Streamlit App
streamlit run app.py

🔗📊 Machine Learning Model

Algorithm: RandomForestClassifier

Dataset: Collection of phishing + legitimate URLs

Evaluation metrics: Accuracy, Precision, Recall

Model saved using joblib

🔗🌱 Future Enhancements

Add deep learning LSTM model

Deploy online using Streamlit Cloud

Add browser extension

Integrate real-time blacklist API checks

Create admin dashboard with analytics

🔗📄 Conclusion

This project successfully identifies phishing URLs using a machine learning–based approach, offering users a fast and effective way to detect malicious websites. It improves internet security and reduces the risk of phishing attacks.

🔗👩‍💻 Author

Sindhuja Reddy Pendyala
B.Tech – Data Science
Machine Learning & Web Development Enthusiast
