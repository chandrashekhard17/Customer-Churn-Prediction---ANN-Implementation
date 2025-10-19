# 🧠 Customer Churn Prediction — ANN Implementation (Streamlit App)

## 📘 Project Overview
The **Customer Churn Prediction** project aims to predict whether a customer is likely to leave a telecom company or continue using its services.  
This project uses an **Artificial Neural Network (ANN)** model for classification and includes a **Streamlit web app** for user interaction.

The goal is to help businesses **identify high-risk customers** and take proactive steps to improve customer retention.

---

## 🚀 Features
- Built using **Artificial Neural Networks (ANN)** for churn classification  
- Interactive **Streamlit** web interface  
- Accepts customer details and predicts churn probability in real-time  
- Clean and modular codebase  
- Includes **requirements.txt** for easy deployment

---

## 🧩 Tech Stack
**Programming Language:** Python  
**Frameworks & Libraries:**
- `TensorFlow` / `Keras` – for ANN model building  
- `NumPy`, `Pandas` – for data manipulation  
- `Matplotlib`, `Seaborn` – for visualization  
- `Scikit-learn` – for preprocessing and metrics  
- `Streamlit` – for web app interface  
- `Joblib` / `Pickle` – for model saving and loading

---

## 🧱 Project Structure
```
Customer-Churn-Prediction---ANN-Implementation/
│
├── app.py                 # Streamlit app file
├── requirements.txt       # List of dependencies
├── model/                 # Saved ANN model and scaler files
├── data/                  # Dataset used for training and testing
├── notebooks/             # Jupyter/Colab notebooks for experimentation
└── README.md              # Project documentation (this file)
```

---

## ⚙️ Installation and Setup

### 1️⃣ Clone this repository
```bash
git clone https://github.com/chandrashekhard17/Customer-Churn-Prediction---ANN-Implementation.git
```

### 2️⃣ Navigate to the project directory
```bash
cd Customer-Churn-Prediction---ANN-Implementation
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit app
```bash
streamlit run app.py
```

---

## 📊 Model Overview
- The ANN model is built using **Keras Sequential API**.  
- It includes multiple dense layers, ReLU activations, and a sigmoid output for binary classification.  
- The model is trained on telecom customer data to predict churn.

**Target Variable:**  
`Exited` → 1 if customer churned, 0 otherwise

**Evaluation Metrics:**
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix

---

## 🌐 Deployment
The project is designed to be deployed on **Streamlit Cloud** or any Python-compatible hosting service.

To deploy:
1. Push the repository to GitHub (already done ✅)
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Connect your GitHub repo and deploy the app.

---

## 💡 How to Use the App
1. Open the Streamlit app.  
2. Enter customer details (e.g., Age, Gender, Tenure, Balance, etc.)  
3. Click on **Predict**.  
4. The app will display whether the customer is likely to **churn** or **stay**.

---

## 📈 Results & Insights
- The ANN model achieves high predictive accuracy on test data.  
- Important features influencing churn include:
  - Credit score  
  - Tenure  
  - Balance  
  - Number of products  
  - Geography and Gender

---

## 👨‍💻 Author
**Chandrashekhar D**  
📧 [chandrashekhard543@gmail.com]
💻 Data Science & Machine Learning Enthusiast

---

## 🏁 Acknowledgments
- Datasets inspired by telecom customer churn analysis projects.  
- Streamlit for providing an easy-to-use deployment platform.  
- TensorFlow/Keras community for ANN tutorials and references.

---

## 📜 License
This project is open-source and available under the **MIT License**.
