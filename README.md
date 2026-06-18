# 📧 Spam Email Classifier

A production-ready **Spam Email Classifier** built with Python, Streamlit, and scikit-learn.  
It uses **TF-IDF Vectorization** and a **Multinomial Naive Bayes** model to classify emails as **Spam** or **Not Spam** with a confidence score.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 ML Model | Multinomial Naive Bayes with TF-IDF |
| 🎨 Modern UI | Clean, responsive Streamlit interface |
| 📊 Confidence Score | Shows prediction probability |
| 🧪 Sample Emails | Pre-loaded examples for quick testing |
| ⚡ Fast Inference | Saved model loads in milliseconds |
| 📱 Mobile Friendly | Works on any screen size |
| 🔄 Retrainable | Separate training script with full metrics |

---

## 📁 Project Structure

```
spam-email-classifier/
│── app.py                  # Streamlit web application
│── train_model.py          # Model training & evaluation script
│── predict.py              # Prediction module
│── requirements.txt        # Python dependencies
│── README.md               # Project documentation
│── .gitignore              # Git ignore rules
│── model.pkl               # Trained model (generated)
│── vectorizer.pkl          # TF-IDF vectorizer (generated)
│── dataset/
│     └── spam.csv          # SMS Spam Collection dataset
│── assets/
│     └── logo.png          # Optional branding
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/spam-email-classifier.git
cd spam-email-classifier
```

### 2. Create a virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download the **SMS Spam Collection** dataset:

- [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- [Kaggle Mirror](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

Place the CSV file at `dataset/spam.csv`.

### 5. Train the model

```bash
python train_model.py
```

You will see accuracy, precision, recall, F1, and a confusion matrix printed to the console.  
Two files are generated: `model.pkl` and `vectorizer.pkl`.

### 6. Run the Streamlit app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## 📸 Screenshots

> _Add screenshots here after running the app._

| Home | Prediction |
|------|------------|
| ![Home](assets/screenshot_home.png) | ![Result](assets/screenshot_result.png) |

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Click **New app** → select your repo → set **Main file** to `app.py`.
4. Click **Deploy**.

> **Note:** Upload `model.pkl` and `vectorizer.pkl` to your repo,  
> or add a startup script that runs `train_model.py` before the app starts.

---

## 🧪 Model Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~97%   |
| Precision | ~100%  |
| Recall    | ~87%   |
| F1 Score  | ~93%   |

_(Exact values depend on the random split.)_

---

## 📜 License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).  
Feel free to use, modify, and distribute.

---

## 🤝 Contributing

Pull requests are welcome!  
For major changes, please open an issue first to discuss what you would like to change.

---

Built with ❤️ using **Python · Streamlit · scikit-learn · NLP**