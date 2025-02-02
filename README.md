# Text Classification using LSTM and TOPSIS

## 📌 Project Overview
This project implements **text classification** using **traditional machine learning models** (Naïve Bayes, SVM, Random Forest) and a **deep learning model (LSTM)**. To objectively determine the best model, we apply **TOPSIS** (Technique for Order of Preference by Similarity to Ideal Solution), ranking models based on multiple performance metrics.

## 🔥 Features
- Uses **IMDb movie reviews dataset** for **sentiment analysis**.
- Compares **ML models (Naïve Bayes, SVM, Random Forest) with LSTM**.
- Uses **TF-IDF** for ML models and **Word Embeddings** for LSTM.
- Evaluates models based on **accuracy, precision, recall, F1-score, and inference time**.
- **Ranks models using TOPSIS** for optimal selection.
- **Visualizes results with bar charts**.

## 📁 Dataset
- **IMDb Movie Reviews** dataset from `nltk.corpus.movie_reviews`
- Contains **positive and negative** movie reviews

## 🚀 Technologies Used
- Python 🐍
- Scikit-learn 🤖
- TensorFlow/Keras 🔥
- NLTK 📚
- Matplotlib 📊

## 📜 Steps to Run
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/text-classification-topsis.git
   cd text-classification-topsis
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
4. Open and run `text_classification_topsis.ipynb`

## 📊 Results & Analysis
- **Performance of each model is evaluated** using standard classification metrics.
- **TOPSIS ranks the models** based on weighted performance scores.
- **Bar charts visualize** the rankings.

## 📌 Sample Output
| Model            | Accuracy | Precision | Recall | F1-Score | Inference Time | TOPSIS Score |
|----------------|----------|-----------|--------|----------|---------------|--------------|
| Naïve Bayes   | 85%      | 82%       | 87%    | 84%      | 0.05s         | 0.72         |
| SVM           | 87%      | 85%       | 88%    | 86%      | 0.12s         | 0.76         |
| Random Forest | 86%      | 83%       | 89%    | 85%      | 0.15s         | 0.74         |
| **LSTM**       | **89%**  | **88%**   | **90%**| **89%**  | **1.2s**      | **0.85**     |

📢 **Conclusion:** LSTM performed the best, but SVM and Random Forest provided competitive results with lower inference times!

## 📬 Contact & Contributions
Feel free to fork the project, raise issues, or contribute! Let's make **text classification ranking** even more insightful. 🚀

#NLP #MachineLearning #DeepLearning #TOPSIS #TextClassification
