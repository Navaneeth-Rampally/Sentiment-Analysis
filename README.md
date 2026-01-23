# 🎭 IMDB Sentiment Analysis Pipeline

## 📖 Project Overview
This project is a modular, end-to-end Machine Learning pipeline designed to perform **Sentiment Analysis** on the IMDB Movie Reviews dataset. Beyond just building a model, the focus of this project was to create a robust **Data Engineering** workflow.

The system automates the ingestion of data from Hugging Face, processes raw text using advanced NLP techniques, and provides a real-time web interface for users to test the model. It is architected with production-grade standards, including comprehensive **logging** and **modular code structure** to ensure scalability and easy debugging.

## 🚀 Key Features
* **Automated Data Ingestion:** Direct API fetching and structuring of the IMDB dataset from Hugging Face.
* **Advanced Text Preprocessing:** A dedicated engine for cleaning, tokenization, stopword removal, and lemmatization to prepare high-quality data.
* **Modular Architecture:** Code is organized into separate modules (Ingestion, Preprocessing, Evaluation) rather than a single script.
* **System Logging:** Integrated logging to track execution flow, errors, and performance metrics during runtime.
* **Interactive UI:** A user-friendly web application built with Streamlit for real-time inference.

## 🛠️ Technologies & Tools Used

### Core Stack
* **Language:** `Python 3`
* **Web Framework:** `Streamlit` (for the user interface)
* **Deployment:** `AWS EC2` (Ubuntu Linux)

### Libraries & Frameworks
| Category | Library | Purpose |
| :--- | :--- | :--- |
| **Data Manipulation** | `Pandas`, `NumPy` | Handling structured data and numerical operations. |
| **NLP** | `NLTK` | Tokenization, Stopwords removal, Lemmatization. |
| **Data Source** | `Datasets` (Hugging Face) | Fetching the IMDB benchmark dataset efficiently. |
| **Machine Learning** | `Scikit-Learn` | Model building, training, and evaluation pipelines. |
| **Visualization** | `Matplotlib`, `Seaborn` | Visualizing data distribution and model performance. |