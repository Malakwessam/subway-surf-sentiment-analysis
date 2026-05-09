# subway-surf-sentiment-analysis
This project analyzes Google Play Store reviews by scraping, cleaning, and processing user feedback, then applying multiple sentiment analysis techniques. It combines rating‑based, VADER, and AFINN methods with majority voting to produce reliable ground‑truth sentiment labels for user reviews
# Sentiment Analysis on Google Play App Reviews

## Overview
This project implements an end-to-end **sentiment analysis pipeline** on Google Play Store app reviews.  
The workflow starts from **web scraping raw user reviews**, followed by **data cleaning and annotation**, then **text representation**, and finally **machine learning models** to predict sentiment.  
The goal is to build reliable sentiment labels and evaluate different text representation techniques and models.



## Project Pipeline

### 1. Web Scraping
**What was done:**  
- Scraped recent Google Play Store reviews using `google-play-scraper`.
- Extracted review text, rating, usernames, replies, and app version.
- Saved the raw data into CSV files for reproducibility.
- 

### 2. Data Cleaning & Preprocessing
**What was done:**  
- Removed unnecessary columns (timestamps, images).
- Cleaned text by removing emojis, URLs, special characters, and extra spaces.
- Normalized text to lowercase.
- Detected language and kept only English reviews.
- Standardized numerical fields such as ratings and likes.


### 3. Sentiment Annotation & Ground Truth Creation
**What was done:**  
- Generated sentiment labels using **three methods**:
  - Rating-based heuristic (Positive / Neutral / Negative).
  - **VADER** sentiment analysis.
  - **AFINN** lexicon-based scoring with negation handling.
- Applied **majority voting** to produce a final `ground_truth_sentiment`.
- Evaluated agreement between annotators using **Cohen’s Kappa**.



### 4. Text Representation
**What was done:**  
- Converted review text into numerical features using:
  - **TF-IDF** (bag-of-words with importance weighting).
  - **GloVe embeddings** (100-dimensional pretrained word vectors).


### 5. Machine Learning Models
**What was done:**  
- Trained and evaluated models using the generated features:
  - **Linear SVM** with TF-IDF features.
  - **Linear Regression** with GloVe embeddings (mapped back to sentiment classes).
- Measured performance using accuracy and classification reports.



## Results Summary
- Final dataset size: **63 reviews**
- Sentiment distribution:
  - Positive: 48
  - Neutral: 10
  - Negative: 5
- Best accuracy achieved:
  - **TF-IDF + Linear SVM ≈ 77%**
  - **GloVe + Linear Regression ≈ 69%**



## Technologies Used
- Python, Pandas, NumPy
- google-play-scraper
- Scikit-learn
- VADER, AFINN
- Gensim (GloVe)



## Output Files
- `cleaned_subway_surfers_reviews.csv`
- `final_ground_truth_dataset.csv`
- `tfidf_features.csv`
- `glove_features.csv`



This project demonstrates a complete NLP workflow from raw data collection to sentiment prediction using multiple representations and models.
