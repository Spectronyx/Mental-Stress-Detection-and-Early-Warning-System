import pandas as pd
import glob
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- SETUP ---
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

# --- 1. LOAD OLD AND NEW DATA ---
# Let's say we have our original data and a new file
base_path = 'archive-2/Original Reddit Data/Labelled Data/'
all_files = glob.glob(os.path.join(base_path, "*.csv"))
all_files.append('new_mental_health_data.csv') # Adding our new file

print(f"Loading {len(all_files)} files...")
df_list = []
for f in all_files:
    if os.path.exists(f):
        df_list.append(pd.read_csv(f))

# Combine everything
df_full = pd.concat(df_list, ignore_index=True)
print(f"Total rows in combined dataset: {len(df_full)}")

# --- 2. PREPROCESS ---
df_full = df_full.dropna(subset=['selftext'])
df_full['full_text'] = df_full['title'].fillna('') + " " + df_full['selftext']

print("Cleaning combined text...")
df_full['cleaned_text'] = df_full['full_text'].apply(clean_text)

# --- 3. ANALYZE SENTIMENT ---
print("Analyzing sentiment on full dataset...")
df_full['sentiment'] = df_full['full_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# --- 4. RETRAIN MODELS ---
print("Retraining TF-IDF and KMeans...")
tfidf = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7)
tfidf_matrix = tfidf.fit_transform(df_full['cleaned_text'])

# Retraining KMeans on full data
k = 5
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
df_full['cluster'] = kmeans.fit_predict(tfidf_matrix)

print("\nUpdate Complete!")
print(f"Sample sentiment for new entries: \n{df_full[['full_text', 'sentiment']].tail(2)}")
print(f"New cluster distribution: \n{df_full['cluster'].value_counts()}")
