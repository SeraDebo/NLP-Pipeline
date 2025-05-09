import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('D:/ML RA/df_de.csv/df_de.csv')

print(df['text'].head())

vectorizer = CountVectorizer(max_features=1000)

X = vectorizer.fit_transform(df['text'].astype(str))  

bow_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

print(bow_df.head())

#---- N-Grams -----

print("\n--- Unigrams ---")
vectorizer_uni = CountVectorizer(ngram_range=(1, 2))  
X_uni = vectorizer_uni.fit_transform(df['text'])
df_uni = pd.DataFrame(X_uni[:100].toarray(), columns=vectorizer_uni.get_feature_names_out())
print(df_uni.head())

# ---------- BIGRAMS ----------
print("\n--- Bigrams ---")
vectorizer_bi = CountVectorizer(ngram_range=(2, 2)) 
X_bi = vectorizer_bi.fit_transform(df['text'])
df_bi = pd.DataFrame(X_bi[:100].toarray(), columns=vectorizer_bi.get_feature_names_out())
print(df_bi.head())

# ---------- TRIGRAMS ----------
print("\n--- Trigrams ---")
vectorizer_tri = CountVectorizer(ngram_range=(3, 3))  
X_tri = vectorizer_tri.fit_transform(df['text'])
df_tri = pd.DataFrame(X_tri[:100].toarray(), columns=vectorizer_tri.get_feature_names_out())
print(df_tri.head())
