import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('D:/ML RA/df_de.csv/df_de.csv')

df = df.dropna(subset=['text'])

tfidf_vectorizer = TfidfVectorizer(max_features=1000)

X_tfidf = tfidf_vectorizer.fit_transform(df['text'])

df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

print(df_tfidf.head())
