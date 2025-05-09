import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

df = pd.read_csv('D:/ML RA/df_de.csv/df_de.csv')  

#----- Tokenization starts -----
tokens_list = []

for text in df['text']:
    tokens = word_tokenize(text) 
    tokens_list.append(tokens)

df['tokens'] = tokens_list

print("tokens:\n", df['tokens'].head())

#----- Stop words removal starts -----
stop_words = set(stopwords.words('english'))

S_words = [word for word in tokens if word.lower() not in stop_words]

print("tokens without Stop Words:\n", S_words)

#----- stemming starts -----
stemming = PorterStemmer()

stemmed_words = [stemming.stem(word) for word in S_words]

print("Stemmed Words:\n", stemmed_words)

#----- lemmatization starts -----
lemmatizer = WordNetLemmatizer()

lemmatized_words = [lemmatizer.lemmatize(word) for word in S_words]
print("Lemmatized Words:\n", lemmatized_words)
