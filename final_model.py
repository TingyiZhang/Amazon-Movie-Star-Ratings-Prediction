import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier

# loading data
df = pd.read_csv('./data/train.csv')

X_test = df[df.isnull().Score]

X_test = X_test['Text'].replace(np.nan, '', regex=True)

df.dropna(inplace=True)

X_train = df['Text']
y_train = df['Score']
vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2)).fit(X_train)
X_train_vectorized = vectorizer.transform(X_train)

# models
clf1 = LogisticRegression(max_iter=300)
clf2 = LinearSVC(max_iter=1000)
clf3 = MultinomialNB()

# ensemble
eclf = VotingClassifier([('lr', clf1), ('svm', clf2), ('nb', clf3)])
eclf.fit(X_train_vectorized, y_train)

prediction = pd.read_csv('./data/test.csv')
prediction['Score'] = eclf.predict(vectorizer.transform(X_test))
prediction.to_csv('submission.csv', index=False)