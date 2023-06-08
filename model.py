import numpy as np
import pandas as pd
import joblib
df = pd.read_csv('Downloads/31640102-airline-tweets.csv')
data = df[['airline_sentiment','text']]
X = df['text']
y = df['airline_sentiment']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=101)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')

tfidf.fit(X_train)

X_train_tfidf = tfidf.transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

from sklearn.naive_bayes  import MultinomialNB

nb = MultinomialNB()
nb.fit(X_train_tfidf,y_train)

from sklearn.linear_model import LogisticRegression


log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_tfidf,y_train)

from sklearn.svm import SVC,LinearSVC 
rbf_svc = SVC()
rbf_svc.fit(X_train_tfidf,y_train)

linear_svc = LinearSVC()
linear_svc.fit(X_train,y_train)

from sklearn.pipeline import Pipeline
pipe = Pipeline([('tfidf',TfidfVectorizer()),('svc',LinearSVC())])
pipe.fit(X,y)
feedback = st.text_input("Kindly Enter your feedback here:")
review = pipe.predict([feedback])

joblib.dump(pipe,'rf_model.sav')
