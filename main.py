import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

# Currently a program that can detect if an input is considered a spam message

#open("/users/alexra/Downloads/archive(1)/youtube-spam-collection-v1/Youtube01-Psy.csv")

data = pd.read_csv("Youtube01-Psy.csv")
data = data[["CONTENT","CLASS"]]

data["CLASS"] = data["CLASS"].map({
    0:"Not Spam",
    1:"Spam"
})

x = np.array(data["CONTENT"])
y = np.array(data["CLASS"])

cv = CountVectorizer()
# sklearn feature that counts the number of instances of a term in each document
# where a term is a word and document is a string of words

x = cv.fit_transform(x) 
xtrain,xtest,ytrain, ytest = train_test_split(
    x,y,
    test_size = 0.2,
    random_state = 42
)

model = BernoulliNB()
model.fit(xtrain,ytrain)
print(model.score(xtest,ytest)) # printing the accuracy of the model
# using bernoulli Naive Bayes Algorithm to Train Model
# model is 98% correct

sample = input()
data = cv.transform([sample]).toarray()
print(model.predict(data))


