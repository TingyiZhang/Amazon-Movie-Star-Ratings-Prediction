# Amazon-Movie-Star-Ratings-Prediction
This is CS506 Spring 2020 Kaggle competition.

The goal of this competition is to predict star rating associated with user reviews from Amazon Movie Reviews using the available features.

The models are restricted to classical machine learning models that covered in class, deep learning models are not allow for this competition.

## Model Selection
- LinearSVC
- Naive Bayes
- Logistic Regression
- Voting Classifier to combine models

The final model takes a little bit long to train(for my machine, approximately 12 hours). So if you want to test it, use a small subset of data first, or decrease the iterations of logistic regression.

## Feature Extraction
- TfidVectorizer to extract reviews in trian.csv

## Results
The final score is here: https://www.kaggle.com/c/bu-cs506-spring-2020-midterm/leaderboard
