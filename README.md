# Twitter-Sentiment-Analysis
This is an sentiment analysis dataset of twitter. Given a message and an entity, the task is to judge the sentiment of the message about the entity. There are three classes in this dataset: Positive, Negative and Neutral. Apart from this the irrelevant tweets that are not relevant to the entity (i.e. Irrelevant).

But while training the dataset the sentiment of the tweets is only considered and not the entity of the tweets.

The attempt is made to complete each step of project i.e. end to end project. The first step is of course to set the problem statement and the last step is project deployment.In this case the project is deployed using Flask framework.

This repository contains the jupyter notebook file named Twitter Sentiment Analysis.ipynb. Except deployment this file contains all the steps of project. First of all after collecting the data I have considered only two important features of dataset Review and Target and first 40000 rows because it was computationally very expensive to consider all rows.Then basic preprocessing of NLP is followed. Then two methods of word embeddings namely CountVectorizer and TfIdfVectorizer are used for converting the text to numbers.The reason behind this is to find which method performs better word embedding.

The two number of matrices are produced using these two methods. Then various base algorithms are trained on both these matrices. Then that one base algorithm whose accuracy is better is selected as final predictor algorithm. The pickle file of this algorithm along with other important base models are stored in folder 'Artifacts'. The python file main.py contains the code for deployment of project using Flask framework.

The folder named templates contains two html files which are used for taking the inputs from user and displaying the result.
