# Web scraping to gain company insights

## Introduction

Customers who book a flight with BA will experience many interaction points with the BA brand. Understanding a customer's feelings, needs, and feedback is crucial for any business, including BA.

## Objective

The team leader wants you to focus on reviews specifically about the airline itself and perform your own analysis to uncover some insights. This first task is focused on scraping and collecting customer feedback and reviewing data from a third-party source and analysing this data to present any insights you may uncover. You could look at topic modelling, sentiment analysis or wordclouds to provide some insight into the content of the reviews.

## NLP Task

The project involves unsupervised learning and natural language processing. To extract insights from text data using web scraping, text preprocessing, exploratory data analysis, and data modeling. The project will involve the following steps:

1. Webscrapping: collect titles and reviews from 20 pages of a website called *Skytrax*.
2. Text preprocesing: clean the text data and prepare it for analysis.
3. EDA: use wordclouds and bar plots to visualize the most common words in the data and gain preliminary insight to the types of sentiment.
5. Data modeling
    - PCA: encode the text data and project it onto a smaller space to visualize if there are any clusters that might hint at different types of sentiment.
    - Topic modeling: use LDA to find the most common topics in the reviews.
    - Sentiment analysis: 
        - Embed the text data using Word2Vec and model the sentiment using K-means clustering.
        <!-- - run a basic NN using one of BERT's pretrained models to predict the sentiments of the reviews. -->
