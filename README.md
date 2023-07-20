# NLP on semantics

## Natural Language Processing

## Sentiment Classification

## Background

Welcome to challenge 4! In this challenge, we will be working with a subset of a dataset published in the ACL (Association for Computational Linguistics) 2019 conference. The task that the dataset addresses is of generating empathetic responses to a user utterance prompt by intelligent conversational agents; e.g. when somebody says "I finally got that promotion at work," a response like "Congratulations, that's great!" is more empathetic than saying "I can't believe anybody promoted you."

[The full paper is available at ](https://arxiv.org/pdf/1811.00207.pdf).

Goal

This challenge focuses on sentiment classification. The task is to predict the class of sentiments from a predefined list of emotions.

The data set originally contained 25K snippets of personal dialogue, labeled by 32 different emotions, which we will subset to contain only 4 emotions.

[You can download the dataset from the link below. We have also attached it to make it easy for you.](https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz)
The subset of emotions which you will classify is {'sad', 'jealous', 'joyful', 'terrified'}

Be aware that the dataset follows a format which is given by the first line in each csv file of the dataset. We will treat attribute 'utterance' as our training attribute and the labels will be the attribute 'context'.

Note that the dataset contains a train, dev and test. You should report your accuracies for test data wherever asked unless specified otherwise. Feel free to train on both train and dev data.
We have prepared some skeleton code to get you started, and pointed out some helpful technical blogs/material you may refer for this challenge. The skeleton code are just some helpful pointers---you do NOT need to follow it. Go crazy if you want!
In general, this challenge involves two parts:

1.  how to encode raw words into ML model digestible format, i.e., numerical matrix
2.  after the encoding, the problem reduces to a simple multiclassification problem, from text to one of the four sentiment types.
    This challenge asks you to do four encodings:

3.  use simple bag of words
4.  use TFIDF
5.  use pre-trained word2vec
6.  use pretrained distilled BERT
    For the first two, you could follow this
    blog:[Links to an external site.](https://maelfabien.github.io/machinelearning/NLP_2/#2-bow-in-sk-learn)
    For word2vec, you may refer to this article: [https://towardsdatascience.com/using-word2vec-](https://towardsdatascience.com/using-word2vec-) to-analyze-news-headlines-and-predict-article-success-cdeda5f14751 (Links to an external site.) For pretrained BERT, you may refer to this: [https://huggingface.co/distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)

H we will be focusing on building relevant features from text input. The traditional classifiers we looked at in previous challenges work well on data which contains numerical or categorical data. In this challenge we will be exploring more on how we can represent textual data in formats suitable for building classifier models.

1.  Data ETL
    he first step is to download the entire dataset which contains many columns and rows which we will not be needing.

    a. First, filter out all rows where the sentiment (‘context’) is not in the list of aforementioned list of sentiments i.e. {'sad', 'jealous', 'joyful', 'terrified'}.

This means you will only be retaining those examples where the sentiment lies within our set of predefined sentiments.

     b. Next, synthesize your training attributes and labels i.e. ‘utterance’ as the attributes and ‘context’ as the label.

2.  Begin by converting the utterances into a sparse bag-of-words (BOW) representation.

Recollect that BOW is essentially a matrix of width equal to the vocabulary size of the data and each row corresponding to each utterance.

In the sparse matrix obtained, each cell value 1 should represent that word j is present in utterance i and value of 0 indicates that it is not, where j is the column index and i is the row index.

3.  What do you think might be a shortcoming of the previous representation of utterance features? You may have guessed there are many words which are not necessarily adding much value to the classifier. For instance, words like ‘the’, ‘is’, ‘and’ are not words that help us humans identify whether an utterance belongs to a specific sentiment class. As you would remember, these words are referred to as ‘stop words’ in the NLP domain.

In this step, you will remove such stop words from the utterance and build the BOW features again so that your BOW representation is free of words that do not add much value to the classifier.

**Hint:** _Check out the NLTK library which has a precompiled list of popular stop words_

Hint: You may want to extend the existing NLTK stop words list based on some data exploration or your understanding of the domain. 4. Normalization: Another problem with the current representation is that we weigh each non-stop-word term the same. A proven way to normalize is Term Frequency - Inverse Document Frequency (TF-IDF). This should normalize frequencies in a weighted fashion to a value between 0 and 1.

**Hint:** _Look at how to use TfidfTransformer in Sklearn_

5.  Build a SGD classifier for the utterance sentiment classification and perform error analysis on the train data. The error analysis must include the test accuracy, confusion matrix and a few misclassified examples and your thoughts on why those utterances were misclassified by the example. You should aim for at least 60% accuracy on the test set.

6.  Build a classifier using pre-trained word embeddings like word2vec or gloVe or as the feature and an MLP classifier. Report the confusion matrix, F1 score and test accuracy.

7.  Build a classifier based on BERT and MLP. Specifically, use the pretrained distilbert-base-uncased to get sentence embeddings. You are not required to fine-tune this pretrained BERT model. And then train an MLP classifier. Explain how you use the BERT output. Specifically, which token(s) output you use? Report the confusion matrix, F1 score and test accuracy.

# What do you think might be a shortcoming of the previous representation of utterance features?

If the records contain new words, then size of the matrix would continue to increase which is not efficient.
There are many zeros in the vector thus resulting in a very sparse matrix which needs to be avoided.
There is not information on grammer from the sentences or ordering of the words in the text hence the text cannot be reconstructed without loosing information.
SGDClassifier Results.
On validation, JOY untterance was the one with lowest prediction score althought it was just less 1% from others.
The average accuracy for the model was 60% which means, for every text that is predicted, it is 60% accurate that it is in the correct label of utterance.
Most of the JOYFUL utterance were being predicted as JEALOUS with 70 values done this.
Below are results achieved by SGDclassifier model

========================================================

    Accuracy : |  59.65250965250966%               |
    F1 Score :  | 59.553625217041805%              |
    Recall Score : |  59.65250965250966%           |
    Precision Score : |  59.741581794172774%       |

=======================================================

MLP with Word2Vec results.
Below is the results achieved by MLP model.
gensim module was used to create word2vec model which was used in representation of text with similarities.
The model did not have good accuracy,

        Accuracy :   46.46074646074646%
        F1 Score :   46.445182667051874%
        Recall Score :   46.46074646074646%
        Precision Score :   46.64679340139845%

Explain how you use the BERT output. Specifically, which token(s) output you use?
Bert is trained using Masked Language Model (MLM) and Next Sentence Prediction task (NSP).
It recieves input as embedding i.e attention masks and sentences input ids.
In each word, the first token is a special token called CLS and sentences are separed by also a special token called SEP
The model's output are embedding for the hidden state. These output are the one to be feeded to layer that can be used for classification.
In our task, we were only interested on these output (Hidden State) which are associated with initial token CLS since it captures the meaning of the entire senteses more than other hence these hidden state can be used for classification with other models as input.
After training MLP model with features extracted using BERT model and tokenizers, the results had improved a bit as follows;
Accuracy : 60.5%
F1 Score : 60.24614807085745%
Recall Score : 60.5%
Precision Score : 60.650467636088266%

#
