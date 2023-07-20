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

# Project Name README

## Shortcoming of the Previous Utterance Features Representation

The previous representation of utterance features had several shortcomings:

1. **Inefficiency with New Words**: As the records contain new words, the size of the matrix would continuously increase, leading to inefficiency in storage and computation.

2. **Sparse Matrix**: The representation resulted in a very sparse matrix due to many zeros in the vector. This sparsity can be problematic and should be avoided.

3. **Lack of Grammar and Word Ordering Information**: The representation did not consider grammar or word ordering in the text, making it impossible to reconstruct the text accurately without losing information.

## SGDClassifier Results

The following are the results obtained using the SGDClassifier model on validation data:

- The utterance labeled as JOY had the lowest prediction score, but it was only less than 1% different from the others.
- The average accuracy of the model was 60%, meaning it correctly predicted the label for 60% of the texts.
- Most JOYFUL utterances were incorrectly predicted as JEALOUS, with 70 instances of misclassification.

SGDClassifier Results:

| Metric          | Score  |
| --------------- | ------ |
| Accuracy        | 46.46% |
| F1 Score        | 46.45% |
| Recall Score    | 46.46% |
| Precision Score | 46.65% |

## MLP with Word2Vec Results

The results obtained using the MLP model with Word2Vec embeddings were not very promising, indicating that the accuracy of the model was not satisfactory.

| Metric          | Score               |
| --------------- | ------------------- |
| Accuracy        | 46.46074646074646%  |
| F1 Score        | 46.445182667051874% |
| Recall Score    | 46.46074646074646%  |
| Precision Score | 46.64679340139845%  |

wordvec metrics

The gensim module was used to create the Word2Vec model for representing text with similarities.

## How BERT Output is Used

The BERT model is trained using a Masked Language Model (MLM) and a Next Sentence Prediction task (NSP). It takes embeddings, attention masks, and sentence input IDs as input. Each word is represented with an embedding, and the first token (CLS) represents the entire sentence, while the special token (SEP) is used to separate sentences.

The output of BERT is a hidden state, which contains embeddings for each word. However, for our task, we are primarily interested in the hidden state associated with the initial token (CLS). This token captures the overall meaning of the entire sentence, making it suitable for classification tasks.

After training the MLP model using features extracted from BERT and tokenizers, the results showed some improvement:

### Metrics using MLP with BERT Embeddings

| Metric          | Score  |
| --------------- | ------ |
| Accuracy        | 60.5%  |
| F1 Score        | 60.25% |
| Recall Score    | 60.5%  |
| Precision Score | 60.65% |

These results demonstrate that using BERT
