# -*- encoding: utf8 -*-
import re

import unicodedata

from pyvi.pyvi import ViTokenizer
from sklearn.ensemble import RandomForestClassifier

from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import datetime
import pandas as pd
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os

from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


def time_diff_str(t1, t2):
    diff = t2 - t1
    mins = int(diff / 60)
    secs = round(diff % 60, 2)
    return str(mins) + " mins and " + str(secs) + " seconds"

def load_model(model):
    print('loading model ...' + model)
    if os.path.isfile(model):
        return joblib.load(model)
    else:
        return None

def list_words(mes):
    words = mes.lower().split()
    return " ".join(words)

def clean_str_vn(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[~`@#$%^&*-+]", " ", string)
    def sharp(str):
        b = re.sub('\s[A-Za-z]\s\.', ' .', ' '+str)
        while (b.find('. . ')>=0): b = re.sub(r'\.\s\.\s', '. ', b)
        b = re.sub(r'\s\.\s', ' # ', b)
        return b
    string = sharp(string)
    string = re.sub(r" : ", ":", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def remove_stopword(question, filename):
    # 1. Convert to lower case, split into individual words
    # words = review.lower().split()
    words = question.split()
    with open(filename, "r") as f3:
        dict_data = f3.read()
        array = dict_data.splitlines()
    meaningful_words = [w for w in words if not w in array]
    return " ".join(meaningful_words)

def word_clean(array, review):
    words = review.lower().split()
    meaningful_words = [w for w in words if w in array]
    return " ".join(meaningful_words)


def build_sentence(input_arr):
    d = {}
    for x in range(len(input_arr)):
        d.setdefault(input_arr[x], x)
    chuoi = []
    for i in input_arr:
        x = d.get(i)
        if x == 0:
            chuoi.append(i)
        for j in input_arr:
            y = d.get(j)
            if y == x + 1:
                z = j.split(' ')
                chuoi.append(z[1])
    return " ".join(chuoi)

def clean_doc(question):
    rm_junk_mark = re.compile(ur'[?,\.\n]')
    normalize_special_mark = re.compile(ur'(?P<special_mark>[\.,\(\)\[\]\{\};!?:“”\"\`\'/])')
    question = normalize_special_mark.sub(u' \g<special_mark> ', question)
    question = rm_junk_mark.sub(u'', question)
    question = re.sub(' +', ' ', question)  # remove multiple spaces in a string
    return question

def load_data(filename):
    col1 = []; col2 = []; col3 = []; col4 = []
    with open(filename, 'r') as f:
        for line in f:
            label1, p, label2, question = line.split(" ", 3)
            question = list_words(question)
            question = clean_str_vn(question)
            # question = review_to_words(question,'datavn/question_stopwords.txt')
            # question = clean_str_vn(question)
            if filename == 'datavn/train':
                if label1 == "ABBR":
                    for i in xrange(1):
                        col1.append(label1)
                        col2.append(label2)
                        col3.append(question)
            col1.append(label1)
            col2.append(label2)
            col3.append(question)

        d = {"label1":col1, "label2":col2, "question": col3}

        train = pd.DataFrame(d)
        if filename == 'datavn/train':
            joblib.dump(train, 'model_balance/train_tfidf12.pkl')
        else:
            joblib.dump(train, 'model_balance/test_tfidf12.pkl')
    return train


def svm():
    train = load_model('model_balance/train_tfidf12.pkl')
    if train is None:
        train = load_data('datavn/train')

    vectorizer = load_model('model_balance/vectorizer_tfidf12.pkl')
    if vectorizer == None:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.7, min_df=2, max_features=1000)
    test = load_model('model_balance/test_tfidf12.pkl')
    if test is None:
        test = load_data('datavn/test')

    print "Data dimensions:", train.shape
    print "List features:", train.columns.values
    print "First review:", train["label1"][0], "|", train["question"][0]

    print "Data dimensions:", test.shape
    print "List features:", test.columns.values
    print "First review:", test["label1"][0], "|", test["question"][0]

    train_text = train["question"].values
    test_text = test["question"].values

    vectorizer.fit(train_text)
    X_train = vectorizer.transform(train_text)
    joblib.dump(vectorizer, 'model_balance/vectorizer_tfidf12.pkl')
    X_train = X_train.toarray()
    y_train = train["label1"]
    y_train2 = train["label2"]

    X_test = vectorizer.transform(test_text)
    X_test = X_test.toarray()
    y_test = test["label1"]
    y_test2 = test["label2"]
    print "---------------------------"
    print "Training"
    print "---------------------------"

    # iterate over classifiers
    clf = load_model('model_balance/tfidf12.pkl')
    if clf is None:
        t0 = time.time()
        clf = SVC(kernel='rbf', C=1000)
        clf.fit(X_train, y_train)
        joblib.dump(clf, 'model_balance/tfidf12.pkl')
        print " %s - Training completed %s" % (datetime.datetime.now(), time_diff_str(t0, time.time()))
    t1 = time.time()
    y_pred = clf.predict(X_test)
    print " %s - Converting completed %s" % (datetime.datetime.now(), time_diff_str(t1, time.time()))
    print " accuracy: %0.3f" % accuracy_score(y_test, y_pred)
    print " f1: %0.3f" % f1_score(y_test, y_pred, average='weighted')

    print "confuse matrix: \n", confusion_matrix(y_test, y_pred, labels=["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"])

    print "-----------------------"
    print "fine grained category"
    print "-----------------------"
    clf2 = load_model('model_balance/tfidf_fine12.pkl')
    if clf2 is None:
        t2 = time.time()
        clf2 = SVC(kernel='rbf', C=1000)
        clf2.fit(X_train, y_train2)
        joblib.dump(clf2, 'model_balance/tfidf_fine12.pkl')
        print " %s - Training for fine grained category completed %s" % (datetime.datetime.now(), time_diff_str(t2, time.time()))
    t3 = time.time()
    y_pred2 = clf2.predict(X_test)
    print " %s - Converting completed %s" % (datetime.datetime.now(), time_diff_str(t3, time.time()))
    print " accuracy for fine grained category: %0.3f\n" % accuracy_score(y_test2,y_pred2)
    print " f1: %0.3f" % f1_score(y_test2, y_pred2, average='weighted')
    with open('result/fail_en1.txt', "w") as f:
        list_y_test = test["label1"].tolist()
        y0 = y_pred.tolist()
        q = test["question"].tolist()
        for i in range(len(list_y_test)):
            if list_y_test[i] != y0[i]:
                f.write(y0[i] + "\t" + list_y_test[i] + "\t" + q[i] + "\n")
    with open('result/fail_en2.txt', "w") as f1:
        list_y_test = test["label2"].tolist()
        y1 = y_pred2.tolist()
        q = test["question"].tolist()
        for i in range(len(list_y_test)):
            if list_y_test[i] != y1[i]:
                f1.write(y1[i] + "\t" + list_y_test[i] + "\t" + q[i] + "\n")
    with open('result/pass_en1.txt', "w") as f2:
        list_y_test = test["label1"].tolist()
        y2 = y_pred.tolist()
        q = test["question"].tolist()
        for i in range(len(list_y_test)):
            if list_y_test[i] == y2[i]:
                f2.write(y2[i] + "\t" + list_y_test[i] + "\t" + q[i] + "\n")
    with open('result/pass_en2.txt', "w") as f3:
        list_y_test = test["label2"].tolist()
        y3 = y_pred2.tolist()
        q = test["question"].tolist()
        for i in range(len(list_y_test)):
            if list_y_test[i] == y3[i]:
                f3.write(y3[i] + "\t" + list_y_test[i] + "\t" + q[i]+ "\n")

def training1():
    train = load_model('model_balance/train_tfidf12.pkl')
    if train is None:
        train = load_data('datavn/train')

    vectorizer = load_model('model_balance/vectorizer_tfidf12.pkl')
    if vectorizer == None:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.7, min_df=2, max_features=1000)
    test = load_model('model_balance/test_tfidf12.pkl')
    if test is None:
        test = load_data('datavn/test')
    train_text = train["question"].values
    test_text = test["question"].values

    vectorizer.fit(train_text)
    X_train = vectorizer.transform(train_text)
    joblib.dump(vectorizer, 'model_balance/vectorizer_tfidf12.pkl')
    X_train = X_train.toarray()
    y_train = train["label1"]
    y_train2 = train["label2"]

    X_test = vectorizer.transform(test_text)
    X_test = X_test.toarray()
    y_test = test["label1"]
    y_test2 = test["label2"]
    print "---------------------------"
    print "Training"
    print "---------------------------"
    clf = load_model('model_balance/tfidf12.pkl')
    if clf is None:
        clf = SVC(kernel='rbf', C=1000)
        clf.fit(X_train, y_train)
        joblib.dump(clf, 'model_balance/tfidf12.pkl')

    clf2 = load_model('model_balance/tfidf_fine12.pkl')
    if clf2 is None:
        clf2 = SVC(kernel='rbf', C=1000)
        clf2.fit(X_train, y_train2)
        joblib.dump(clf2, 'model_balance/tfidf_fine12.pkl')


def predict_ex(mes):
    print mes
    vectorizer = load_model('model_balance/vectorizer_tfidf12.pkl')
    clf = load_model('model_balance/tfidf12.pkl')
    clf2 = load_model('model_balance/tfidf_fine12.pkl')
    if clf is None or clf2 is None:
        training1()
        clf = load_model('model/model_balance/tfidf12')
        clf2 = load_model('model_balance/tfidf_fine12.pkl')

    mes = unicodedata.normalize("NFC", mes.strip())
    mes = clean_str_vn(mes)
    test_message = ViTokenizer.tokenize(mes).encode('utf8')
    test_message = clean_str_vn(test_message)
    test_message = list_words(test_message)
    clean_test_reviews = []
    clean_test_reviews.append(test_message)
    d2 = {"message": clean_test_reviews}
    test2 = pd.DataFrame(d2)
    test_text2 = test2["message"].values.astype('str')
    test_data_features = vectorizer.transform(test_text2)
    test_data_features = test_data_features.toarray()
    # print test_data_features
    s = clf.predict(test_data_features)[0]
    s2 = clf2.predict(test_data_features)[0]
    return s + " " +s2

if __name__ == '__main__':
    svm()
    # rf()
