# -*- coding=utf-8 -*-
"""
Project: Queries Questions Clustering
Description: Text Clustering
Author:ZhangJunwen
email:jwzhangbu@126.com
Date:  Thu., July 01, 2020 AM 13:56AM
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import jieba
import json
import random
from gensim import corpora, models, matutils


class TextCluster(object):

    def __init__(self,
                 qids,
                 texts,
                 stop_words_file="clustering/utils/stopwords.txt",
                 theta=0.5,
                 shuffle=False
                 ):
        self.qids = qids
        self.texts = texts
        self.stop_words_file = stop_words_file
        self.theta = theta
        self.shuffle = shuffle

    @classmethod
    def text_parse(cls, raw_data):
        """
          Convert input json file into list type for next job of sentence clustering
        """
        with open(raw_data, 'r', encoding='utf-8') as f:
            dict_data = json.load(f)
            thr = dict_data['threshold']
            pdata = dict_data['data']
            shuf = dict_data['shuffle']
            plen = len(pdata)
            if shuf:
                random.shuffle(pdata)
            else:
                random.Random(2020).shuffle(pdata)
            queries = []
            ids = []
            for i in range(plen):
                queries.append(pdata[i]['question'])
                ids.append(pdata[i]['id'])
            return cls(qids=ids, texts=queries, theta=thr, shuffle=shuf)

    def word_segment(self, texts):
        stopwords = [lambda x: x.stip() for line in open(self.stop_words_file, 'r', encoding='utf-8').readlines()]
        word_segmentation = []
        words = jieba.cut(texts)
        for word in words:
            if word == ' ':
                continue
            if word not in stopwords:
                word_segmentation.append(word)
        return word_segmentation

    def get_Tfidf_vector_representation(self, word_segmentation, pivot=10, slope=0.1):
        """
        employ VSM(vector space model) to get documents' space vector, optionally doc2vec or other algorithms could be used
        to obtain sentence vector
        """
        # get the mapping relationship of voc and voc2id in form of dictionary
        dictionary = corpora.Dictionary(word_segmentation)
        # get the sentence vector representation
        corpus = [dictionary.doc2bow(word) for word in word_segmentation]
        # further to obtain Tfidf vector representation of the sentence
        tfidf = models.TfidfModel(corpus, pivot=pivot, slope=slope)
        corpus_tfidf = tfidf[corpus]
        return corpus_tfidf

    def getMaxSimilarity(self, dictTopic, vector):
        """
        cosine algorithm is used to compute the text similarity between the new entry of doc with the existing docs and other
        algorithms like kullback_leibler, jaccard, hellinger,etc of similarity compute methods
        """
        maxValue = 0
        maxIndex = -1
        for k, cluster in dictTopic.items():
            singleSimilarity = np.mean([matutils.cossim(vector, v) for v in cluster])
            if singleSimilarity > maxValue:
                maxValue = singleSimilarity
                maxIndex = k
        return maxIndex, maxValue

    def single_pass(self, corpus, qids, texts, theta):
        dictTopic = {}
        queryId = {}
        clusterTopic = {}
        numTopic = 0
        cnt = 0
        for vector, text, qid in zip(corpus, texts, qids):
            if numTopic == 0:
                dictTopic[numTopic] = []
                dictTopic[numTopic].append(vector)
                clusterTopic[numTopic] = []
                clusterTopic[numTopic].append(text)
                queryId[numTopic] = []
                queryId[numTopic].append(qid)
                numTopic += 1
            else:
                maxIndex, maxValue = self.getMaxSimilarity(dictTopic, vector)
                # first document as seed is built as topic, the subsequent given text is allocated to the most similar topic existing.
                if maxValue > theta:
                    dictTopic[maxIndex].append(vector)
                    clusterTopic[maxIndex].append(text)  # bug
                    queryId[maxIndex].append(qid)
                # or create a new topic
                else:
                    dictTopic[numTopic] = []
                    dictTopic[numTopic].append(vector)
                    clusterTopic[numTopic] = []
                    clusterTopic[numTopic].append(text)
                    queryId[numTopic] = []
                    queryId[numTopic].append(qid)
                    numTopic += 1
            cnt += 1
            if cnt % 1000 == 0:
                print("processing {}...".format(cnt))
        return dictTopic, clusterTopic, queryId

    def cluster_result(self, theta=0.25):
        """
        get the ultimate clustering results including text_id, cluster_num_id, key topic and key words
        """
        qidMat = self.qids
        datMat = self.texts
        word_segmentation = []
        for i in range(len(datMat)):
            word_segmentation.append(self.word_segment(datMat[i]))
        # get the space vector representation of text data
        corpus_tfidf = self.get_Tfidf_vector_representation(word_segmentation)
        dictTopic, clusterTopic, queryId = self.single_pass(corpus_tfidf, qidMat, datMat, theta)
        # select key cluster group and sort the clustered topics in descending order according to group numbers
        ldTopic = list(dictTopic.items())
        lclsTopic = list(clusterTopic.items())
        lqId = list(queryId.items())
        cls_id = 1
        cls_data = []
        iter_data = {}
        for idx, q in zip(lqId, lclsTopic):
            for x, y in zip(idx[1], q[1]):
                iter_data['query_id'] = x
                iter_data['question'] = y
                iter_data['cluster_id'] = cls_id
                cls_data.append(iter_data)
                iter_data = {}
            cls_id += 1
        result = {"responseMsg": "OK", "data": cls_data}
        print(result)


if __name__ == "__main__":
    # tc = TextCluster()
    pass
