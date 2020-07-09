# -*- coding=utf-8 -*-
"""
Project: Queries Questions Clustering
Description: Text Clustering
Author:ZhangJunwen
email:jwzhangbu@126.com
Date:  Thu., July 01, 2020 AM 13:56AM
"""
import os
from clustering.algorithms.query_clustering import TextCluster

abs_path = os.path.abspath('.')


def get_cluster():
    cluster_data = os.path.join(abs_path, "clustering", "data", "query.json")
    TextCluster.text_parse(cluster_data).cluster_result()
