import numpy as np
import pandas as pd
import math
from scipy.stats import norm

# split training set by class
def split_trainset(trainset):
    class_types = np.unique(trainset.iloc[-1])
    header_list = list(trainset.columns)
    header = header_list[-1]
    class_trainset = dict()
    for c_type in class_types:
        grouped = trainset.groupby(header)
        grouped_class = grouped.get_group(c_type)
        class_trainset[c_type] = grouped_class
    return class_trainset

# create mean trainset
def mean_trainset(class_trainset):
    classes = class_trainset.keys()
    trainset_mean = dict()
    for class_type in classes:
        class_df = class_trainset[class_type]
        get_mean = class_df.mean(axis=0, numeric_only=True)
        trainset_mean[class_type] = get_mean
    return trainset_mean

# create standard dev trainset
def std_trainset(class_trainset):
    classes = class_trainset.keys()
    trainset_std = dict()
    for class_type in classes:
        class_df = class_trainset[class_type]
        get_std = class_df.std(axis=0, numeric_only=True)
        trainset_std[class_type] = get_std
    return trainset_std

# gaussian probability function
def gauss_pdf(x, mean, std):
    return norm.pdf(x, mean, std)

# bayes theorem
def bayes(example, trainset_mean, trainset_std, class_type):
    series_mean = trainset_mean[class_type]
    series_std = trainset_std[class_type]
    return series_mean


# predict class
def predict(trainset_mean, trainset_std, example):
    class_prob = dict()
    classes = trainset_mean.keys()
    for class_type in classes:
        mean_series = trainset_mean[class_type]
        std_series = trainset_std[class_type]

        
        
        

