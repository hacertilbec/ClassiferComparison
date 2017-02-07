import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.cross_validation import train_test_split

def mergeDataFrames(dataFrameList, mergeElements):
    """
    dataFrameList : list of dataframe names
    mergeElements : list of columns to merge given dataframes.
                    columns from mergeElements will be used to
                    merge dataframes from beginning to end,
                    so it should be ordered.
    """

    num_of_dfs = len(dataFrameList)
    new_df = None
    for i in range(1,num_of_dfs):
        if i==1:
            on_ = mergeElements[0]
            new_df = pd.merge(dataFrameList[0], dataFrameList[1], on=on_)
        else:
            on_ = mergeElements[i-1]
            new_df = pd.merge(new_df, dataFrameList[i], on=on_)

    return new_df


def evaluationmetrics(correctLabels, predictedLabels, predictProba):

    print "accuracy: ",
    print accuracy_score(correctLabels, predictedLabels)

    print "precision: ",
    print precision_score(correctLabels, predictedLabels, average=None)

    print "recall: ",
    print recall_score(correctLabels, predictedLabels, average=None)

    print "f1: ",
    print f1_score(correctLabels, predictedLabels, average=None)

    print "logloss: ",
    print log_loss(correctLabels, predictProba)


def classifierPerformances(df, labelColumn, classifierList, preprocessing = True, testSize = 0.3):
    """
    df : dataframe
    preprocessing : all columns except label column will be hashed
                    if preprocessing=True. For preprocessing=False,
                    dataframe will be used as it is, so all columns
                    must consist of numbers.
    testSize : test size while splitting data into train/test datasets
    labelColumn : label column
    classifierList : list of scikitlearn classifiers
    """

    featureColumns = list(df.columns)
    featureColumns.remove(labelColumn)

    train_set, test_set = train_test_split(df, test_size = testSize)

    X = train_set[featureColumns]
    y = train_set[labelColumn]

    test_x = test_set[featureColumns]
    test_y = test_set[labelColumn]

    if preprocessing == True :
        X = X.applymap(lambda x: hash(x))
        #y = y.apply(lambda x: int(x.strip('"')))

        test_x = test_x.applymap(lambda x: hash(x))
        #test_y = test_y.apply(lambda x: int(x.strip('"')))

    for classifier in classifierList:
        model = classifier.fit(X,y)
        predictions = model.predict(test_x)
        predictproba = model.predict_proba(test_x)
        print str(classifier).split('(')[0], '\n'

        evaluationmetrics(test_y, predictions, predictproba)
        print "\n----\n"      
