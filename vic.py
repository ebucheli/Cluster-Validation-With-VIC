import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

from sklearn.metrics import roc_curve, auc

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier

from tqdm import tqdm

import os


def main():

    clusters_path = './QS_Partitions_CSV'

    cluster_files = os.listdir(clusters_path)
    cluster_files.sort()

    all_vs = []

    for cluster_file in cluster_files:

        print('Working on Cluster {}'.format(cluster_file))

        df = pd.read_csv(os.path.join(clusters_path,cluster_file))
        df = df.drop([df.columns[0],df.columns[1]],axis = 1)

        values = df.values
        y = values[:,-1]
        x = values[::-1]

        kf = KFold(n_splits = 10, shuffle = True, random_state = 0)
        kf.shuffle
        kf.get_n_splits(x)


        v = 0
        v_prime1 = 0
        v_prime2 = 0
        v_prime3 = 0
        #v_prime4 = 0
        v_prime5 = 0

        clf1 = []
        aucs1 = []
        clf2 = []
        aucs2 = []
        clf3 = []
        aucs3 = []
        #clf4 = []
        #aucs4 = []
        clf5 = []
        aucs5 = []

        i = 0

        for train_index, test_index in kf.split(x):

            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf1.append(RandomForestClassifier(n_estimators = 100,max_depth = 10,random_state = 0))
            clf2.append(SVC())
            clf3.append(GaussianNB())
            #clf4.append(LinearDiscriminantAnalysis())
            clf5.append(GradientBoostingClassifier())


            clf1[i].fit(X_train,y_train)
            y_hat_this_1 = clf1[i].predict(X_test)
            fpr1, tpr1, thresholds1 = roc_curve(y_test,y_hat_this_1)
            this_auc_1 = auc(fpr1,tpr1)
            v_prime1 += this_auc_1


            clf2[i].fit(X_train,y_train)
            y_hat_this_2 = clf2[i].predict(X_test)
            fpr2, tpr2, thresholds2 = roc_curve(y_test,y_hat_this_2)
            this_auc_2 = auc(fpr2,tpr2)
            v_prime2 += this_auc_2


            clf3[i].fit(X_train,y_train)
            y_hat_this_3 = clf3[i].predict(X_test)
            fpr3, tpr3, thresholds3 = roc_curve(y_test,y_hat_this_3)
            this_auc_3 = auc(fpr3,tpr3)
            v_prime3 += this_auc_3


            #clf4[i].fit(X_train,y_train)
            #y_hat_this_4 = clf4[i].predict(X_test)
            #fpr4, tpr4, thresholds4 = roc_curve(y_test,y_hat_this_4)
            #this_auc_4 = auc(fpr4,tpr4)
            #v_prime4 += this_auc_4


            clf5[i].fit(X_train,y_train)
            y_hat_this_5 = clf5[i].predict(X_test)
            fpr5, tpr5, thresholds5 = roc_curve(y_test,y_hat_this_5)
            this_auc_5 = auc(fpr5,tpr5)
            v_prime5 += this_auc_5

        this_v = np.max((v_prime1/10,v_prime2/10,v_prime3/10,v_prime5/10))
        print('V = {}\n'.format(this_v))
        all_vs.append(this_v)
    max_v = np.argmax(all_vs)
    print("Max v: {}".format(cluster_files[max_v]))


if __name__ == '__main__':
    main()
