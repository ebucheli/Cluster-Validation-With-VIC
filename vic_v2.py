import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

from sklearn.metrics import roc_curve, auc

import models

from tqdm import tqdm

import os


def main():

    clusters_path = './QS_Partitions_CSV'

    cluster_files = os.listdir(clusters_path)
    cluster_files.sort()

    all_vs = []

    f = open('./vic_report.txt','w')
    f.write('=====VIC Cluster Validation Report=====\n\n')

    #with open('./vic_report.txt','w') as f:
        #f.write('=====VIC Cluster Validation Report=====\n\n')

    for cluster_file in cluster_files:

        print('Working on Partition {}'.format(cluster_file.split('.')[0]))

        df = pd.read_csv(os.path.join(clusters_path,cluster_file))
        df = df.drop([df.columns[0],df.columns[1]],axis = 1)

        values = df.values
        y = values[:,-1]
        x = values[::-1]

        kf = KFold(n_splits = 10, shuffle = True, random_state = 0)
        kf.shuffle
        kf.get_n_splits(x)

        v = 0

        classifiers = ['random_forest','svm','naive_bayes','lda','gradient_boosting']

        for classifier in tqdm(classifiers):

            v_prime = 0
            for train_index, test_index in kf.split(x):

                X_train, X_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]

                this_auc = models.train_and_test(classifier,X_train,y_train,X_test,y_test)

                v_prime = v_prime + this_auc

            v = max(v,v_prime/10)

        print('\nDone! V = {}\n'.format(v))
        all_vs.append(v)

        #with open('./vic_report.txt', 'w') as f:
        f.write('{}: v = {}\n'.format(cluster_file.split('.')[0],v))

    max_v = np.argmax(all_vs)
    print("Max v: {}".format(cluster_files[max_v].split('.')[0]))

    #with open('./vic_report.txt', 'w') as f:
    f.write('\n\n==========\n\n'.format(cluster_file,v))
    f.write('Best v value for partition {} with {}'.format(cluster_files[max_v].split('.')[0],all_vs[max_v]))



if __name__ == '__main__':
    main()
