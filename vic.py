import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc

import models

from tqdm import tqdm

import os
import click

@click.command()
@click.option('--clusters_path',
              default = './QS_Partitions_CSV',
              help = 'Directory containing the partitions to analyize')
@click.option('--outfile', default = './vic_report.txt', help = 'Name for report file')

def main(clusters_path, outfile):

    cluster_files = [f for f in os.listdir(clusters_path) if f.endswith('.csv')]
    cluster_files.sort()

    all_vs = []
    all_scores = []
    f = open(outfile,'w')
    f.write('=====VIC Cluster Validation Report=====\n\n')

    classifiers = ['random_forest','svm','naive_bayes','lda','gradient_boosting', 'logistic_regression']


    f.write('Num of Classifiers: {}\n'.format(len(classifiers)))
    f.write('Chosen Classifiers:\n')
    for classifier in classifiers:
        f.write('\t{}\n'.format(classifier))
    f.write('\n\n')

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

        for classifier in tqdm(classifiers):
            v_prime = 0
            for train_index, test_index in kf.split(x):

                X_train, X_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]

                this_auc = models.train_and_test(classifier,X_train,y_train,X_test,y_test)

                v_prime = v_prime + this_auc

            all_scores.append(v_prime/10)

            if v_prime/10 > v:
                best_classifier = classifier
                v = v_prime/10
            #v = max(v,v_prime/10)
        print(all_scores)
        print('\nDone! V = {}\n'.format(v))
        all_vs.append(v)

        f.write('{}:\n\tv = {}\n\tBest Classifier: {}\n'.format(cluster_file.split('.')[0],v,best_classifier))

    max_v = np.argmax(all_vs)
    print("Max v: {}".format(cluster_files[max_v].split('.')[0]))

    f.write('\n\n==========\n\n'.format(cluster_file,v))
    f.write('Best v value for partition {} with {}'.format(cluster_files[max_v].split('.')[0],all_vs[max_v]))
    print('Finished!: Report saved in {}'.format(outfile))



if __name__ == '__main__':
    main()
