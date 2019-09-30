import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc

import models

from tqdm import tqdm

import os
import json
import click

@click.command()
@click.option('--clusters_path',
              default = './QS_Partitions_CSV',
              help = 'Directory containing the partitions to analyize')
@click.option('--outfile', default = './vic_report.txt', help = 'Name for report file')

def main(clusters_path, outfile):

    # Load partition file names so we can iterate over them and declare variables.

    cluster_files = [f for f in os.listdir(clusters_path) if f.endswith('.csv')]
    cluster_files.sort()
    report_dict = dict([])
    all_vs = []

    # Open Report File
    f = open(outfile,'w')
    f.write('=====VIC Cluster Validation Report=====\n\n')

    # Set the classifiers to use.
    #Must use names according to train_and_test() in models.py
    classifiers = ['random_forest','svm','naive_bayes','lda','gradient_boosting', 'logistic_regression']

    # Write classifiers used in the report
    f.write('Num of Classifiers: {}\n'.format(len(classifiers)))
    f.write('Chosen Classifiers:\n')
    for classifier in classifiers:
        f.write('\t{}\n'.format(classifier))
    f.write('\n\n')

    # Iterate over every partition file
    for cluster_file in cluster_files:
        partition_scores = []
        print('Working on Partition {}'.format(cluster_file.split('.')[0]))

        # Load partition file
        df = pd.read_csv(os.path.join(clusters_path,cluster_file))
        # We added this because the first column indicated the rank, you might
        # need to comment the next line.
        df = df.drop([df.columns[0],df.columns[1]],axis = 1)

        # Load as numpy array
        values = df.values
        y = values[:,-1]
        x = values[:,:-1]

        # Setup 10-fold Cross Validation
        kf = KFold(n_splits = 10, shuffle = True, random_state = 0)
        kf.shuffle
        kf.get_n_splits(x)

        v = 0

        # Iterate over the selected classifiers
        for classifier in tqdm(classifiers):
            v_prime = 0
            # Iterate over each fold
            for train_index, test_index in kf.split(x):

                X_train, X_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Train and test the model with the current folds. Return AUC
                this_auc = models.train_and_test(classifier,X_train,y_train,X_test,y_test)
                v_prime = v_prime + this_auc

            # Keep the score for each classifier for the analysis of results.
            partition_scores.append(v_prime/10)

            # Update v
            if v_prime/10 > v:
                best_classifier = classifier
                v = v_prime/10

            # Save results for json report
            partition_name = cluster_file.split('.')[0]
            report_dict[partition_name] = dict([('best_classifier',0),('scores',0),('v',0)])
            report_dict[partition_name]['best_classifier'] = best_classifier
            report_dict[partition_name]['scores'] = partition_scores
            report_dict[partition_name]['v'] = v

        # Save to json report
        with open('report.json', 'w') as fp:
            json.dump(report_dict, fp)

        print('\nDone! V = {}\n'.format(v))
        all_vs.append(v)

        f.write('{}:\n\tv = {}\n\tBest Classifier: {}\n'.format(cluster_file.split('.')[0],v,best_classifier))

    # Finish and report best partition
    max_v = np.argmax(all_vs)
    print("Max v: {}".format(cluster_files[max_v].split('.')[0]))

    f.write('\n\n==========\n\n'.format(cluster_file,v))
    f.write('Best v value for partition {} with {}'.format(cluster_files[max_v].split('.')[0],all_vs[max_v]))
    print('Finished!: Report saved in {}'.format(outfile))



if __name__ == '__main__':
    main()
