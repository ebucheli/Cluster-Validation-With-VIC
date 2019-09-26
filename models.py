from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_curve, auc


def train_and_test(classifier, X_train,y_train,X_test,y_test):

    if classifier == 'random_forest':
        clf = RandomForestClassifier(n_estimators = 100,max_depth = 10,random_state = 0)
    elif classifier == 'svm':
        clf = SVC()
    elif classifier == 'naive_bayes':
        clf = GaussianNB()
    elif classifier == 'lda':
        clf = LinearDiscriminantAnalysis(solver = 'lsqr')
    elif classifier == 'gradient_boosting':
        clf = GradientBoostingClassifier()
    else:
        print("I can\'t use classifier {}".format(classifier))
        exit()

    clf.fit(X_train,y_train)
    y_hat = clf.predict(X_test)
    fpr,tpr,thr = roc_curve(y_test,y_hat)
    this_auc = auc(fpr,tpr)

    return this_auc
