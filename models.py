from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_curve, auc

#import tensorflow as tf

def train_and_test(classifier, X_train,y_train,X_test,y_test):

    if classifier == 'random_forest':
        clf = RandomForestClassifier(n_estimators = 100,max_depth = 10,random_state = 0)
    elif classifier == 'svm':
        clf = SVC(gamma = 'scale')
    elif classifier == 'naive_bayes':
        clf = GaussianNB()
    elif classifier == 'lda':
        clf = LinearDiscriminantAnalysis(solver = 'lsqr')
    elif classifier == 'gradient_boosting':
        clf = GradientBoostingClassifier()
    elif classifier == 'mlp':
        clf = MLP()
    else:
        print("I can\'t use classifier {}".format(classifier))
        exit()

    clf.fit(X_train,y_train)
    y_hat = clf.predict(X_test)
    fpr,tpr,thr = roc_curve(y_test,y_hat)
    this_auc = auc(fpr,tpr)

    return this_auc

# class MLP:
#     def __init__(self,hidden_units = 64,hidden_layers = 2,num_classes = 2,input_size = 758):
#         self.hidden_units = hidden_units
#         self.hidden_layers = hidden_layers
#         self.num_classes = num_classes
#         self.input_size = input_size
#
#         self.X = tf.placeholder(tf.float32,shape = [None,self.input_size])
#         self.y = tf.placeholder(tf.float32,shape = [None,])
#
#         self.y = tf.cast(self.y, tf.int32)
#         y_oh = tf.one_hot(self.y, self.num_classes)
#
#         A = tf.keras.layers.Dense(self.hidden_units,activation = 'relu')(self.X)
#         A = tf.keras.layers.Dropout(0.5)(A)
#
#         if self.hidden_layers > 1:
#             for i in range(self.hidden_layers-1):
#                 A = tf.keras.layers.Dense(self.hidden_units,activation='relu')(A)
#                 A = tf.keras.layers.Dropout(0.5)(A)
#
#         logits = tf.keras.layers.Dense(self.num_classes)(A)
#         y_scores = tf.keras.layers.Activation('softmax')(logits)
#         self.y_hat = tf.argmax(y_scores,axis = 1)
#
#         loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_oh, logits = logits))
#         acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_scores,axis = 1),tf.argmax(y_oh,axis = 1)),dtype = tf.float32))
#         optimizer = tf.train.AdamOptimizer()
#         self.train_st = optimizer.minimize(loss)
#
#         self.sess = tf.Session()
#
#     def fit(self,X_train,y_train):\
#
#         self.sess.run(tf.global_variables_initializer())
#
#         for i in range(100):
#             self.sess.run(self.train_st,feed_dict = {self.X:X_train,self.y:y_train})
#
#     def predict(self,X_test):
#         y_hat_this = self.sess.run(self.y_hat,feed_dict = {self.X: X_test})
#         return y_hat_this
