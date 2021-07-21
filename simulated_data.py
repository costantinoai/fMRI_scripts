# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 17:38:01 2021

@author: 45027900
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sn

# generate 8 trials, 4 per class (1 class/run), spread out over 4 runs in 1000 voxels
runs_n = 5
labels_n = 4
X_r = []
for run in range(0,runs_n):
    a = np.random.normal(0,0.5,1000)
    b = np.random.normal(1,0.5,1000)
    d = np.random.normal(3,0.5,1000)
    e = np.random.normal(2,0.5,1000)
    X_r.append(a)
    X_r.append(b)
    X_r.append(d)
    X_r.append(e)
    
X_r = np.array(X_r)
y_r = np.tile([0, 1, 3, 2], runs_n)
runs = (np.array([[run+1] * labels_n for run in range(runs_n)])).flatten()

# add random gaussian noise to the data (x:0, std:4)
noise =  np.random.normal(0,5,(len(y_r),1000))
X_r = X_r + noise

# Import from model_selection module. groupkfold with 4 folds (splits) is a leave-one(run)-out
from sklearn.model_selection import GroupKFold

# we initialize GroupKFold with 4 splits -> it is exactly the same as
# the LeaveOneGroupOut cross-validator, since we only have 4 groups
gkf = GroupKFold(n_splits=runs_n)
performance = np.zeros(gkf.n_splits)
cf = []
for i ,(train_idx, test_idx) in enumerate(gkf.split(X=X_r, y=y_r, groups=runs)):
    print("Indices of train-samples: %r" % train_idx.tolist())
    print("Indices of test-samples: %r" % test_idx.tolist())
    print("... which correspond to following runs: %r" % runs[test_idx].tolist())
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    X_train = X_r[train_idx]
    X_test = X_r[test_idx]
    y_train = y_r[train_idx]
    y_test = y_r[test_idx]
    clf = SVC(kernel='linear', C=1e-20).fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    accuracy_training = clf.score(X_train, y_train)
    performance[i] = clf.score(X_test, y_test)
    # # Plot non-normalized confusion matrix
    # titles_options =[("Normalized confusion matrix - Run " + str(i), None)]
    # for title, normalize in titles_options:
    #     disp = plot_confusion_matrix(clf, X_test, y_test,
    #                                   cmap=plt.cm.Blues,
    #                                   normalize=normalize)
    #     disp.ax_.set_title(title)
    #     print(title)
    #     print(disp.confusion_matrix)
    #     cf.append(disp.confusion_matrix)
    # plt.show()
    print(f'Training accuracy: {accuracy_training}')
    print(f'LABELS: {y_test}\nPREDICTED: {y_hat}\nACCURACY: {performance[i]}, STEP: {i+1}\n\n')

# Plot non-normalized confusion matrix
final_cf = np.sum(cf, 0)
cmap = sn.color_palette("YlGnBu", as_cmap=True)
hm = sn.heatmap(final_cf, annot=True, cmap = cmap)
# make frame visible
for _, spine in hm.spines.items():
    spine.set_visible(True)
plt.title('Normalized confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')

acc = np.average(performance)
print(f'TOTAL ACCURACY: {acc}')



