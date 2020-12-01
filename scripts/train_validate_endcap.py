import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pathlib
import tarfile
from tools.TrackData import TrackData
from tools.Classifier import KDEClassifier
import seaborn as sns; sns.set(style="white", color_codes=True)
from scipy import stats
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, f1_score, confusion_matrix, classification_report
from utils import plot_confusion_matrix, evaluate_performance
from argparse import ArgumentParser


parser = ArgumentParser(
        description="KDE-based classifier training and predictions pipeline for pixel-endcap doublets")
parser.add_argument(
        '--data',
	    '-d',
        help="MC training data file path")
parser.add_argument(
        '--bandwidths',
        '-b',
        help='optimum barrel KDE bandwidths file path')
# parser.add_argument(
#         '--tripletvalidation',
#         '-t',
#         default=False,
#         help='Execute triplet validation stage')


args = parser.parse_args()
training_data = args.data
bandwidths = args.bandwidths
method = args.method
# triplet_validation = args.tripletvalidation


path = str(pathlib.Path().absolute())


# Load the data and form pixel-barrel doublets
print("Processing data & forming doublets...")
td = TrackData(training_data)
td.read_and_merge_data()
training_df = td.calculate_cott()
pix_bar_layers = set([0, 1, 2, 3])
pix_ec_layers = set([8, 9, 10, 20, 21, 22])
pixel_endcap_doublets = td.generate_endcap_doublets(training_df, pix_bar_layers, pix_ec_layers)
print("Number of pixel-endcap doublets: ", str(len(pixel_endcap_doublets)))


# generate weta bands
weta04 = td.weta_band(pixel_endcap_doublets, 0.0, 0.4)
weta06 = td.weta_band(pixel_endcap_doublets, 0.4, 0.6)
weta08 = td.weta_band(pixel_endcap_doublets, 0.6, 0.8)
weta09 = td.weta_band(pixel_endcap_doublets, 0.8, 0.9)
weta10 = td.weta_band(pixel_endcap_doublets, 0.9, 1.0)
weta12 = td.weta_band(pixel_endcap_doublets, 1.0, 1.2)
weta13 = td.weta_band(pixel_endcap_doublets, 1.2, 1.3)
weta14 = td.weta_band(pixel_endcap_doublets, 1.3, 1.4)
weta15 = td.weta_band(pixel_endcap_doublets, 1.4, 1.5)
weta16 = td.weta_band(pixel_endcap_doublets, 1.5, 1.6)
weta18 = td.weta_band(pixel_endcap_doublets, 1.6, 1.8)
weta20 = td.weta_band(pixel_endcap_doublets, 1.8, 2.0)
weta22 = td.weta_band(pixel_endcap_doublets, 2.0, 2.2)
weta_large = td.weta_band(pixel_endcap_doublets, 2.0, 10)


# Define variables
weta = [0.0, 0.4, 0.6, 0.8, 1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]
print("Reading optimum bandwidths...")
optimal_kde_bw = pd.read_csv(bandwidths)
bandwidths = optimal_kde_bw['ss_corr_avg'].tolist()
accept_reject_validation = pd.DataFrame(columns=['weta', 'tau', 'predictions', 'targets'])
thresholds = []

X_data = [weta04.loc[:,['tau']], weta06.loc[:,['tau']], weta08.loc[:,['tau']], 
          weta10.loc[:,['tau']], weta12.loc[:,['tau']], weta13.loc[:,['tau']],
         weta14.loc[:,['tau']], weta15.loc[:,['tau']], weta16.loc[:,['tau']],
         weta18.loc[:,['tau']], weta20.loc[:,['tau']]]

Y_data = [weta04.loc[:,['target']], weta06.loc[:,['target']], weta08.loc[:,['target']], 
          weta10.loc[:,['target']], weta12.loc[:,['target']], weta13.loc[:,['target']],
         weta14.loc[:,['target']], weta15.loc[:,['target']], weta16.loc[:,['target']],
         weta18.loc[:,['target']], weta20.loc[:,['target']]]


# Setup classifiers
kde_04 = KDEClassifier(bandwidth=bandwidths[0])
kde_06 = KDEClassifier(bandwidth=bandwidths[1])
kde_08 = KDEClassifier(bandwidth=bandwidths[2])
kde_10 = KDEClassifier(bandwidth=bandwidths[3])
kde_12 = KDEClassifier(bandwidth=bandwidths[4])
kde_13 = KDEClassifier(bandwidth=bandwidths[5])
kde_14 = KDEClassifier(bandwidth=bandwidths[6])
kde_15 = KDEClassifier(bandwidth=bandwidths[7])
kde_16 = KDEClassifier(bandwidth=bandwidths[8])
kde_18 = KDEClassifier(bandwidth=bandwidths[9])
kde_20 = KDEClassifier(bandwidth=bandwidths[10])

clfs = [kde_04, kde_06, kde_08, kde_10, kde_12, kde_13, kde_14, kde_15, kde_16, kde_18, kde_20]


# Train and predict each classifier
for i, kde in enumerate(clfs):

    w = weta[i+1]
    X = X_data[i]
    y = Y_data[i]
    n = str(weta[i]) + " to " + str(weta[i+1])

    print("\nRunning classifier for " + n)
    print("bandwidth: ", kde.bandwidth)
    
    # train the classifier
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    kde.fit(X_train, y_train)
    y_pred = kde.predict(X_val, 0.5)

    # calc confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred.tolist(), labels=[0,1]).ravel()
    TPR, TNR = tp / (tp + fn), tn / (tn + fp)
    FPR, FNR = 1 - TNR, 1 - TPR
    print("TPR: %.3f, TNR: %.3f, \nFPR: %.3f, FNR: %.3f" % (TPR, TNR, FPR, FNR))

    # calc roc score
    kde_probs = kde.predicted_proba_[:, 1]
    kde_roc_auc = roc_auc_score(y_val, kde_probs)
    kde_fpr, kde_tpr, kde_threshold = roc_curve(y_val, kde_probs)
    print('KDE: ROC AUC=%.3f' % (kde_roc_auc))

    # find threshold for TPR 95% & generate new predictions
    cott_thres = pd.DataFrame(data={'fpr' : kde_fpr, 'tpr' : kde_tpr, 'threshold' : kde_threshold})
    threshold = cott_thres.loc[cott_thres.tpr >= 0.95].iloc[0]['threshold']
    thresholds.append(threshold)
    print('Threshold: %.5f' % (threshold))

#     # run kde with new threshold cut
    print("Re-run kde clf with new threshold")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    kde.fit(X_train, y_train)
    y_pred = kde.predict(X_val, threshold)
    y_pred_after_cut = y_pred
    
    # calc confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred_after_cut.tolist()).ravel()
    TPR, TNR = tp / (tp + fn), tn / (tn + fp)
    FPR, FNR = 1 - TNR, 1 - TPR
    print("TPR: %.3f, TNR: %.3f, \nFPR: %.3f, FNR: %.3f" % (TPR, TNR, FPR, FNR))

    # append results to df
    kde_band_pred = pd.DataFrame({'weta' : w, 'tau' : X_val['tau'], 
                         'predictions' : y_pred_after_cut, 'targets' : y_val['target']})
    accept_reject_validation = pd.concat([accept_reject_validation, kde_band_pred], ignore_index=False)

print("KDE Classifier training & cross-val predictions complete!")


# Save predictions to file
print("Saving predictions to file...\n")
accept_reject_validation.to_csv(path + '/endcap-kde-predictions.csv', index=True)


# Plot acceptance-rejection region & save to file
sns_plot = sns.lmplot('weta', 'tau', data=accept_reject_validation, hue='predictions', fit_reg=False, height=8, aspect=1.5)
ax = plt.gca()
ax.set_title("Predictions: Acceptance-Rejection Region for all pixel-endcap doublets")
sns_plot.savefig(path + "/endcap-kde-predictions.png")


# Evaluate overall performance
print("Overall performance:")
y_pred = accept_reject_validation['predictions'].to_list()
y_true = accept_reject_validation['targets'].to_list()
evaluate_performance(y_true, y_pred, ['0', '1'])
