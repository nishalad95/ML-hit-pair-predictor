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
        description="KDE-based classifier training and predictions pipeline for pixel-barrel doublets")
parser.add_argument(
        '--data',
	    '-d',
        help="MC training data file path")
parser.add_argument(
        '--bandwidths',
        '-b',
        help='optimum barrel KDE bandwidths file path')
parser.add_argument(
        '--tripletvalidation',
        '-t',
        default=False,
        help='Execute triplet validation stage')

args = parser.parse_args()
training_data = args.data
bandwidths = args.bandwidths
method = args.method
triplet_validation = args.tripletvalidation


path = str(pathlib.Path().absolute())


# Load the data and form pixel-barrel doublets
print("Processing data & forming doublets...")
td = TrackData(training_data)
td.read_and_merge_data()
training_df = td.calculate_cott()
pix_bar_layers = set([0, 1, 2, 3])
pix_barrel_doublets = td.generate_doublets(training_df, pix_bar_layers, label=True)
print("Number of pixel-barrel doublets: ", str(len(pix_barrel_doublets)))


# generate weta bands
weta04 = td.weta_band(pix_barrel_doublets, 0.0, 0.4, balanced=True)
weta06 = td.weta_band(pix_barrel_doublets, 0.4, 0.6, balanced=True)
weta08 = td.weta_band(pix_barrel_doublets, 0.6, 0.8, balanced=True)
weta10 = td.weta_band(pix_barrel_doublets, 0.8, 1.0, balanced=True)
weta12 = td.weta_band(pix_barrel_doublets, 1.0, 1.2, balanced=True)
weta14 = td.weta_band(pix_barrel_doublets, 1.2, 1.4, balanced=True)
weta16 = td.weta_band(pix_barrel_doublets, 1.4, 1.6, balanced=True)
weta18 = td.weta_band(pix_barrel_doublets, 1.6, 1.8, balanced=True)
weta20 = td.weta_band(pix_barrel_doublets, 1.8, 2.0, balanced=True)
weta22 = td.weta_band(pix_barrel_doublets, 2.0, 2.2, balanced=True)
weta24 = td.weta_band(pix_barrel_doublets, 2.2, 2.4, balanced=True)
weta26 = td.weta_band(pix_barrel_doublets, 2.4, 2.6, balanced=True)
weta28 = td.weta_band(pix_barrel_doublets, 2.6, 2.8, balanced=True)
weta_large = td.weta_band(pix_barrel_doublets, 2.8, 100, balanced=True)


# Downsample high statistics bands:
max_size = 30000
weta04 = td.downsample(weta04, max_size)
weta06 = td.downsample(weta06, max_size)
weta08 = td.downsample(weta08, max_size)
weta10 = td.downsample(weta10, max_size)
weta12 = td.downsample(weta12, max_size)
weta14 = td.downsample(weta14, max_size)
weta16 = td.downsample(weta16, max_size)


# Define variables
weta = [0.0, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]
print("Reading optimum bandwidths...")
optimal_kde_bw = pd.read_csv(bandwidths)
bandwidths = optimal_kde_bw['ss_corr_avg'].tolist()
accept_reject_validation = pd.DataFrame(columns=['weta', 'tau', 'predictions', 'targets'])
thresholds = []

X_data = [weta04.loc[:,['tau']], weta06.loc[:,['tau']], weta08.loc[:,['tau']], 
          weta10.loc[:,['tau']], weta12.loc[:,['tau']], weta14.loc[:,['tau']], 
          weta16.loc[:,['tau']], weta18.loc[:,['tau']], weta20.loc[:,['tau']],
          weta22.loc[:,['tau']], weta24.loc[:,['tau']], weta26.loc[:,['tau']]]

Y_data = [weta04.loc[:,['target']], weta06.loc[:,['target']], weta08.loc[:,['target']],
          weta10.loc[:,['target']], weta12.loc[:,['target']], weta14.loc[:,['target']],
          weta16.loc[:,['target']], weta18.loc[:,['target']], weta20.loc[:,['target']],
          weta22.loc[:,['target']], weta24.loc[:,['target']], weta26.loc[:,['target']]]


# Setup classifiers
kde_04 = KDEClassifier(bandwidth=bandwidths[0])
kde_06 = KDEClassifier(bandwidth=bandwidths[1])
kde_08 = KDEClassifier(bandwidth=bandwidths[2])
kde_10 = KDEClassifier(bandwidth=bandwidths[3])
kde_12 = KDEClassifier(bandwidth=bandwidths[4])
kde_14 = KDEClassifier(bandwidth=bandwidths[5])
kde_16 = KDEClassifier(bandwidth=bandwidths[6])
kde_18 = KDEClassifier(bandwidth=bandwidths[7])
kde_20 = KDEClassifier(bandwidth=bandwidths[8])
kde_22 = KDEClassifier(bandwidth=bandwidths[9])
kde_24 = KDEClassifier(bandwidth=bandwidths[10])
kde_26 = KDEClassifier(bandwidth=bandwidths[11])
clfs = [kde_04, kde_06, kde_08, kde_10, kde_12, kde_14, kde_16, kde_18, kde_20, kde_22, kde_24, kde_26]


# Train and predict each classifier
for i, kde in enumerate(clfs):
    
    w = weta[i+1]
    X = X_data[i]
    y = Y_data[i]
    n = str(weta[i]) + " to " + str(weta[i+1])

    print("\nRunning classifier for weta " + n)
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
    cut = 0.95
    
    threshold = cott_thres.loc[cott_thres.tpr >= cut].iloc[0]['threshold']
    thresholds.append(threshold)
    print('Threshold: %.5f' % (threshold))

    # run kde with new threshold cut
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

print("KDE Classifier training & cross-val predictions complete!\n")


# Save predictions to file
print("Saving predictions to file...\n")
accept_reject_validation.to_csv(path + '/barrel-kde-predictions.csv', index=True)


# Plot acceptance-rejection region & save to file
sns_plot = sns.lmplot('weta', 'tau', data=accept_reject_validation, hue='predictions', fit_reg=False, height=8, aspect=1.5)
ax = plt.gca()
ax.set_title("Predictions: Acceptance-Rejection Region for all pixel-barrel doublets")
sns_plot.savefig(path + "/barrel-kde-predictions.png")


# Evaluate overall performance
print("Overall performance:")
y_pred = accept_reject_validation['predictions'].to_list()
y_true = accept_reject_validation['targets'].to_list()
evaluate_performance(y_true, y_pred, ['0', '1'])



def form_feature_matrix(df):
    features = pd.DataFrame({
            "label": df['label'],
            "inner_doublet": df['inner_doublet'],
            "outer_doublet": df['outer_doublet'],
            "cot(t1)": df['cot(t1)'],
            "|cot(t1)|": df['|cot(t1)|'],
            "cot(t2)": df['cot(t2)'],
            "|cot(t2)|": df['|cot(t2)|'],
            "weta2": df['weta2'],
            })

    inner_doublets = pd.DataFrame({
                "label": features['label'], 
                "cot(t)": features['cot(t1)'],
                "tau": features['|cot(t1)|'],
                "weta": features['weta2'],
                "doublet": 'i',
                "target": features['inner_doublet'],
                 })

    outer_doublets = pd.DataFrame({
                "label": features['label'], 
                "cot(t)": features['cot(t2)'],
                "tau": features['|cot(t2)|'],
                "weta": features['weta2'],
                "doublet": 'o',
                "target": features['outer_doublet'],
                 })

    feature_matrix = pd.concat([inner_doublets, outer_doublets], ignore_index=False)

    return feature_matrix


def binomial_error(k_t, N_t):
    return (1 / N_t) * np.sqrt(k_t * (1 - (k_t / N_t)))


# Execute triplet validation
if triplet_validation:

    print("\nExecuting triplet seed validation...")
    triplet_df = training_df.head(300000)
    all_doublets = form_feature_matrix(triplet_df)


    # weta bands
    weta04_trk = td.weta_band(all_doublets, 0.0, 0.4)
    weta06_trk = td.weta_band(all_doublets, 0.4, 0.6)
    weta08_trk = td.weta_band(all_doublets, 0.6, 0.8)
    weta10_trk = td.weta_band(all_doublets, 0.8, 1.0)
    weta12_trk = td.weta_band(all_doublets, 1.0, 1.2)
    weta14_trk = td.weta_band(all_doublets, 1.2, 1.4)
    weta16_trk = td.weta_band(all_doublets, 1.4, 1.6)
    weta18_trk = td.weta_band(all_doublets, 1.6, 1.8)
    weta20_trk = td.weta_band(all_doublets, 1.8, 2.0)
    weta22_trk = td.weta_band(all_doublets, 2.0, 2.2)
    weta24_trk = td.weta_band(all_doublets, 2.2, 2.4)
    weta26_trk = td.weta_band(all_doublets, 2.4, 2.6)
    weta28_trk = td.weta_band(all_doublets, 2.6, 2.8)
    weta_large_trk = td.weta_band(all_doublets, 2.8, 1000)


    # Initialize variables:
    accept_reject_triplets = pd.DataFrame(columns=['label', 'tau', 'weta', 
                                                    'doublet', 'target', 'predictions'])

    X_trk_data = [weta04_trk, weta06_trk, weta08_trk, weta10_trk, 
                weta12_trk, weta14_trk, weta16_trk, weta18_trk, 
                weta20_trk, weta22_trk, weta24_trk, weta26_trk]

    Y_trk_data = [weta04_trk.loc[:,['target']], weta06_trk.loc[:,['target']], 
                weta08_trk.loc[:,['target']], weta10_trk.loc[:,['target']], 
               weta12_trk.loc[:,['target']], weta14_trk.loc[:,['target']],
               weta16_trk.loc[:,['target']], weta18_trk.loc[:,['target']], 
               weta20_trk.loc[:,['target']], weta22_trk.loc[:,['target']], 
               weta24_trk.loc[:,['target']], weta26_trk.loc[:,['target']]]

    seed_filter_eff = []
    tot_rejection_rate = []
    seed_filtering_error = []
    tot_rejection_error = []

    for i, kde in enumerate(clfs):
        
        w = weta[i+1]
        accept_reject = X_trk_data[i].copy()
        x = X_trk_data[i].loc[:,['tau']]
        y = Y_trk_data[i]
        threshold = thresholds[i]
        
        # make predictions
        print("Performing kde predictions on band: ", str(w), "\nUsing threshold: ", str(threshold))
        y_trk_pred = kde.predict(x, threshold)
        print("Predictions for weta band: " + str(w) + " complete!")
        
        # doublet form: predictions for each constituent doublet
        accept_reject['predictions'] = y_trk_pred
        accept_reject_triplets = pd.concat([accept_reject_triplets, accept_reject], ignore_index=False, sort=False)       

        # calc confusion matrix
        label1 = accept_reject.loc[accept_reject.label == 1]
        y_target = label1.loc[:,['target']]
        y_predicted = label1['predictions'].tolist()
        tn, fp, fn, tp = confusion_matrix(y_target, y_predicted, labels=[0,1]).ravel()
        TPR, TNR = tp / (tp + fn), tn / (tn + fp)
        FPR, FNR = 1 - TNR, 1 - TPR
        print("TPR: %.3f, TNR: %.3f, \nFPR: %.3f, FNR: %.3f" % (TPR, TNR, FPR, FNR))
        
        # triplet form:
        triplets_band = triplet_df.loc[(triplet_df['weta2'] > weta[i]) & (triplet_df['weta2'] <= weta[i+1])]
        suc_triplets_band = triplets_band.loc[triplets_band.label == 1]
        
        # 1. pb middle hits
        pb_middle_track_prop = suc_triplets_band.loc[(suc_triplets_band.isPixel2 == 1) 
                                                    & (suc_triplets_band.layer2 <= 3)]
        # 2. other middle hits - by default we accept these for now
        other_middle_hits_track_prop = suc_triplets_band.loc[(suc_triplets_band.layer2 > 3) 
                                                            & (suc_triplets_band.isPixel2 == 1)]    
        
        # from pb middle hits, accept those triplets with prediction of both doublets == 1
        inner_accepted = accept_reject.loc[(accept_reject.doublet == 'i') & (accept_reject.predictions == 1)]
        outer_accepted = accept_reject.loc[(accept_reject.doublet == 'o') & (accept_reject.predictions == 1)]
        common_idx = inner_accepted.index.intersection(outer_accepted.index).to_list()
        kde_accepted_track_prop = pb_middle_track_prop[pb_middle_track_prop.index.isin(common_idx)]
        
        
        # seed filtering efficiency = TPR = TP / TP + FN
        good_accepted_triplets = len(other_middle_hits_track_prop) + len(kde_accepted_track_prop)  
        all_accepted_triplets = len(suc_triplets_band)
        seed_filtering_efficiency = (good_accepted_triplets / all_accepted_triplets) * 100
        s_error = binomial_error(good_accepted_triplets, all_accepted_triplets) * 100
        # print("Seed filtering efficiency: " + str(seed_filtering_efficiency * 100) + "%")
        print("Seed filtering efficiency: {:.{}f} % with {:.{}f} % error "
            .format(seed_filtering_efficiency, 3, s_error, 3))
        

        # total rejection rate: all_rejected_triplets consist of triplets with doublet predictions not 1 & 1
        all_rejected_triplets = triplets_band.drop(common_idx)
        total_rejection_rate = (len(all_rejected_triplets) / len(triplets_band)) * 100
        t_error = binomial_error(len(all_rejected_triplets), len(triplets_band)) * 100
        # print("Total rejection rate: " + str(total_rejection_rate * 100) + " % \n")
        print("Total rejection rate: {:.{}f} % with {:.{}f} % error \n"
            .format(total_rejection_rate, 3, t_error, 3))
        
        # append to arrays
        seed_filter_eff.append(seed_filtering_efficiency)
        tot_rejection_rate.append(total_rejection_rate)
        seed_filtering_error.append(s_error)
        tot_rejection_error.append(t_error)

    print("Triplets tracking efficiency evaluation complete!")
    
    
    seed_filter_eff = np.array(seed_filter_eff)
    tot_rejection_rate = np.array(tot_rejection_rate)
    seed_filtering_error = np.array(seed_filtering_error)
    tot_rejection_error = np.array(tot_rejection_error)


    # plot and save triplet validation metrics as a function of track params
    plt.figure(figsize=(10,7))
    plt.errorbar(weta[0:12], seed_filter_eff, yerr=seed_filtering_error, fmt='o', label='seed filtering efficiency', c='orange')
    plt.errorbar(weta[0:12], tot_rejection_rate, yerr=tot_rejection_error, fmt='o', label='total rejection rate', c='blue')
    plt.title("Triplet tracking efficiency metrics for KDE Classifier on barrel doublets")
    plt.ylim([0, 100])
    plt.xlabel('weta2')
    plt.ylabel('%')
    plt.legend(loc='best')
    plt.savefig('pixel-barrel-triplet-validation.png')



    # Overall triplet tracking efficiency metrics:
    # triplet form:
    suc_triplets = triplet_df.loc[triplet_df.label == 1]

    # 1. pb middle hits
    pb_middle_track_prop = suc_triplets.loc[(suc_triplets.isPixel2 == 1) 
                                                & (suc_triplets.layer2 <= 3)]
    # 2. other middle hits - by default we accept these for now
    other_middle_hits_track_prop = suc_triplets.loc[(suc_triplets.layer2 > 3) 
                                                        & (suc_triplets.isPixel2 == 1)]    

    # from pb middle hits, accept those triplets with prediction of both doublets == 1
    inner_accepted = accept_reject_triplets.loc[(accept_reject_triplets.doublet == 'i') & (accept_reject_triplets.predictions == 1)]
    outer_accepted = accept_reject_triplets.loc[(accept_reject_triplets.doublet == 'o') & (accept_reject_triplets.predictions == 1)]
    common_idx = inner_accepted.index.intersection(outer_accepted.index).to_list()
    kde_accepted_track_prop = pb_middle_track_prop[pb_middle_track_prop.index.isin(common_idx)]


    # seed filtering efficiency = TPR = TP / TP + FN
    good_accepted_triplets = len(other_middle_hits_track_prop) + len(kde_accepted_track_prop)  
    all_accepted_triplets = len(suc_triplets)
    seed_filtering_efficiency = (good_accepted_triplets / all_accepted_triplets) * 100
    s_error = binomial_error(good_accepted_triplets, all_accepted_triplets) * 100
    #     print("Seed filtering efficiency: " + str(seed_filtering_efficiency * 100) + "%")
    print("Seed filtering efficiency: {:.{}f} % with {:.{}f} % error "
        .format(seed_filtering_efficiency, 3, s_error, 3))


    # total rejection rate: all_rejected_triplets consist of triplets with doublet predictions not 1 & 1
    all_rejected_triplets = triplet_df.drop(common_idx)
    total_rejection_rate = (len(all_rejected_triplets) / len(triplet_df)) * 100
    t_error = binomial_error(len(all_rejected_triplets), len(triplet_df)) * 100
    #     print("Total rejection rate: " + str(total_rejection_rate * 100) + " % \n")
    print("Total rejection rate: {:.{}f} % with {:.{}f} % error \n"
        .format(total_rejection_rate, 3, t_error, 3))

