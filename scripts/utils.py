import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, f1_score, confusion_matrix, classification_report


def plot_good_bad_kde_multiple(x_good_all, x_bad_all, bandwidths, weta, bins=100, annotation=False):

    fig = plt.figure(figsize=(25, 15))
    fig.tight_layout()
    ax1 = fig.add_subplot(111, frame_on=False)
    plt.suptitle(f"Gaussian KDE fitted to Pixel-Barrel middle spacepoints nbins: 100, optimum bandwidth found by silverman method")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel('$|cot(θ)|$', labelpad=50)
    ax1.set_ylabel('Probability Density of KDE', labelpad=45)

    for i in range(len(x_good_all)):
        
        x_good = x_good_all[i]
        x_bad = x_bad_all[i]
        b = bandwidths[i]
        weta = weta_bands[i]
        xlims = (-0.25,9)
        kdelims = (0, 10)
        
        ax1 = fig.add_subplot(4, 3, i+1)
        sns.set()

        ax = sns.distplot(x_good, hist=True, kde=True, norm_hist=True, bins=bins, ax=ax1, color='blue',
                     hist_kws={"lw":2},
                     kde_kws={"lw": 2, "clip": kdelims, "bw": b,
                              "gridsize": 5000, "kernel":"gau", "label":"Good doublets"})

        ax.set(xlabel=None)
        # annotate the peak value
        x_data = ax.lines[0].get_xdata()
        y_data = ax.lines[0].get_ydata()
        maxid = np.argmax(y_data)
        plt.plot(x_data[maxid],y_data[maxid], 'bo', ms=10)
        plt.annotate(r"$|cot(θ)|$: " + str('%.5g' % x_data[maxid]), (x_data[maxid] + 0.2, y_data[maxid]))

        ax1.grid(False)
        plt.xlim(xlims)
        plt.title(str(weta[0]) + " < η ≤ " + str(weta[1]) + " b: " + str('%.2g' % b))

        ax = sns.distplot(x_bad, hist=True, kde=True, norm_hist=True, bins=bins, ax=ax1, color='red',
                 hist_kws={"lw":2, "alpha":0.15},
                 kde_kws={"lw": 2, "clip": kdelims, "bw": b, "gridsize": 5000, 
                          "kernel":"gau", "alpha":0.4, "label":"Bad doublets"})
        ax.set(xlabel=None)
 
#     plt.savefig('pix_bar_kdes_good_bad_pair_dist_plts_3.png')
    plt.show()



def plot_confusion_matrix(cm, labels, pos, fmt, normalisation=False, title="Confusion Matrix"):
    
    if normalisation:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    ax= plt.subplot(pos)
    sns.heatmap(cm, annot=True, ax = ax, fmt=fmt, cmap='Greens') #annot=True to annotate cells

    # labels, title and ticks
    ax.set_ylim([0,2])
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels') 
    ax.set_title(title) 
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)


def evaluate_performance(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision =  tp / (tp + fp)
    recall = tp / (tp + fn)
#     print("tn:", tn, "fp:", fp, "fn:", fn, "tp:", tp)
    print("Confusion matrix:")
    TPR, TNR = tp / (tp + fn), tn / (tn + fp)
    FPR, FNR = 1 - TNR, 1 - TPR
    print("TPR: %.3f, TNR: %.3f, \nFPR: %.3f, FNR: %.3f" % (TPR, TNR, FPR, FNR))
    print("Precision: %.3f" % precision)
    print("Recall: %.3f" % recall)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred))
    
    #plt.figure(figsize=(15,5))
    #plot_confusion_matrix(cm, labels, '120', 'g')
    #plot_confusion_matrix(cm, labels, '121', '.2f', normalisation=True, title="Normalised Confusion Matrix")



def plot_ROC_Precision_Recall(ns_fpr, ns_tpr, filename, kde_fpr, kde_tpr, bandwidth, 
                              y_test, kde_recall, kde_precision):    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
    
    # ROC
    axes[0].plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    axes[0].plot(kde_fpr, kde_tpr, marker='.', label='KDE b=' + str(bandwidth[0]))
    axes[0].set_title(r"ROC Curve Plot for |cot(t)|")
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend()
    
    # Preicision-Recall
    no_skill = len(y_test.loc[(y_test['target']==1)]) / len(y_test)
    axes[1].plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    axes[1].plot(kde_recall, kde_precision, marker='.', label='KDE b=' + str(bandwidth[0]))
    axes[1].set_title(r"Precision-Recall plot for |cot(t)|")
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend()
    
    fig.tight_layout()
#     plt.savefig("../plots/11022020-ROC_PR_Plots_all_weta_correct_class/" + filename + ".png")


def generate_ns_roc(y_test):
    ns_probs = [0 for _ in range(len(y_test))]
    ns_auc = roc_auc_score(y_test, ns_probs)
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    return ns_fpr, ns_tpr