from scipy import signal, integrate, interpolate
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc



def min_max_normalize(signal_: np.ndarray, min_val: int =0, max_val: int=1) -> np.ndarray:
    """
    Min-max normalizes the signal.
    """
    s_norm = min_val + (signal_ - np.nanmin(signal_)) * (max_val - min_val) / \
        (np.nanmax(signal_) - np.nanmin(signal_))

    return s_norm


def fix_timestamps(ts_old, packet_size):
    """
    Fixes timestamps for a series of timestamps where two or more timestamps 
    are the same.
    """
    # Unique timestamps.
    ts = ts_old[::packet_size]
    # Unique timestamp indices.
    ts_idxs = np.arange(0, ts_old.size, packet_size)
    int_f = interpolate.interp1d(ts_idxs, ts, kind='linear', fill_value='extrapolate')
    idxs = np.arange(0, ts_old.size)
    ts_new = int_f(idxs)
    return ts_new

def compute_fs(timestamps):
    """
    Computes sampling frequency based on timestamps. 
    """
    ts_diff = np.diff(timestamps)
    fs = np.mean(1_000.0 / ts_diff)
    return fs



def roc_curves(y_true, y_pre, labels):
    """Compute and plot the ROC Curves for each class, also macro and micro."""
 
    fpr, tpr, roc_auc = dict(), dict(), dict()
    # AUROC, fpr and tpr for each label
    for i in range(len(labels)):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pre[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pre.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Interpolate all ROC curves at these points to compute macro-average ROC area
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    mean_tpr = np.zeros_like(fpr_grid)
    for i in range(len(labels)):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average the mean TPR and compute AUC
    mean_tpr /= len(labels)
    
    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 5))
    fig.suptitle('ROC Curves')

    # Plotting micro-average and macro-average ROC curves
    ax1.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))

    ax1.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]))

    # Plotting ROCs for each class
    for i in range(len(labels)):
        ax2.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(labels[i], roc_auc[i]))

    # Adding labels and titles for plots
    ax1.plot([0, 1], [0, 1], 'k--'); ax2.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlim([0.0, 1.0]); ax2.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05]); ax2.set_ylim([0.0, 1.05])
    ax1.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax2.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax1.legend(loc="lower right", prop={'size': 8}); ax2.legend(loc="lower right", prop={'size': 6})

