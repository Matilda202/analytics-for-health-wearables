import numpy as np
from signal_processing_functions import min_max_normalize
import scipy
from math import log, e

def get_activity_features(data, size, fs):
    """Features from Logacjov et al. "HARTH: A Human Activity Recognition Dataset for Machine Learning", 2021"""
    features = []
    #Calculate gravity and movement component of the accelerometer signal
    gravity = np.zeros((len(data), size))
    movement = np.zeros((len(data), size))
    for i in range(len(gravity)):
        sos = scipy.signal.butter(4, [1], btype='lowpass', fs=fs, output='sos')
        gravity[i,:] = scipy.signal.sosfiltfilt(sos, data[i,:])
        movement[i,:] = data[i,:] - gravity[i,:]
    
    #Statistical features from gravity component: mean, median, std, coefficient of variation, 25th and 75th percentile, min and max 
    features.append(np.mean(gravity, axis=1))
    features.append(np.median(gravity, axis=1))
    features.append(np.std(gravity, axis=1))
    features.append(np.std(gravity, axis=1, ddof=1)/np.mean(gravity, axis=1))
    features.append(np.percentile(gravity, 25, axis=1))
    features.append(np.percentile(gravity, 75, axis=1))
    features.append(np.min(gravity, axis=1))
    features.append(np.max(gravity, axis=1))
    
    #Statistical features from movement component: skewness, kutrosis, 
    features.append(scipy.stats.skew(movement, axis=1))
    features.append(scipy.stats.kurtosis(movement, axis=1))
    #Signal energy
    features.append(np.sum(np.square(movement), axis=1))
    #Frequency-domain magnitudes’ mean and std
    
    features.append(np.mean(np.abs(np.fft.rfft(movement, axis=1)), axis=1))
    features.append(np.std(np.abs(np.fft.rfft(movement, axis=1)), axis=1))
    
    #Dominant frequency and it's magnitude
    features.append(np.fft.rfftfreq(movement.shape[1])[np.abs(np.fft.rfft(movement, axis=1)).argmax(axis=1)])
    features.append(np.abs(np.fft.rfft(movement, axis=1)).max(axis=1))
    
    #Spectral centroid
    sums = np.sum(np.abs(np.fft.rfft(movement, axis=1)), axis=1)
    sums = np.where(sums, sums, 1.)  # Avoid dividing by zero
    features.append(np.sum(np.fft.rfftfreq(movement.shape[1]) * np.abs(np.fft.rfft(movement, axis=1)), axis=1) / sums)
    #Total signal power (sum of power amplitudes)
    features.append(np.sum(np.abs(np.fft.rfft(movement, axis=1))**2, axis=1))
   
    #Number of zero crossings and turning points from gravity and movement components
    filt_m = scipy.signal.savgol_filter(np.copy(movement), 5, 2, deriv=1)
    filt_g = scipy.signal.savgol_filter(np.copy(gravity), 5, 2, deriv=1)
    features.append((np.diff(np.sign(filt_m), axis=1) != 0).sum(axis=1))
    features.append((np.diff(np.sign(filt_g), axis=1) != 0).sum(axis=1))
    features.append((np.diff(np.sign(movement), axis=1) != 0).sum(axis=1))

    #First and second degree polynomial fit of pitch, roll and theta of wrist and thigh sensor
    pitch_t = np.arctan((data[0,:]/(np.square(data[1,:])+np.square(data[2,:]))))
    roll_t = np.arctan((data[1,:]/(np.square(data[0,:])+np.square(data[2,:]))))
    theta_t = np.arctan(((np.square(data[1,:])+np.square(data[0,:]))/data[2,:]))
    pitch_w = np.arctan((data[3,:]/(np.square(data[4,:])+np.square(data[5,:]))))
    roll_w = np.arctan((data[4,:]/(np.square(data[3,:])+np.square(data[2,:]))))
    theta_w = np.arctan(((np.square(data[3,:])+np.square(data[4,:]))/data[5,:]))
    p_t1 = np.ravel(np.polyfit(np.arange(len(pitch_t)), pitch_t, 1))
    p_t2 = np.ravel(np.polyfit(np.arange(len(pitch_t)), pitch_t, 2))
    r_t1 = np.ravel(np.polyfit(np.arange(len(roll_t)), roll_t, 1))
    r_t2 = np.ravel(np.polyfit(np.arange(len(roll_t)), roll_t, 2))
    t_t1 = np.ravel(np.polyfit(np.arange(len(theta_t)), theta_t, 1))
    t_t2 = np.ravel(np.polyfit(np.arange(len(theta_t)), theta_t, 2))
    p_w1 = np.ravel(np.polyfit(np.arange(len(pitch_w)), pitch_w, 1))
    p_w2 = np.ravel(np.polyfit(np.arange(len(pitch_w)), pitch_w, 2))
    r_w1 = np.ravel(np.polyfit(np.arange(len(roll_w)), roll_w, 1))
    r_w2 = np.ravel(np.polyfit(np.arange(len(roll_w)), roll_w, 2))
    t_w1 = np.ravel(np.polyfit(np.arange(len(theta_w)), theta_w, 1))
    t_w2 = np.ravel(np.polyfit(np.arange(len(theta_w)), theta_w, 2))

    #Axis correlation
    corr = axes_corr(data)
    features.append(corr[0:8])
    features.append(corr[8:])
    
  
    for i, feat in enumerate(features):
        features[i] = min_max_normalize(feat)

    features = np.ravel(features)
    feats = np.concatenate((p_t1, p_t2, r_t1, r_t2, t_t1, t_t2, p_w1, p_w2, r_w1, r_w2, t_w1, t_w2))
    feats = min_max_normalize(feats)
    features = np.concatenate((features, feats))
    
    features = np.nan_to_num(features)


    return np.ravel(features)

def axes_corr(data):
    """
    Compute correlation between all axes (and the magnitudes).
    """
    corr = []
    for i in range(6):
        for j in range(6):
            if j <= i: continue
            x = data[:, i]
            y = data[:, j]
            corr.append((np.mean((x * y)) - np.mean(x) * np.mean(y)) /
                                                    (np.std(x) * np.std(y)))
    # correlation between the magnitudes:
    x = data[:, -2]
    y = data[:, -1]
    corr.append((np.mean((x * y)) - np.mean(x) * np.mean(y)) /
                                            (np.std(x) * np.std(y)))
   
    corr = np.array(corr)

    return corr

