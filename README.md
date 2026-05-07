# Analytics for Health Wearables

A series of Jupyter notebooks covering biosignal processing from health wearables. The exercises progress from basic signal handling (loading, filtering, resampling) to advanced topics (peak detection, HRV, BCG-based heart rate, and activity recognition).

---

## Structure

```
analytics-for-health-wearables/
├── ex1_analytics_for_health_wearables/   Signal loading, visualization, resampling, filtering
├── ex2_analytics_for_health_wearables/   R-peak detection, HRV, Poincaré plots, PTT
├── ex3_analytics_for_health_wearables/   Segmentation, ensemble averaging, AFib vs sinus
├── ex4_analytics_for_health_wearables/   BCG HR detection via CWT + autocorrelation
├── ex5_analytics_for_health_werables/    Activity recognition with XGBoost
└── Mean_Absolute_Error/                  HR extraction benchmark (MAE evaluation)
```

---

## Exercises

### Exercise 1 — Signal Loading, Visualization & Resampling
**Notebook:** `ex1_analytics_for_health_wearables/exercise1.ipynb`  
**Data:** `ecg_ppg_clean.csv`, `ecg_ppg_motion.csv`

Two simultaneous recordings: one without motion artifacts (clean) and one with motion. Each file contains one ECG channel and two PPG sensors (green, red, IR wavelengths each), recorded at 128 Hz (ECG) and 100 Hz (PPG).

**What is done:**
- Load both CSV files and inspect NaN distribution (PPG has ~3455 missing rows, ECG has none)
- Drop NaN rows, then plot all 7 signal channels for both conditions
- Select ECG and PPG-1-green for resampling; normalize amplitude and convert timestamps
- Resample both signals to 200 Hz using cubic interpolation (`scipy.interpolate.interp1d`)
- Visualize signals in the frequency domain using Welch power spectral density
- Apply a 4th-order Butterworth bandpass filter (ECG: 0.5–40 Hz, PPG: 0.5–10 Hz) and a simple moving average filter

---

### Exercise 2 — R-Peak Detection & Heart Rate Variability
**Notebook:** `ex2_analytics_for_health_wearables/ex2.ipynb`  
**Data:** `ecg_ppg_clean.csv`, `ecg_ppg_motion.csv`

**What is done:**
- Normalize and resample ECG and PPG from both clean and motion recordings
- Apply a Butterworth bandpass filter for baseline and high-frequency noise removal
- Implement a derivative filter for T-wave suppression (Pan-Tompkins style)
- Detect R-peaks via squaring + moving window integration (MWI)
- Compute RR intervals and heart rate; remove physiologically impossible beats (< 40 or > 180 BPM) using median filtering
- Use `heartpy` as an alternative peak detection reference
- Compute RR interval differences for HRV, then plot Poincaré and Lorenz plots
- Estimate Pulse Transit Time (PTT) between ECG R-peaks and PPG systolic peaks

---

### Exercise 3 — Segmentation, Ensemble Averaging & AFib Detection
**Notebook:** `ex3_analytics_for_health_wearables/ex3.ipynb`  
**Data:** `ecg_ppg_clean.csv`, `ecg_ppg_motion.csv`, `ecg_afib.csv`, `ecg_sinus.csv`, `ppg_afib.csv`, `ppg_sinus.csv`

**What is done:**
- Segment the Butterworth-filtered ECG into 10s non-overlapping windows (overlap is configurable)
- Detect R-peaks and extract individual beat waveforms (±300ms / +350ms around each peak)
- Compute ensemble-averaged (sample-wise mean) ECG beat and assess quality via Pearson correlation
- Detect the T-wave from the ensemble ECG waveform (peak search in 0.1–0.4s window after R)
- Repeat for PPG: ensemble-average PPG beats, detect the diastolic notch (0.2–0.5s after systolic peak)
- Compare Atrial Fibrillation vs. Sinus Rhythm using:
  - HRV metrics: Mean RR, SDNN, RMSSD
  - RR interval histograms
  - RR interval autocorrelation (AFib decays rapidly; Sinus shows periodicity)

---

### Exercise 4 — BCG Heart Rate via CWT and Autocorrelation
**Notebook:** `ex4_analytics_for_health_wearables/EX4_ASSIGNMENT_continuous_wavelet_transform_based_autocorrelation_BCG_HR_detection.ipynb`  
**Data:** `bcg_exercise_data/` — 7 subjects (sub1, sub3, sub14, sub19, sub23, sub26, sub33)

Ballistocardiography (BCG) measures micro-vibrations caused by heartbeats through sensors placed under a mattress. Each subject CSV contains ECG (ground truth) and 8 BCG channels: Film0–3 (piezoelectric film sensors) and LC_BCG0–3 (load cell sensors).

**What is done:**
- Downsample all signals from 1000 Hz to 200 Hz using a 100 Hz anti-aliasing lowpass filter + decimation by 5
- Clean ECG using NeuroKit2 and detect R-peaks to compute ECG heart rate per 4s window
- For each BCG channel, extract 4s windows (plus 1s padding), apply Continuous Wavelet Transform (CWT) using Morlet wavelets in the 5–35 Hz band
- Average CWT coefficients across scales, normalize, compute gradient, apply rolling median smoothing
- Compute autocorrelation of the processed signal; extract HR estimate from autocorrelation peak spacing
- For each subject, compute Median Absolute Error (MAE) between BCG HR and ECG HR for all 8 channels; report the best-performing channel per subject

---

### Exercise 5 — Activity Recognition with XGBoost
**Notebook:** `ex5_analytics_for_health_werables/exercise5/activity_recognition_exercise.ipynb`  
**Data:** Accelerometer recordings from wrist and thigh sensors (2 subjects, multiple sessions, 8 activities: lying, sitting, standing, walking, jogging, stairsDown, stairsUp, cycling)

**What is done:**
- Load raw accelerometer data (CSV files with timestamp + x/y/z per sensor)
- Estimate original sampling frequency, lowpass filter (20 Hz cutoff) and resample to 50 Hz
- Segment each signal into 1-second (50-sample) windows; compute total acceleration per sensor per window
- Extract time- and frequency-domain features from each segment using `get_activity_features()`
- Train an XGBoost classifier; evaluate using Leave-One-Subject-Out cross-validation
  - Cross-validation accuracy: ~83%
  - Test set accuracy (new subjects): ~78%
- Plot confusion matrix and ROC curves

---

### Mean Absolute Error — HR Extraction Benchmark
**Notebook:** `Mean_Absolute_Error/mean_absolute_error.ipynb`  
**Data:** `data/ecg_data_with_hr_labels.pkl` — 200 ECG signals with ground-truth HR labels

**What is done:**
- Load 200 ECG signals from a pickle file
- Apply a 6th-order Butterworth lowpass filter (45 Hz cutoff) to each signal
- Detect R-peaks using `scipy.signal.find_peaks` with an amplitude threshold of 40% of the signal maximum
- Compute RR intervals and estimate average HR per signal
- Evaluate against ground truth using Mean Absolute Error

**Result:** MAE ≈ 8.43 BPM

---

## Key concepts covered

| Topic | Exercises |
|---|---|
| Signal loading & NaN handling | Ex1 |
| Normalization & timestamp conversion | Ex1, Ex2, Ex3 |
| Resampling (cubic interpolation) | Ex1, Ex2, Ex3, Ex5 |
| Butterworth bandpass filtering | Ex1, Ex2, Ex3, Ex4, Ex5 |
| Power Spectral Density (Welch) | Ex1 |
| R-peak / beat detection | Ex2, Ex3, Ex4, MAE |
| Heart rate & RR intervals | Ex2, Ex3, Ex4, MAE |
| HRV (SDNN, RMSSD, Poincaré) | Ex2, Ex3 |
| Ensemble averaging | Ex3 |
| AFib vs Sinus rhythm | Ex3 |
| Continuous Wavelet Transform | Ex4 |
| Autocorrelation-based HR | Ex4 |
| Activity classification (XGBoost) | Ex5 |
| Cross-validation (LOSO) | Ex5 |
| MAE evaluation | Ex4, MAE |
