## Target transformation by using CEEMDAN - PE
### CEEMDAN - Complete Ensemble Empirical Mode Decomposition With Adaptive Noise
* CEEMDAN is a variant of the ensemble empirical mode decomposition (EEMD) algorithm, which provides an accurate reconstruction of the original signal and achieves better mode spectrum separation at a lower computational cost.
### PE - Permutation Entropy
* PE algorithm is a dynamic mutation detection method that can easily and accurately locate the time when the mutation occurs and amplifies the small changes of the signal. 
### A. Generating Process
* Step 1: Obtaining multiple IMFs by CEEMDAN decomposition
* Step 2: Calculate PE according to different IMFs
* Step 3: IMFs with PE differences of 0.1 or less are summed and combined to form multiple (possibly one) new time series
* Step 4: Do one step prediction (horizon = 1) by using different models

### B. Fitting models
* ARIMA
* XGBoost

### C. Evaluation Criteria
* RMSE
* SMAPE