# ECG Atrial Fibrillation Detection: SVM vs. CNN vs. SCCNet

Final project for *Introduction to Machine Learning* (Group 9).

We detect atrial fibrillation (AF) from 12-lead ECG recordings in the PTB-XL subset of the
[PhysioNet/CinC Challenge 2020](https://physionet.org/content/challenge-2020/1.0.2/) dataset, and compare
three approaches:

| Model | Input | Accuracy |
|-------|-------|----------|
| SVM (hand-crafted features via NeuroKit2) | ECG features | 0.9715 |
| 1D CNN | raw ECG `(12, 1000)` | 0.9832 |
| SCCNet (spatial-conv network adapted from EEG) | raw ECG `(12, 1000)` | 0.9510 |

AF cases are only about 7% of the data, so training uses class weighting and we also look at F1, AUC and
confusion matrices. See the [report](report/group9_final_report.pdf) for details.

## Repository layout

```
notebooks/
  ECG_preprocessing.ipynb       # PTB-XL loading, AF labels, filtering, resampling, normalization -> .npy (for CNN / SCCNet)
  ECG_SVM_preprocessing.ipynb   # same pipeline + NeuroKit2 feature extraction for the SVM
  ECG_SCCNet.ipynb              # SCCNet model, training loop, learning curves, confusion matrix
report/
  group9_final_report.pdf       # final report
docs/
  Final_Project_Guidelines.pdf  # course project guidelines
```

## Pipeline

1. **Labels**: AF = SNOMED CT codes `164889003` (atrial fibrillation) or `164890007` (atrial flutter); everything else is non-AF.
2. **Preprocessing**: 0.5-40 Hz band-pass filter, resample to 100 Hz, fix to 10 s (1000 samples; truncate or zero-pad), per-record channel-wise z-score.
3. **Split**: 80/20 stratified train/validation split (`random_state=42`).
4. **Training** (SCCNet): cross-entropy with balanced class weights, Adam, lr 1e-3, batch size 64, 30 epochs.

## Running

The notebooks were written for **Google Colab** (GPU runtime for SCCNet) and read/write from Google Drive:

- Paths such as `/content/drive/MyDrive/Colab Notebooks/ML_final_project` are hard-coded in the first cell of each notebook. Edit `PROJECT_ROOT` to match your own Drive layout.
- Download the PTB-XL part of the challenge data (`training/ptb-xl`) from PhysioNet, using the `wget` line in the notebooks.
- Run `ECG_preprocessing.ipynb` (or `ECG_SVM_preprocessing.ipynb`) first, then `ECG_SCCNet.ipynb`.

Dependencies: `numpy`, `scipy`, `scikit-learn`, `wfdb`, `neurokit2`, `pandas`, `tqdm`, `torch`, `matplotlib` (see [requirements.txt](requirements.txt)).

## Data and large files

Raw data, preprocessed arrays and trained models are not stored in this repo because of their size. They are in the
team's Google Drive folder: <https://drive.google.com/drive/folders/1kWKXySBY4qnA8gK7CoUhgI9bx9PeHQ9t>
(`data/`, `models/`, `raw_challenge2020/`).

The 1D CNN notebook is not part of this repo (it was not in the Drive folder).
