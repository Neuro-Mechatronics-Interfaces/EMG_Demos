import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from scipy.signal import butter, filtfilt, iirnotch, lfilter_zi, lfilter
import warnings
from sklearn.decomposition import PCA
from Filter_data import SignalProcessor


def notch_filter(data, f0=60.0, Q=30, fs=500):
    b, a = iirnotch(f0, Q, fs)
    zi = lfilter_zi(b, a)
    return filtfilt(b, a, data)

def bandpass_filter(data, lowcut, highcut, fs=500, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def moving_window_rms(signal, window_size):
    return np.sqrt(np.convolve(signal ** 2, np.ones(window_size) / window_size, mode='same'))

def rest_baseline(df):
    means = df[df['stimulus_state'] == 'rest'].iloc[:, :8].mean()
    stds = df[df['stimulus_state'] == 'rest'].iloc[:, :8].std()
    return means, stds

def process_gesture_data(path, gesture_name, fs=500, num_channels=8, window_size=200, plot=True):
    df = pd.read_csv(path)
    smoothed_signals = []
    pre_rect_signals = [] 
    rectified_signals = [] 
    processor = SignalProcessor(sensor=None, fs=500)

    for i in range(num_channels):
        raw = df.iloc[:, i].values
        # band = bandpass_filter(raw, 8, 248, fs)
        # notch = notch_filter(band, 60, fs) 
        data_preproc = processor.preprocess_filters(raw)
        pre_rect_signals.append(data_preproc[:])

        # --- Create the rectified signal  ---
        rectified = processor.rectify_emg(data_preproc[:])
        rectified_signals.append(rectified)

        # RMS is calculated on the rectified signal
        rms = processor.moving_window_rms(rectified, window_size) 
        smoothed_signals.append(rms)

    filt_df = pd.DataFrame(smoothed_signals).T
    filt_df.columns = [f"Channel {i+1}" for i in range(num_channels)]
    filt_df['stimulus_state'] = df['stimulus_state'].iloc[:].values
    filt_df['gesture'] = gesture_name

    if plot:
        fig, axes = plt.subplots(num_channels, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(f"Filtered EMG Data - {gesture_name}", fontsize=16)
        for i in range(num_channels):
            axes[i].plot(filt_df.iloc[:, i], linewidth=0.6)
            axes[i].set_title(f'Channel {i + 1}')
            axes[i].grid(True)
        axes[-1].set_xlabel("Time Points")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    return filt_df, pd.DataFrame(pre_rect_signals).T, pd.DataFrame(rectified_signals).T

def adjust_labels(df, gesture):
    mean, std = rest_baseline(df)
    high = (df.iloc[:, :8] > mean + std).all(axis=1)
    low = (df.iloc[:, :8] < mean + std).all(axis=1)
    df.loc[high, 'stimulus_state'] = gesture
    df.loc[low, 'stimulus_state'] = 'rest'
    df.loc[df['stimulus_state'].isin(['up', 'down', 'hold']), 'stimulus_state'] = gesture
    return df.iloc[:, :8].assign(stimulus_state=df['stimulus_state'])

def sliding_window(df, window_size=200, stride=20):
    signal = df.iloc[:, :-1].to_numpy()
    labels = df.iloc[:, -1].to_numpy()
    X, y = [], []
    for i in range(0, len(signal) - window_size + 1, stride):
        win = signal[i:i+window_size]
        label = Counter(labels[i:i+window_size]).most_common(1)[0][0]
        X.append(win)
        y.append(label)
    return np.array(X), np.array(y)

#-----------------------
def slide_filt(df, window_size=200, stride=100):
    signal = df.iloc[:, :].to_numpy()
    X= []

    for i in range(0, len(signal) - window_size + 1, stride):
        win = signal[i:i+window_size]
        X.append(win)

    return np.array(X)
#-----------------------


def compute_rms(windows):
    return np.sqrt(np.mean(np.square(windows), axis=1))



#-----------------------
def mean_absolute_value(data):
    return np.mean(np.abs(data), axis=1)

def count_zero_crossings_diff(data):
    diff_signal = np.diff(np.sign(data), axis=1)
    return np.sum(np.abs(diff_signal), axis = 1) / 2
#-----------------------

class LDA_Trainer_Multi:
    def __init__(self, gesture_filepaths, file_identifier="unknown", file_suffix=0, classifier_type="MLP_Classifier"):
        self.gesture_filepaths = gesture_filepaths
        self.file_identifier = file_identifier
        self.file_suffix = file_suffix
        self.classifier_type = classifier_type
        self.mlp_model = None
        self.model_save_path = None
        self.pca_model = None  # Add placeholder for PCA model
        self.pca_save_path = None # Add placeholder for PCA model path

    def train_model(self):
        warnings.filterwarnings("ignore", category=FutureWarning)
        all_smoothed_data = []
        all_pre_rect_data = [] 
        all_rectified_data = [] 

        for gesture, path in self.gesture_filepaths.items():
            print(f"Processing: {gesture} from {os.path.basename(path)}")
            smoothed_df, pre_rect_df, rect_df = process_gesture_data(path, gesture, fs=500, plot=True)
            
            smoothed_df = adjust_labels(smoothed_df, gesture)
            all_smoothed_data.append(smoothed_df)
            all_pre_rect_data.append(pre_rect_df)
            all_rectified_data.append(rect_df) 

        combined_smoothed = pd.concat(all_smoothed_data, ignore_index=True)
        combined_pre_rect = pd.concat(all_pre_rect_data, ignore_index=True)
        combined_rectified = pd.concat(all_rectified_data, ignore_index=True) 


        smoothed_win, y_labels = sliding_window(combined_smoothed, window_size=200, stride=100)
        pre_rect_win = slide_filt(combined_pre_rect, window_size=200, stride=100)
        rectified_win = slide_filt(combined_rectified, window_size=200, stride=100) 

        # --- Calculate features from the windows ---
        X_rms = compute_rms(smoothed_win)
        X_mav = mean_absolute_value(rectified_win) # <-- USE RECTIFIED WINDOWS
        X_zc = count_zero_crossings_diff(pre_rect_win) # <-- USE PRE-RECTIFIED WINDOWS
        X_features = np.concatenate([X_rms, X_mav, X_zc], axis=1)


        df_rms = pd.DataFrame(X_rms, columns=[f'Ch{i+1}' for i in range(X_rms.shape[1])])
        df_rms['Label'] = y_labels

        sns.pairplot(df_rms, hue='Label', palette='tab10', corner=True)
        plt.suptitle("Pairwise RMS Features", y=1.02)
        plt.tight_layout()
        plt.show()

        # --- PCA STEP ---
        # 1. Initialize PCA.
        self.pca_model = PCA(n_components=2)

        # 2. FIT on the training data and TRANSFORM it.
        X_pca = self.pca_model.fit_transform(X_features)
        print(f"PCA applied. Original features: {X_features.shape[1]}, PCA features: {X_pca.shape[1]}")
        plt.figure(figsize=(8, 6))
        for cls in np.unique(y_labels):
            idx = y_labels == cls
            plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=str(cls), alpha=0.6)

        plt.title("PCA on RMS Features from Sliding Windows")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(title="Stimulus State")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


        X_train, X_test, y_train, y_test = train_test_split(X_pca, y_labels, test_size=0.2, random_state=42) #-----------------------

        if self.classifier_type == "LDA_X":
            self.mlp_model = LinearDiscriminantAnalysis()
        elif self.classifier_type == "MLP_Classifier":
            self.mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
            print("Trained MLP")
        else:
            raise ValueError(f"Unsupported classifier: {self.classifier_type}")

        self.mlp_model.fit(X_train, y_train)

        y_pred = self.mlp_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc*100:.2f}%")

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
        disp.plot(cmap='Blues')
        plt.title(f"{self.classifier_type} Confusion Matrix")
        plt.show()

        # --- SAVING STEP ---
        acc_int = int(acc * 100)
        gesture_names = "_".join(self.gesture_filepaths.keys())
        self.model_save_path = f"{self.classifier_type}_{acc_int}_{self.file_identifier}_{gesture_names}_{self.file_suffix}.pkl"
        joblib.dump(self.mlp_model, self.model_save_path)
        print(f"Model saved to: {self.model_save_path}")

        # Save the PCA model
        self.pca_save_path = f"PCA_for_{self.classifier_type}_{self.file_identifier}_{gesture_names}_{self.file_suffix}.pkl"
        joblib.dump(self.pca_model, self.pca_save_path)
        print(f"PCA model saved to: {self.pca_save_path}")


        return acc

    def load_model(self):
        self.mlp_model = joblib.load(self.model_save_path)
        self.pca_model = joblib.load(self.pca_save_path)
        return self.mlp_model, self.pca_model



# if __name__ == "__main__":
#     gest_paths = {"wflex" : "E:\Mindrove_venv\Smart_select_Wflex_data_2025_06_16_0_unfilt.csv",
#      "wexten": "E:\Mindrove_venv\Smart_select_WExten_data_2025_06_16_0_unfilt.csv",
#     "findex" : "E:\Mindrove_venv\Smart_select_FIndex_data_2025_06_16_0_unfilt.csv"
#      }

#     trainer = LDA_Trainer_Multi( 
#         gesture_filepaths = gest_paths,
#         file_identifier="testing",
#         file_suffix=0,
#         classifier_type="MLP_Classifier"
#     )

#     accuracy = trainer.train_model()
#     print(f"Offline training complete. Accuracy: {accuracy:.2%}")

