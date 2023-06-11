import numpy as np
import pandas as pd
import librosa
import os, json
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

class AudioFeatureExtractor:
    def __init__(self) -> None:
        pass

    def mean_mfccs(self, x):
        return [np.mean(feature) for feature in librosa.feature.mfcc(y=x, n_mfcc=40)]

    def extract_features(self, path_to_audio: str) -> np.array:
        x, sr = librosa.load(path=path_to_audio)
        
        features = self.mean_mfccs(x)

        mean_rms = np.mean(librosa.feature.rms(y=x))
        features.append(mean_rms)

        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=x))
        features.append(spectral_centroid)

        zcr = np.mean(librosa.feature.zero_crossing_rate(y=x))
        features.append(zcr)

        return (features, self.spectrogram(wav=x, sr=sr))
    
    def spectrogram(self, wav, sr=None, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
        # From https://medium.com/@hasithsura/audio-classification-d37a82d6715
        # wav,sr = librosa.load(file_path,sr=sr)
        if wav.shape[0]<5*sr:
            wav = np.pad(wav,int(np.ceil((5*sr-wav.shape[0])/2)),mode='reflect')
        else:
            wav = wav[:5*sr]
        spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft,
                    hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
        spec_db = librosa.power_to_db(spec,top_db=top_db)
        # return spec_db
        return self.spec_to_image(spec_db)

    def spec_to_image(self, spec, eps=1e-6):
        # From https://medium.com/@hasithsura/audio-classification-d37a82d6715
        mean = spec.mean()
        std = spec.std()
        spec_norm = (spec - mean) / (std + eps)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
        spec_scaled = spec_scaled.astype(np.uint8)
        return spec_scaled
    
    def spectrogram_from_file(self, file):
        x, sr = librosa.load(path=file)
        return self.spectrogram(wav=x, sr=sr)
    
def extract_features_from_urban8k_dataset(dataset_dir="input", istop=None, pickle=False):
    """
    UrbanSound8K dataset
    http://serv.cusp.nyu.edu/projects/urbansounddataset

    Extracts features into CSV file
    """
    df = pd.read_csv('input/metadata/UrbanSound8K.csv')
    feature_extractor = AudioFeatureExtractor()

    audio_files_info = []
    audio_files_info_pickle = []

    i = 1
    for root, dirs, files in os.walk(f"{dataset_dir}/audio"):
        for file in files:
            if not file.endswith(".wav"):
                continue
        
            path = root + "/" + file
            features, spectrogram = feature_extractor.extract_features(path)
            class_ = df.loc[df['slice_file_name'] == file]['class'].values
            class_id = df.loc[df['slice_file_name'] == file]['classID'].values

            info = {
                "path": path,
                "class": class_[0],
                "class_id": class_id[0],
                "features": features
            }
            audio_files_info.append(info)

            if pickle:
                info["spectrogram"] = spectrogram
                audio_files_info_pickle.append(info)

            i += 1
            print(f"{i} {path}")

            if istop is not None and i >= istop:
                break

        if istop is not None and i >= istop:
            break

    frame = pd.DataFrame.from_records(np.array(audio_files_info))
    frame.to_csv("data/extracted_features.csv")  

    if pickle:
        frame = pd.DataFrame.from_records(np.array(audio_files_info_pickle))
        frame.to_pickle("data/extracted_features.pkl")
          
    return audio_files_info

def read_features_from_csv(path_to_csv="data/extracted_features.csv"):
    # frame = pd.read_csv(path_to_csv, converters={'features': lambda x: np.array(pd.eval(x)) })
    frame = pd.read_csv(path_to_csv, converters={'features': pd.eval })

    train_frame, test_frame = train_test_split(frame, test_size=0.2, random_state=RANDOM_STATE)

    x_train = np.array(train_frame["features"].values.tolist())
    x_test = np.array(test_frame["features"].values.tolist())
    
    y_train = train_frame['class_id'].values
    y_test = test_frame['class_id'].values

    return (train_frame, test_frame, x_train, x_test, y_train, y_test) 

def read_features_from_pickle(path_to_pickle="data/extracted_features.pkl"):
    frame = pd.read_pickle(path_to_pickle)

    train_frame, test_frame = train_test_split(frame, test_size=0.2, random_state=RANDOM_STATE)

    # return train_frame, test_frame

    x_train = np.array(train_frame["features"].values.tolist())
    x_test = np.array(test_frame["features"].values.tolist())
    spectrogram_train = np.array(train_frame['spectrogram'].values.tolist())
    spectrogram_test = np.array(test_frame['spectrogram'].values.tolist())
    
    y_train = train_frame['class_id'].values
    y_test = test_frame['class_id'].values

    return (train_frame, test_frame, x_train, x_test, y_train, y_test, spectrogram_train, spectrogram_test) 

if __name__ == '__main__':
    extract_features_from_urban8k_dataset(istop=None, pickle=True)
    # read_features_from_csv()
    # read_features_from_pickle()
    
