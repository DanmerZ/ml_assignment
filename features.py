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

        return features
    
def extract_features_from_urban8k_dataset(dataset_dir="input"):
    """
    UrbanSound8K dataset
    http://serv.cusp.nyu.edu/projects/urbansounddataset

    Extracts features into JSON file
    """
    df = pd.read_csv('input/metadata/UrbanSound8K.csv')
    feature_extractor = AudioFeatureExtractor()

    audio_files_info = []

    i = 0
    for root, dirs, files in os.walk(f"{dataset_dir}/audio"):
        for file in files:
            if not file.endswith(".wav"):
                continue
        
            path = root + "/" + file
            features = feature_extractor.extract_features(path)
            class_ = df.loc[df['slice_file_name'] == file]['class'].values
            class_id = df.loc[df['slice_file_name'] == file]['classID'].values

            audio_files_info.append({
                "path": path,
                "class": class_[0],
                "class_id": class_id[0],
                "features": features
            })
            i += 1
            print(f"{i} {path}")

    frame = pd.DataFrame.from_records(np.array(audio_files_info))
    frame.to_csv("extracted_features.csv")        

    # with open("extracted_features.json", "w") as f:
    #     json.dump(audio_files_info, f, indent=2)

    return audio_files_info

def read_features_from_csv(path_to_csv="extracted_features.csv"):
    # frame = pd.read_csv(path_to_csv, converters={'features': lambda x: np.array(pd.eval(x)) })
    frame = pd.read_csv(path_to_csv, converters={'features': pd.eval })

    train_frame, test_frame = train_test_split(frame, test_size=0.2, random_state=RANDOM_STATE)

    x_train = np.array(train_frame["features"].values.tolist())
    x_test = np.array(test_frame["features"].values.tolist())
    
    y_train = train_frame['class_id'].values
    y_test = test_frame['class_id'].values

    return (train_frame, test_frame, x_train, x_test, y_train, y_test) 


if __name__ == '__main__':
    # extract_features_from_urban8k_dataset()
    read_features_from_csv()

