import pandas as pd
from torch.utils.data import Dataset, DataLoader
from features import AudioFeatureExtractor, RANDOM_STATE
from torchvision.transforms import ToTensor, Compose, Resize

import pathlib

transfrom = Compose([
    ToTensor(),
    Resize(size=(64,64), antialias=True),
])

BATCH_SIZE = 16

class Urban8kDataset(Dataset):
    def __init__(self, df, load_from_pickle=False, pickle_file='data/extracted_features.pkl'):
        self.audio_info = df
        self.feature_extractor = AudioFeatureExtractor()
        self.cache = {}
        self.transform = transfrom

        self.load_from_pickle = load_from_pickle
        if self.load_from_pickle:
            self.frame = pd.read_pickle(pickle_file)

    def __len__(self):
        return self.audio_info.shape[0]

    def __getitem__(self, idx):
        if idx not in self.cache.keys():
            if not self.load_from_pickle:
                dataset_path = '.'
                path = dataset_path + '/' + self.audio_info.iloc[idx]['path']
                label = self.audio_info.iloc[idx]['class_id']
                spectrogram = self.feature_extractor.spectrogram_from_file(path)
            else:
                spectrogram = self.frame.iloc[idx]['spectrogram']
                label = self.frame.iloc[idx]['class_id']

            self.cache[idx] = (spectrogram, label)
        else:
            spectrogram, label = self.cache[idx]

        return (self.transform(spectrogram), label)

def get_dataset_loaders(limit=None, pickle=False, project_dir='.'):
    from sklearn.model_selection import train_test_split
    extracted_features_csv=f'{project_dir}/data/extracted_features.csv'
    frame = pd.read_csv(extracted_features_csv)

    if limit is not None:
        frame = frame[0:limit]

    train_frame, test_frame = train_test_split(frame, test_size=0.2, random_state=RANDOM_STATE)

    pickle_file = f"{project_dir}/data/extracted_features.pkl"
    train_dataset = Urban8kDataset(train_frame, load_from_pickle=pickle, pickle_file=pickle_file)
    test_dataset = Urban8kDataset(test_frame, load_from_pickle=pickle, pickle_file=pickle_file)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return (train_loader, test_loader, train_dataset, test_dataset)

if __name__ == '__main__':

    train_loader, test_loader, train_dataset, test_dataset = get_dataset_loaders(limit=None, pickle=True)
    print(len(train_dataset))
    print(train_dataset[10][0].shape)
    print(train_dataset[10][0].shape)

    print(len(test_dataset))
    print(test_dataset[10])