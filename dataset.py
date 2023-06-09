import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from features import AudioFeatureExtractor, RANDOM_STATE
from torchvision.transforms import ToTensor, Compose, Resize

class Urban8kDataset(Dataset):
    def __init__(self, df):
        self.audio_info = df
        self.feature_extractor = AudioFeatureExtractor()
        self.cache = {}
        self.transform = Compose([
            ToTensor(),
            Resize(size=(64,64)),
        ])

    def __len__(self):
        return self.audio_info.shape[0]

    def __getitem__(self, idx):
        if idx not in self.cache.keys():
            dataset_path = '.'
            path = dataset_path + '/' + self.audio_info.iloc[idx]['path']
            label = self.audio_info.iloc[idx]['class_id']
            spectrogram = self.feature_extractor.spectrogram_from_file(path)

            self.cache[idx] = (spectrogram, label)
        else:
            spectrogram, label = self.cache[idx]

        # return (self.transform(spectrogram).permute(1, 2, 0), label)
        return (self.transform(spectrogram), label)

def get_dataset_loaders(limit=None):
    from sklearn.model_selection import train_test_split
    frame = pd.read_csv('extracted_features.csv')

    if limit is not None:
        frame = frame[0:limit]

    train_frame, test_frame = train_test_split(frame, test_size=0.2, random_state=RANDOM_STATE)

    train_dataset = Urban8kDataset(train_frame)
    test_dataset = Urban8kDataset(test_frame)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    return (train_loader, test_loader, train_dataset, test_dataset)

if __name__ == '__main__':

    train_loader, test_loader, train_dataset, test_dataset = get_dataset_loaders(limit=1000)
    print(len(train_dataset))
    print(train_dataset[10][0].shape)
    print(train_dataset[10][0].shape)

    print(len(test_dataset))
    print(test_dataset[10])