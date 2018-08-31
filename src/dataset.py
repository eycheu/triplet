import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    def __init__(self, path, n_neg=1000):
        print("Loading Dummy Dataset ...")
        self.df = pd.read_hdf(path)
        
        self.signal_groups = [
                 "signal1",
                 "signal2",
                 "signal3",
                 "signal4",
                 "signal5",
                 "signal6",
                 "signal7",
                 "signal8",
                 "signal9"
                ]

        self.signal_sequences = []
        for s_day in range(-29, 0, 1):
            e_day = s_day + 1
            str_s_day = str(s_day).replace("-", "m")
            str_e_day = str(e_day).replace("-", "m")
            s = "day_{}_{}".format(str_s_day, str_e_day)
            self.signal_sequences.append(s)
        
        # Drop NA 
        self.feature_cols = []
        for grp in self.signal_groups:
            self.feature_cols += ["{}_{}".format(seq, grp) for seq in self.signal_sequences]
        self.df = self.df[["class"] + self.feature_cols].dropna()

        df_pos = self.df[self.df["class"]==1].copy()
        df_neg = self.df[self.df["class"]==0].sample(n_neg)
        self.df = pd.concat([df_pos, df_neg])
        print("Class distribution:")
        print(self.df["class"].value_counts())

        self.labels = self.df["class"].values
        self.n_grps = len(self.signal_groups)
        self.n_steps = len(self.signal_sequences)
        print("Number of signals: {}".format(self.n_grps))
        print("Number of signal steps: {}".format(self.n_steps))
        print("Number of records: {}".format(self.__len__()))
        
        list_col = []
        for i, grp in enumerate(self.signal_groups):
            list_col.extend(["{}_{}".format(seq, grp) for seq in self.signal_sequences])

        self.cache_x = self.df[list_col].values.reshape((len(self.df), self.n_grps, self.n_steps)).astype(dtype=np.float32)
    
    def __getitem__(self, index):
        x = self.cache_x[index, :, :]
        x = x.reshape((1, x.shape[0], x.shape[1]))
        target = self.labels[index]
        return x, target

    def __len__(self):
        return len(self.df)

