import os
from typing import Iterable, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf

def preprocess_data_batch(data_batch: Iterable):
    # TODO: add data preprocessing
    data_batch = tf.cast(np.stack(data_batch), dtype=tf.float32)
    return data_batch

def preprocess_label_batch(label_batch: Iterable):
    # TODO: add label preprocessing
    label_batch = tf.cast(np.stack(label_batch), dtype=tf.float32)
    return label_batch

class ADNIDataLoader(tf.keras.utils.Sequence):

    def __init__(self,
                 data_path: str,
                 metadata_path: str,
                 patients: Optional[Iterable[str]] = None,
                 batch_size: int = None,
                 shuffle_all: bool = True,
                 shuffle_batch: bool = True):

        self.data_path = data_path
        self.metadata_path = metadata_path

        self.metadata = pd.read_csv(metadata_path)

        if patients is None:
            self.patients = sorted(os.listdir(data_path))
        else:
            self.patients = list(patients)

        if shuffle_all:
            np.random.shuffle(self.patients)

        self.batch_size = batch_size
        self.shuffle_all = shuffle_all
        self.shuffle_batch = shuffle_batch

    def __len__(self) -> int:
        return int(np.ceil(len(self.patients) / self.batch_size))

    def __getitem__(self, item: int) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        if self.batch_size * (item + 1) >= len(self.patients):
            batch_patients = self.patients[self.batch_size * item:]
        else:
            batch_patients = self.patients[self.batch_size * item:self.batch_size * (item + 1)]

        if self.shuffle_batch:
            np.random.shuffle(batch_patients)

        data_batch = []
        label_batch = []
        for patient in batch_patients:
            if self.verbose:
                print(f"Loading patient {patient}.")
            patient_dir = os.path.join(self.data_path, patient)
            try:
                data_batch.append(nib.load(patient_dir)).get_fdata()
            except FileNotFoundError as e:
                if self.verbose:
                    print(f"Missing file for patient {patient}: {e.filename}")

            label_batch.append(self.metadata[self.metadata["Image Data ID"] == patient]["Group"].iloc[0])

        data_batch = preprocess_data_batch(data_batch)
        label_batch = preprocess_label_batch(label_batch)

        return data_batch, label_batch