import os
from dataclasses import dataclass

import tensorflow as tf


@dataclass
class DataLoader:
    data_dir: str = "data"
    batch_size: int = 4
    audio_maxlen: int = 50000

    file_pattern: str = ".wav"
    buffer_size: int = 10000

    def decode_wav(self, file_path):
        audio, _ = tf.audio.decode_wav(tf.io.read_file(file_path))
        return tf.squeeze(audio)[: self.audio_maxlen]

    def __call__(self, is_train=False):
        file_paths = []
        self._fetch_and_push_files(self.data_dir, file_paths)
        print(f"We could load {len(file_paths)} files from `{self.data_dir}` directory")
        dataset = tf.data.Dataset.from_tensor_slices(file_paths).map(self.decode_wav)
        if is_train:
            dataset.shuffle(self.buffer_size)
        dataset = dataset.padded_batch(self.batch_size, padded_shapes=self.audio_maxlen)
        return dataset

    def _fetch_and_push_files(self, data_dir, file_paths):
        """All files will be recursively collected from the `data_dir`"""
        ls = os.listdir(data_dir)
        for f in ls:
            f = os.path.join(data_dir, f)
            if f.endswith(self.file_pattern):
                f = os.path.abspath(f)
                file_paths.append(f)
                continue

            if os.path.isdir(f):
                self._fetch_and_push_files(f, file_paths)


if __name__ == "__main__":
    """Testing Area"""

    tr_dataset = DataLoader(data_dir="timit/data/TRAIN")(is_train=True)

    for batch in tr_dataset:
        print("BATCH SHAPE", batch.shape)
        break

    val_dataset = DataLoader(data_dir="timit/data/TEST")(is_train=False)
