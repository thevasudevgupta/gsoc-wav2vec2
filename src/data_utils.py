
from dataclasses import dataclass
import tensorflow as tf
import os
import soundfile as sf


def collate_fn(batch):
    batch = tf.convert_to_tensor(batch, dtype=tf.float32)
    return batch


@dataclass
class DataLoader:
    is_train: bool
    data_dir: str = "data"
    batch_size: int = 4
    audio_maxlen: int = -1

    file_pattern: str = ".wav"
    buffer_size: int = 10000

    def list_files(self):
        return 

    def fetch_audio(self, f):
        print(f)
        f = f.numpy().item()
        audio, samplerate = sf.read(os.path.join(self.data_dir, f))
        assert samplerate == 16000, "It is advisable to have sample rate of 16000 for training Wav2Vec2"
        return audio[None, :self.audio_maxlen]

    def __call__(self):
        dataset = self._fetch_dataset()
        if self.is_train:
            dataset.shuffle(self.buffer_size)        
        dataset = dataset.padded_batch(self.batch_size).map(collate_fn)
        return dataset

    def _fetch_dataset(self):
        file_path = os.path.join(self.data_dir, "*" + self.file_pattern)
        return tf.data.Dataset.list_files(file_path).map(self.fetch_audio)


if __name__ == "__main__":
    """Testing Area"""

    tr_dataloader = DataLoader(is_train=True, data_dir="timit/data/TRAIN/DR1/FCJF0", batch_size=4)
    print(tr_dataloader)
    tr_dataset = tr_dataloader()

    for batch in tr_dataset:
        print(batch)
        exit()

    # val_dataloader = DataLoader(is_train=False, batch_size=4)
    # print(val_dataloader)
    # val_dataset = val_dataloader()
