import os
from dataclasses import dataclass

from wav2vec2 import Wav2Vec2Processor
import tensorflow as tf
import tensorflow_io as tfio
from functools import partial


@dataclass
class DataLoader:
    data_dir: str = "data"
    batch_size: int = 4
    audio_maxlen: int = 50000
    pad_id: int = 0

    processor: Wav2Vec2Processor = Wav2Vec2Processor(is_tokenizer=False)

    file_pattern: str = ".wav"
    buffer_size: int = 10000

    def __post_init__(self):
        assert self.file_pattern in [".wav", ".flac"]

    def decode_sound(self, file_path):
        audio = tf.io.read_file(file_path)
        if self.file_pattern == ".wav":
            audio, _ = tf.audio.decode_wav(audio)
        else:
            audio = tfio.audio.decode_flac(audio, dtype=tf.int16)
            audio = tf.cast(audio, dtype=tf.float32)
        return tf.squeeze(audio)[:self.audio_maxlen]

    def __call__(self, is_train=False):
        file_paths = []
        self._fetch_and_push_files(self.data_dir, file_paths)
        print(f"We could load {len(file_paths)} files from `{self.data_dir}` directory")
        dataset = tf.data.Dataset.from_tensor_slices(file_paths).map(self.decode_sound)
        dataset = dataset.map(self.processor)
        if is_train:
            dataset.shuffle(self.buffer_size)
        pad_id = float(self.pad_id)
        dataset = dataset.padded_batch(self.batch_size, padded_shapes=self.audio_maxlen, padding_values=pad_id)
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

    tr_dataset = DataLoader(data_dir="../data/librispeech/test-clean", file_pattern=".flac")(is_train=True)

    for batch in tr_dataset:
        print("BATCH SHAPE", batch.shape)
        break

    val_dataset = DataLoader(data_dir="../data/timit/data/TEST")(is_train=False)

    for batch in val_dataset:
        print("BATCH SHAPE", batch.shape)
        break