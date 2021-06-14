import os
from dataclasses import dataclass

from wav2vec2 import Wav2Vec2Processor
import tensorflow as tf
import tensorflow_io as tfio


@dataclass
class DataLoader:
    data_dir: str = "../data/LibriSpeech/data"
    batch_size: int = 16
    audio_maxlen: int = 400000
    pad_id: int = 0

    processor: Wav2Vec2Processor = Wav2Vec2Processor(is_tokenizer=False)

    dataset: str = "librispeech" # timit
    buffer_size: int = 10000

    def __post_init__(self):
        assert self.dataset in ["librispeech", "timit"]
        if self.dataset == "librispeech":
            self.file_pattern = ".flac"
        else:
            self.file_pattern = ".wav"

    def decode_sound(self, file_path):
        audio = tf.io.read_file(file_path)
        if self.file_pattern == ".wav":
            audio, _ = tf.audio.decode_wav(audio)
        else:
            audio = tfio.audio.decode_flac(audio, dtype=tf.int16)
            audio = tf.cast(audio, dtype=tf.float32)
        return tf.squeeze(audio)[:self.audio_maxlen]

    def __call__(self, shuffle=False):

        # fetch audio file names
        file_paths = []
        self._fetch_and_push_files(self.data_dir, file_paths, self.file_pattern)
        end = len(self.file_pattern)
        file_names = [os.path.basename(f)[:-end] for f in file_paths]

        # reading .txt file
        if self.dataset == "librispeech":
            samples = self._fetch_librispeeh_txt()
            all_mapping = [(fp, samples.pop(fn, None)) for fp, fn in zip(file_paths, file_names)]
            mapping = [j for j in all_mapping if j[1] is not None]
            print(f"DISCARDING {len(all_mapping) - len(mapping)} samples")

        print(f"LOADED {len(mapping)} FILES FROM {self.data_dir}")

        dataset = tf.data.Dataset.from_tensor_slices(mapping)
        dataset = dataset.map(lambda x: (self.decode_sound(x[0]), x[1]))
        dataset = dataset.map(lambda i, j: (self.processor(i), j))

        # shuffling for training
        if shuffle:
            dataset.shuffle(self.buffer_size)

        pad_id = float(self.pad_id)
        speech_dataset = dataset.map(lambda i,j: i).padded_batch(self.batch_size, padded_shapes=self.audio_maxlen, padding_values=pad_id)
        txt_dataset = dataset.map(lambda i,j: j).batch(self.batch_size)
        dataset = tf.data.Dataset.zip((speech_dataset, txt_dataset))

        return dataset

    def _fetch_librispeeh_txt(self):
        txt_paths = []
        self._fetch_and_push_files(self.data_dir, txt_paths, ".txt")
        all_samples = {}
        for file in txt_paths:
            with open(file, "r") as f:
                samples = f.read().split("\n")
                samples = {s.split()[0]: " ".join(s.split()[1:]) for s in samples if len(s.split()) > 2}
                all_samples.update(samples)
        return all_samples

    def _fetch_and_push_files(self, data_dir, file_paths: list, file_pattern: str):
        """All files will be recursively collected from the `data_dir`"""
        ls = os.listdir(data_dir)
        for f in ls:
            f = os.path.join(data_dir, f)
            if f.endswith(file_pattern):
                f = os.path.abspath(f)
                file_paths.append(f)
                continue

            if os.path.isdir(f):
                self._fetch_and_push_files(f, file_paths, file_pattern)


if __name__ == "__main__":
    """Testing Area"""

    tr_dataset = DataLoader(data_dir="../data/librispeech/test-clean")(shuffle=True)

    for batch in tr_dataset:
        print("BATCH SHAPE", batch[0].shape, batch[1])
        break

    val_dataset = DataLoader(data_dir="../data/librispeech/test-clean")(shuffle=False)

    for batch in val_dataset:
        print("BATCH SHAPE", batch[0].shape, batch[1])
        break
