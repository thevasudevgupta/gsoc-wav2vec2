import os
from dataclasses import dataclass

import tensorflow as tf
import tensorflow_io as tfio

from wav2vec2 import Wav2Vec2Processor


AUTOTUNE = tf.data.AUTOTUNE


@dataclass
class LibriSpeechDataLoaderArgs:
    data_dir: str = "../data/LibriSpeech/data"
    batch_size: int = 16

    audio_maxlen: int = 400000
    audio_pad_id: int = 0

    labels_maxlen: int = 128
    labels_pad_id: int = 0


class LibriSpeechDataLoader:
    def __init__(self, args: LibriSpeechDataLoaderArgs):
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size

        self.audio_pad_id = float(args.audio_pad_id)
        self.labels_pad_id = args.labels_pad_id

        self.audio_maxlen = args.audio_maxlen
        self.labels_maxlen = args.labels_maxlen

        self.processor = Wav2Vec2Processor(is_tokenizer=False)
        self.tokenizer = Wav2Vec2Processor(is_tokenizer=True)

    def __call__(self, seed=None) -> tf.data.Dataset:

        dataset = self._build_and_fetch_dataset()

        # shuffling for training
        if seed is not None:
            dataset.shuffle(self.batch_size * 10, seed=seed)

        padded_shapes = (self.audio_maxlen, self.labels_maxlen)
        padding_values = (self.audio_pad_id, self.labels_pad_id)

        dataset = dataset.padded_batch(
            self.batch_size, padded_shapes=padded_shapes, padding_values=padding_values
        )

        return dataset.prefetch(AUTOTUNE)

    def _build_and_fetch_dataset(self):

        # fetch audio file names
        file_paths = []
        self._fetch_and_push_files(self.data_dir, file_paths, ".flac")
        end = len(".flac")  # remove the extension
        file_names = [os.path.basename(path)[:-end] for path in file_paths]

        # reading .txt file
        samples = self._fetch_librispeeh_txt()

        # combine text & file-paths
        text_by_filepath = [
            (file_path, samples.pop(file_name, None))
            for file_path, file_name in zip(file_paths, file_names)
        ]
        num_contaminated_samples = len(text_by_filepath)
        text_by_filepath = [
            inputs for inputs in text_by_filepath if inputs[1] is not None
        ]
        print(f"DISCARDING {num_contaminated_samples - len(text_by_filepath)} samples")

        print(f"LOADED {len(text_by_filepath)} FILES FROM {self.data_dir}")

        # we can't apply tokenizer to tf.string. Hence applying it now.
        labels_by_filepath = [
            (file_path, self._prepare_label(text))
            for file_path, text in text_by_filepath
        ]

        file_paths, labels = list(zip(*labels_by_filepath))

        label_dataset = tf.data.Dataset.from_tensor_slices(list(labels))

        input_dataset = tf.data.Dataset.from_tensor_slices(list(file_paths))
        input_dataset = input_dataset.map(
            self.decode_sound, num_parallel_calls=AUTOTUNE
        )
        input_dataset = input_dataset.map(self.processor, num_parallel_calls=AUTOTUNE)

        dataset = tf.data.Dataset.zip((input_dataset, label_dataset))
        return dataset

    def decode_sound(self, file_path):
        audio = tf.io.read_file(file_path)
        audio = tfio.audio.decode_flac(audio, dtype=tf.int16)
        audio = tf.cast(audio, tf.float32)
        return tf.squeeze(audio)[: self.audio_maxlen]

    def _prepare_label(self, string):
        def _pad(sample: list):
            while len(sample) < max_length:
                sample.append(pad_id)
            return sample

        max_length = self.labels_maxlen
        pad_id = self.labels_pad_id
        label = _pad(self.tokenizer(string))
        return label[:max_length]

    def _fetch_librispeeh_txt(self) -> dict:
        """
        Read data from all the `.txt` files and returns dictionary mapping `file_id` & `text`

        Eg:
            103201 Wav2Vec2 is SOTA model
            104332 Wav2Vec2-U is awesome
        returns:
            {
                "103201": "Wav2Vec2 is SOTA model",
                "104332": "Wav2Vec2-U is awesome",
            }
        """
        txt_paths = []
        self._fetch_and_push_files(self.data_dir, txt_paths, ".txt")
        all_samples = {}
        # this reads data from the files and maps file_id to the text
        for file in txt_paths:
            with open(file, "r") as f:
                samples = f.read().split("\n")
                samples = {
                    s.split()[0]: " ".join(s.split()[1:])
                    for s in samples
                    if len(s.split()) > 2
                }
                all_samples.update(samples)
        return all_samples

    def _fetch_and_push_files(self, data_dir, file_paths: list, file_pattern: str):
        """All files will be recursively collected from the `data_dir`"""
        listdir = os.listdir(data_dir)
        for f in listdir:
            f = os.path.join(data_dir, f)
            if f.endswith(file_pattern):
                f = os.path.abspath(f)
                file_paths.append(f)
                continue

            if os.path.isdir(f):
                self._fetch_and_push_files(f, file_paths, file_pattern)


if __name__ == "__main__":
    """Testing Area"""
    tokenizer = Wav2Vec2Processor(is_tokenizer=True)

    tr_data_args = LibriSpeechDataLoaderArgs(data_dir="../data/librispeech/test-clean")
    tr_dataloader = LibriSpeechDataLoader(tr_data_args)
    tr_dataset = tr_dataloader(seed=0)

    for batch in tr_dataset:
        print("BATCH SHAPE", batch[0].shape, batch[1])
        print("BATCH", tokenizer.decode(batch[1][0].numpy().tolist()))
        break

    val_data_args = LibriSpeechDataLoaderArgs(data_dir="../data/librispeech/test-clean")
    val_dataloader = LibriSpeechDataLoader(val_data_args)
    val_dataset = val_dataloader()

    for batch in val_dataset:
        print("BATCH SHAPE", batch[0].shape, batch[1])
        break
