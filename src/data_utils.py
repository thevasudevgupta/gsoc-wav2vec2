import os
from dataclasses import dataclass
from functools import partial
from typing import List, Tuple

import tensorflow as tf
import soundfile as sf

import numpy as np
from wav2vec2 import Wav2Vec2Processor


SPEECH_DTYPE = np.float32
LABEL_DTYPE = np.int32
AUTOTUNE = tf.data.AUTOTUNE


def tfrecords_generator(files):
    def _read_tfrecord(record):
        if isinstance(record, tf.Tensor):
            record = record.numpy()

        record = tf.train.Example.FromString(record)
        speech_bytes = record.features.feature["speech"].bytes_list.value[0]
        label_bytes = record.features.feature["label"].bytes_list.value[0]

        speech = np.frombuffer(speech_bytes, dtype=SPEECH_DTYPE)
        label = np.frombuffer(label_bytes, dtype=LABEL_DTYPE)

        return tf.constant(speech, dtype=tf.float32), tf.constant(label, dtype=tf.int32)

    records = tf.data.TFRecordDataset(files)
    for record in records:
        speech, label = _read_tfrecord(record)
        yield speech, label


@dataclass
class LibriSpeechDataLoaderArgs:
    data_dir: str = "../data/LibriSpeech/data"
    batch_size: int = 16
    buffer_size: int = 10000

    audio_maxlen: int = 400000
    audio_pad_id: int = 0

    labels_maxlen: int = 128
    labels_pad_id: int = 0


class LibriSpeechDataLoader:
    def __init__(self, args: LibriSpeechDataLoaderArgs, required_sample_rate: int = 16000):
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size

        self.required_sample_rate = required_sample_rate

        self.audio_pad_id = float(args.audio_pad_id)
        self.labels_pad_id = args.labels_pad_id

        self.audio_maxlen = args.audio_maxlen
        self.labels_maxlen = args.labels_maxlen

        self.processor = Wav2Vec2Processor(is_tokenizer=False)
        self.tokenizer = Wav2Vec2Processor(is_tokenizer=True)

        self._num_samples = None

    def __call__(
        self, from_tfrecords=False, seed=None, drop_remainder=True
    ) -> tf.data.Dataset:

        if not from_tfrecords:
            dataset = self.build_and_fetch_dataset()
        else:
            files = os.listdir(self.data_dir)
            files = [os.path.abspath(os.path.join(self.data_dir, f)) for f in files if f.endswith(".tfrecord")]
            assert len(files) > 0, f"Unable to find `.tfrecord` in `{self.data_dir}``"
            print("Available files:\n", files)

            records_generator = partial(tfrecords_generator, files=files)
            output_signature = (
                tf.TensorSpec(shape=(None), dtype=tf.float32),
                tf.TensorSpec(shape=(None), dtype=tf.int32),
            )
            dataset = tf.data.Dataset.from_generator(records_generator, output_signature=output_signature)

        # shuffling for training
        if seed is not None:
            dataset.shuffle(self.buffer_size, seed=seed)

        padded_shapes = (self.audio_maxlen, self.labels_maxlen)
        padding_values = (self.audio_pad_id, self.labels_pad_id)

        dataset = dataset.map(self.restrict_to_maxlen)
        dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes=padded_shapes,
            padding_values=padding_values,
            drop_remainder=drop_remainder,
        )

        return dataset.prefetch(AUTOTUNE)

    def restrict_to_maxlen(self, speech, labels):
        """This must be called before doing padding"""
        speech, labels = speech[: self.audio_maxlen], labels[: self.labels_maxlen]
        return speech, labels

    def build_and_fetch_dataset(self):

        # fetch audio file names
        file_paths = []
        self._fetch_and_push_files(self.data_dir, file_paths, ".flac")
        end = len(".flac")  # remove the extension
        file_names = [os.path.basename(path)[:-end] for path in file_paths]

        # reading .txt file
        texts = self._fetch_librispeeh_txt()

        # combine text & file-paths
        text_by_filepath = [
            (file_path, texts.pop(file_name, None))
            for file_path, file_name in zip(file_paths, file_names)
        ]
        num_contaminated_samples = len(text_by_filepath)
        text_by_filepath = [
            inputs for inputs in text_by_filepath if inputs[1] is not None
        ]
        print(f"DISCARDING {num_contaminated_samples - len(text_by_filepath)} samples")

        print(f"LOADED {len(text_by_filepath)} FILES FROM {self.data_dir}")

        self._num_samples = len(text_by_filepath)

        inputs_generator = partial(self._inputs_generator, text_by_filepath)
        output_signature = (
            tf.TensorSpec(shape=(None), dtype=tf.float32),
            tf.TensorSpec(shape=(None), dtype=tf.int32),
        )
        dataset = tf.data.Dataset.from_generator(
            inputs_generator, output_signature=output_signature
        )
        return dataset

    def __len__(self):
        if self._num_samples is None:
            raise NotImplementedError
        return self._num_samples

    def read_sound(self, file_path):
        with open(file_path, "rb") as f:
            audio, sample_rate = sf.read(f)
        if sample_rate != self.required_sample_rate:
            raise ValueError(f"sample rate (={sample_rate}) of your files must be {self.required_sample_rate}")
        audio = tf.constant(audio, dtype=tf.float32)
        return tf.transpose(audio)

    def _inputs_generator(self, text_by_filepath: List[Tuple[str, str]]):
        for file_path, text in text_by_filepath:
            speech = self.read_sound(file_path)
            speech = self.processor(speech)
            label = tf.constant(self.tokenizer(text), dtype=tf.int32)
            yield speech, label

    def _fetch_librispeeh_txt(self) -> dict:
        """
        Read data from all the `.txt` files and returns dictionary mapping `file_id` & `text`.

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
        """All files will be recursively collected from the `data_dir`."""
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

    data_args = LibriSpeechDataLoaderArgs(data_dir="../data/LibriSpeech/test-clean")
    dataloader = LibriSpeechDataLoader(data_args)
    dataset = dataloader(seed=2)

    for batch in dataset.take(32):
        print("BATCH SHAPE", batch[0].shape)
        print("BATCH", tokenizer.decode(batch[1][0].numpy().tolist()))

    data_args = LibriSpeechDataLoaderArgs(data_dir="../data/LibriSpeech/test-clean")
    dataloader = LibriSpeechDataLoader(data_args)
    dataset = dataloader(seed=None, from_tfrecords=True)

    for batch in dataset.take(32):
        print("BATCH SHAPE", batch[0].shape)
