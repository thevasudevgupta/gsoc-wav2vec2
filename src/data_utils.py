import os
from dataclasses import dataclass, field
from functools import partial
from typing import List, Tuple

import tensorflow as tf

import soundfile as sf
from wav2vec2 import Wav2Vec2Processor


SPEECH_DTYPE = tf.float32
LABEL_DTYPE = tf.int32
AUTOTUNE = tf.data.AUTOTUNE


def read_tfrecords(record):
    desc = {
        "speech": tf.io.FixedLenFeature((), tf.string),
        "label": tf.io.FixedLenFeature((), tf.string),
    }
    record = tf.io.parse_single_example(record, desc)

    speech = tf.io.parse_tensor(record["speech"], out_type=SPEECH_DTYPE)
    label = tf.io.parse_tensor(record["label"], out_type=LABEL_DTYPE)

    return speech, label


class CommonDataLoader:
    def __init__(self, batch_size, buffer_size, audio_pad_id, labels_pad_id, audio_maxlen, labels_maxlen):
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.audio_pad_id = float(audio_pad_id)
        self.labels_pad_id = labels_pad_id

        self.audio_maxlen = audio_maxlen
        self.labels_maxlen = labels_maxlen

        self.processor = Wav2Vec2Processor(is_tokenizer=False)
        self.tokenizer = Wav2Vec2Processor(is_tokenizer=True)

    def batchify(self, dataset, seed=None, drop_remainder=True):
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


@dataclass
class LibriSpeechDataLoaderArgs:
    from_tfrecords: bool = False
    tfrecords: List[str] = field(
        default_factory=lambda: ["gs://gsoc-librispeech/test/test-clean.tfrecord"]
    )
    data_dir: str = "../data/LibriSpeech/test-clean"

    batch_size: int = 16
    buffer_size: int = 10000

    audio_maxlen: int = 400000
    audio_pad_id: int = 0

    labels_maxlen: int = 128
    labels_pad_id: int = 0

    def __post_init__(self):
        if self.from_tfrecords:
            self.data_dir = None
            assert (
                self.tfrecords is not None
            ), "You must specify `tfrecords` when `from_tfrecords=True`."
        else:
            self.tfrecords = None
            assert (
                self.data_dir is not None
            ), "You must specify `data_dir` when `from_tfrecords=False`."


@dataclass
class TimitDataLoaderArgs:
    data_dir: str = "../data/timit/data/TRAIN"

    batch_size: int = 16
    buffer_size: int = 10000

    audio_maxlen: int = 400000
    audio_pad_id: int = 0

    labels_maxlen: int = 128
    labels_pad_id: int = 0


class LibriSpeechDataLoader(CommonDataLoader):
    def __init__(
        self, args: LibriSpeechDataLoaderArgs, required_sample_rate: int = 16000
    ):
        super().__init__(args.batch_size, args.buffer_size, args.audio_pad_id, args.labels_pad_id, args.audio_maxlen, args.labels_maxlen)

        self.from_tfrecords = args.from_tfrecords
        self.tfrecords = args.tfrecords
        self.data_dir = args.data_dir

        self.required_sample_rate = required_sample_rate

        self._num_samples = None

    def __call__(self, seed=None, drop_remainder=True) -> tf.data.Dataset:

        if not self.from_tfrecords:
            dataset = self.build_and_fetch_dataset()
        else:
            print(f"Reading tfrecords from {self.tfrecords}", end=" ... ")
            dataset = tf.data.TFRecordDataset(self.tfrecords)
            dataset = dataset.map(read_tfrecords, num_parallel_calls=AUTOTUNE)
            print("Done!")

        return self.batchify(dataset, seed=seed, drop_remainder=drop_remainder)

    def build_and_fetch_dataset(self):
        """
        This method builds the tf.data.Dataset from the data present in `data_dir`
        It uses `.from_generator` under the hood, so all the constraints related to that applies

        Returns:
            tf.data.Dataset[tf.Tensor, tf.Tensor]
            1st index - speech tensor
            2nd index - text label tensor
        """
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
            tf.TensorSpec(shape=(None), dtype=SPEECH_DTYPE),
            tf.TensorSpec(shape=(None), dtype=LABEL_DTYPE),
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
            raise ValueError(
                f"sample rate (={sample_rate}) of your files must be {self.required_sample_rate}"
            )
        audio = tf.constant(audio, dtype=SPEECH_DTYPE)
        return tf.transpose(audio)

    def _inputs_generator(self, text_by_filepath: List[Tuple[str, str]]):
        for file_path, text in text_by_filepath:
            speech = self.read_sound(file_path)
            speech = self.processor(speech)
            label = tf.constant(self.tokenizer(text), dtype=LABEL_DTYPE)
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


class TimitDataLoader(CommonDataLoader):
    def __init__(self, args: TimitDataLoaderArgs):
        super().__init__(args.batch_size, args.buffer_size, args.audio_pad_id, args.labels_pad_id, args.audio_maxlen, args.labels_maxlen)

        self.data_dir = args.data_dir

        self.wav_ext = ".WAV"
        self.txt_ext = ".TXT"

    def __call__(self, seed=None, drop_remainder=True):
        wav_files, txt_files = [], []
        self._fetch_and_push_files(self.data_dir, wav_files, self.wav_ext)
        self._fetch_and_push_files(self.data_dir, txt_files, self.txt_ext)

        wav_files = set([f[:-len(self.wav_ext)] for f in wav_files])
        txt_files = set([f[:-len(self.txt_ext)] for f in txt_files])

        # consider only those files which has both text & speech
        files = list(wav_files & txt_files)
        print(f"found {len(files)} samples in {self.data_dir}")

        wav_files = [f+self.wav_ext for f in files]
        txt_files = [f+self.txt_ext for f in files]

        labels = [self._prepare_labels(self.read_timit_txt(f)) for f in txt_files]

        dataset = tf.data.Dataset.from_tensor_slices((wav_files, labels))
        dataset = dataset.map(lambda sound_path, label: (self.read_sound(sound_path), label))
        return self.batchify(dataset, seed=seed, drop_remainder=drop_remainder)

    def _prepare_labels(self, text: str):
        def _pad(sample: list):
            while len(sample) < max_length:
                sample.append(pad_id)
            return sample

        max_length = self.labels_maxlen
        pad_id = self.labels_pad_id
        # TODO : add some processing for cleaning text a bit
        return _pad(self.tokenizer(text))

    def read_timit_txt(self, file_path: str):
        with open(file_path, "r") as f:
            text = f.read().split()[2:]
        return " ".join(text)

    def read_sound(self, file_path):
        audio = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio)
        return self.processor(tf.squeeze(audio))


if __name__ == "__main__":
    """Testing Area"""

    tokenizer = Wav2Vec2Processor(is_tokenizer=True)

    data_args = TimitDataLoaderArgs(data_dir="../data/timit/data/TRAIN")
    dataloader = TimitDataLoader(data_args)
    dataset = dataloader(seed=None)

    print("########### done ###########")
    for batch in dataset.take(32):
        print("BATCH SHAPE:", batch[0].shape)
        print("BATCH:", tokenizer.decode(batch[1][0].numpy().tolist()))

    # data_args = LibriSpeechDataLoaderArgs(
    #     from_tfrecords=True, tfrecords=["../data/test/test-clean.tfrecord"]
    # )
    # dataloader = LibriSpeechDataLoader(data_args)
    # dataset = dataloader(seed=None)

    # for batch in dataset.take(32):
    #     print("BATCH SHAPE:", batch[0].shape)
    #     print("BATCH:", tokenizer.decode(batch[1][0].numpy().tolist()))

    # data_args = LibriSpeechDataLoaderArgs(from_tfrecords=False)
    # dataloader = LibriSpeechDataLoader(data_args)
    # dataset = dataloader(seed=None)

    # for batch in dataset.take(32):
    #     print("BATCH SHAPE", batch[0].shape)
