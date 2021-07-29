import sys


sys.path.extend(["src", "../src"])

import unittest
from functools import partial

import tensorflow as tf

import numpy as np
from data_utils import LibriSpeechDataLoader, LibriSpeechDataLoaderArgs, TimitDataLoader, TimitDataLoaderArgs
from utils import if_path_exists, try_download_file
from wav2vec2 import Wav2Vec2Processor


TFRECORD_URL = "https://huggingface.co/datasets/vasudevgupta/gsoc-librispeech/resolve/main/dev-clean/dev-clean-0.tfrecord"
LIBRISPEECH_DIR = "data/LibriSpeech/dev-clean"
WAV_PATH = "data/sample.wav"
FLAC_PATH = "data/sample2.flac"


class DataLoaderTester(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Wav2Vec2Processor(is_tokenizer=True)

    def test_librispeech_tfrecords(self):
        file_path = try_download_file(TFRECORD_URL)
        data_args = LibriSpeechDataLoaderArgs(from_tfrecords=True, tfrecords=[file_path])
        dataloader = LibriSpeechDataLoader(data_args)

        for batch in dataloader(seed=None):
            self.assertEqual((16, 400000), batch[0].shape)
            target = "THE PATERNAL PARENT HAS A RIGHT TO HIS INFANTS NO DOUBT THAT WAS BOZLE'S LAW"
            self.assertEqual(target, self.tokenizer.decode(batch[1][0].numpy().tolist()))
            break

    @partial(if_path_exists, path=LIBRISPEECH_DIR)
    def test_librispeech_original(self):
        data_args = LibriSpeechDataLoaderArgs(from_tfrecords=False, data_dir=LIBRISPEECH_DIR)
        dataloader = LibriSpeechDataLoader(data_args)

        for batch in dataloader(seed=None):
            self.assertEqual((16, 400000), batch[0].shape)
            break

    @partial(if_path_exists, path=FLAC_PATH)
    def test_read_flac(self):
        dataloader = LibriSpeechDataLoader(LibriSpeechDataLoaderArgs())
        audio = dataloader.read_sound(FLAC_PATH)
        target = tf.constant([
            6.1035156e-05, 1.8310547e-04, 5.4931641e-04, 3.6621094e-04, 3.3569336e-04, 5.1879883e-04, 7.6293945e-04, 8.5449219e-04
        ])
        self.assertTrue(np.allclose(audio[32:40].numpy(), target.numpy()))

    @partial(if_path_exists, path=WAV_PATH)
    def test_read_wav(self):
        dataloader = TimitDataLoader(TimitDataLoaderArgs())
        audio = dataloader.read_sound(WAV_PATH)
        target = tf.constant([
            0.01438822, 0.01776027, 0.01438822, 0.02113231, 0.01438822, 0.00764414, 0.00764414, -0.00921606
        ])
        self.assertTrue(np.allclose(audio[32:40].numpy(), target.numpy()))
