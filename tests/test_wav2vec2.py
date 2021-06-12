import sys

sys.path.append("./src")

import unittest
from functools import partial

import numpy as np
import tensorflow as tf
from utils import (
    is_torch_available,
    is_transformers_available,
    requires_lib
)

from wav2vec2 import Wav2Vec2ForCTC, Wav2Vec2Processer

MODEL_ID = "wav2vec2-base-960h"
HF_MODEL_ID = "facebook/wav2vec2-base-960h"
SEED = 0

if is_torch_available():
    import torch

if is_transformers_available():
    from transformers import Wav2Vec2ForCTC as HFWav2Vec2ForCTC, Wav2Vec2FeatureExtractor as HFWav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer as HFWav2Vec2CTCTokenizer


class Wav2Vec2Tester(unittest.TestCase):

    def _get_batches(self):
        batch, _ = tf.audio.decode_wav(tf.io.read_file("data/sample.wav"))
        batch = tf.transpose(batch, perm=(1, 0))

        tf.random.set_seed(SEED)
        batch = tf.concat([batch, tf.random.normal(batch.shape)], axis=0)
        hf_batch = torch.from_numpy(batch.numpy()).float()
        return batch, hf_batch

    @partial(requires_lib, lib=["torch", "transformers"])
    def _test_inference(self, test_graph_mode=False):
        @tf.function(autograph=True, jit_compile=True)
        def tf_forward(*args, **kwargs):
            return tf_model(*args, **kwargs)

        batch, hf_batch = self._get_batches()

        tf_model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, input_shape=batch.shape)
        hf_model = HFWav2Vec2ForCTC.from_pretrained(HF_MODEL_ID)

        if test_graph_mode:
            tf_out = tf_forward(batch, training=False)["logits"]
        else:
            tf_out = tf_model(batch, training=False)["logits"]
        with torch.no_grad():
            hf_out = hf_model(hf_batch)["logits"]

        tf_out = tf_out.numpy()
        hf_out = hf_out.numpy()

        assert tf_out.shape == hf_out.shape
        assert np.allclose(hf_out, tf_out, atol=0.004), f"difference:, {np.max(hf_out - tf_out)}"

    def test_inference(self):
        self._test_inference(test_graph_mode=False)

    def test_jit_and_graph_mode(self):
        self._test_inference(test_graph_mode=True)

    @partial(requires_lib, lib=["transformers"])
    def test_feature_extractor(self):
        batch, hf_batch = self._get_batches()
        tf_processor = Wav2Vec2Processer(is_tokenizer=False)
        hf_processor = HFWav2Vec2FeatureExtractor.from_pretrained(HF_MODEL_ID)

        tf_out = tf_processor(batch)
        hf_out = hf_processor(hf_batch.numpy().tolist())["input_values"]
        assert np.allclose(tf_out, hf_out, atol=0.01), f"difference:, {np.max(hf_out - tf_out)}"

    @partial(requires_lib, lib=["torch", "transformers"])
    def test_end2end(self):
        # data loading
        b1 = tf.transpose(tf.audio.decode_wav(tf.io.read_file("data/sample.wav"))[0], perm=(1, 0))
        b2 = tf.transpose(tf.audio.decode_wav(tf.io.read_file("data/SA2.wav"))[0], perm=(1, 0))
        batch = tf.concat([b1[:, :40000], b2[:, :40000]], axis=0)

        # data processing
        tf_processor = Wav2Vec2Processer(is_tokenizer=False)
        hf_processor = HFWav2Vec2FeatureExtractor.from_pretrained(HF_MODEL_ID)

        hf_batch = hf_processor(batch.numpy().tolist())["input_values"]
        hf_batch = torch.tensor(hf_batch, dtype=torch.float)
        batch = tf_processor(batch)

        assert batch.shape == hf_batch.shape
        assert np.allclose(batch, hf_batch, atol=1e-5), f"difference:, {np.max(batch - hf_batch)}"

        # model inference
        tf_model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, input_shape=batch.shape)
        hf_model = HFWav2Vec2ForCTC.from_pretrained(HF_MODEL_ID)
        
        tf_out = tf_model(batch, training=False)["logits"]
        with torch.no_grad():
            hf_out = hf_model(hf_batch)["logits"]
        tf_out = tf_out.numpy()
        hf_out = hf_out.numpy()

        assert tf_out.shape == hf_out.shape
        assert np.allclose(hf_out, tf_out, atol=0.004), f"difference:, {np.max(hf_out - tf_out)}"

        # decoding
        tf_tokenizer = Wav2Vec2Processer(is_tokenizer=True, vocab_path="data/vocab.json")
        hf_tokenizer = HFWav2Vec2CTCTokenizer.from_pretrained(HF_MODEL_ID)

        tf_out = np.argmax(tf_out, axis=-1).squeeze()
        hf_out = np.argmax(hf_out, axis=-1).squeeze()

        tf_pred = [tf_tokenizer.decode(output) for output in tf_out.tolist()]
        hf_pred = hf_tokenizer.batch_decode(hf_out)
        assert tf_pred == hf_pred, f"{tf_pred} VS {hf_pred}"
