import sys

sys.path.append("./src")

import unittest

import numpy as np
import tensorflow as tf
from utils import (
    is_torch_available,
    is_transformers_available,
    requires_torch,
    requires_transformers,
)

from wav2vec2 import Wav2Vec2ForCTC

MODEL_ID = "wav2vec2-base-960h"
HF_MODEL_ID = "facebook/wav2vec2-base-960h"

if is_torch_available():
    import torch

if is_transformers_available():
    from transformers import Wav2Vec2ForCTC as HFWav2Vec2ForCTC


class Wav2Vec2Tester(unittest.TestCase):
    @requires_transformers
    @requires_torch
    def _test_inference(self, test_graph_mode=False):
        @tf.function(autograph=True, jit_compile=True)
        def tf_forward(*args, **kwargs):
            return tf_model(*args, **kwargs)

        batch, _ = tf.audio.decode_wav(tf.io.read_file("data/sample.wav"))
        batch = tf.transpose(batch, perm=(1, 0))
        hf_batch = torch.from_numpy(batch.numpy()).float()

        tf_model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, input_shape=batch.shape)
        hf_model = HFWav2Vec2ForCTC.from_pretrained(HF_MODEL_ID)

        if test_graph_mode:
            tf_out = tf_forward(batch, training=False)["logits"]
        else:
            tf_out = tf_model(batch, training=False)["logits"]
        with torch.no_grad():
            hf_out = hf_model(hf_batch)["logits"]

        assert tf_out.shape == hf_out.shape
        assert np.allclose(hf_out.numpy(), tf_out.numpy(), atol=0.004)

    def test_inference(self):
        self._test_inference(test_graph_mode=False)

    def test_jit_and_graph_mode(self):
        self._test_inference(test_graph_mode=True)
