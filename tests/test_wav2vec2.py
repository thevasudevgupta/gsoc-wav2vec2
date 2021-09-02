import unittest
from functools import partial

import tensorflow as tf

import numpy as np
import tensorflow_hub as hub
from convert_torch_to_tf import get_tf_pretrained_model
from utils import is_torch_available, is_transformers_available, requires_lib
from wav2vec2 import CTCLoss, Wav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Model, Wav2Vec2Processor
from wav2vec2.tensorflow_addons import Conv1DWithWeightNorm


if is_torch_available():
    import torch
    import torch.nn as nn

if is_transformers_available():
    from transformers import (
        Wav2Vec2CTCTokenizer as HFWav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor as HFWav2Vec2FeatureExtractor,
        Wav2Vec2ForCTC as HFWav2Vec2ForCTC,
        Wav2Vec2Model as HFWav2Vec2Model
    )

MODEL_ID = "vasudevgupta/gsoc-wav2vec2-960h"
HF_MODEL_ID = "facebook/wav2vec2-base-960h"
HF_MODEL_IDS = ["facebook/wav2vec2-base-960h", "facebook/wav2vec2-base"]
SEED = 0


class Wav2Vec2Tester(unittest.TestCase):
    def _get_batches(self):
        batch, _ = tf.audio.decode_wav(tf.io.read_file("data/sample.wav"))
        batch = tf.transpose(batch, perm=(1, 0))

        tf.random.set_seed(SEED)
        batch = tf.concat([batch, tf.random.normal(batch.shape)], axis=0)
        hf_batch = torch.from_numpy(batch.numpy()).float()

        np.random.seed(SEED)
        labels = np.random.randint(1, 30, size=(2, 24))
        tf_labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        hf_labels = torch.from_numpy(labels).long()
        return batch, hf_batch, tf_labels, hf_labels

    @partial(requires_lib, lib=["torch", "transformers"])
    def _test_inference(self, model_id, hf_model_id, test_graph_mode=False):
        @tf.function(autograph=True, jit_compile=True)
        def tf_forward(*args, **kwargs):
            return tf_model(*args, **kwargs)

        batch, hf_batch, _, _ = self._get_batches()

        tf_model = Wav2Vec2Model.from_pretrained(model_id, input_shape=batch.shape)
        hf_model = HFWav2Vec2Model.from_pretrained(hf_model_id)

        if tf_model.config.is_robust:
            attention_mask = np.ones(batch.shape, dtype=np.int32)
            attention_mask[0, -1000:] = attention_mask[1, -132:] = 0
            hf_attention_mask = torch.from_numpy(attention_mask)
            attention_mask = tf.convert_to_tensor(attention_mask)
        else:
            attention_mask = hf_attention_mask = None

        if test_graph_mode:
            tf_out = tf_forward(batch, attention_mask=attention_mask, training=False)
        else:
            tf_out = tf_model(batch, attention_mask=attention_mask, training=False)
        with torch.no_grad():
            hf_out = hf_model(hf_batch, attention_mask=hf_attention_mask)

        tf_logits = tf_out.numpy()
        hf_logits = hf_out["last_hidden_state"].numpy()

        assert tf_logits.shape == hf_logits.shape, "Oops, logits shape is not matching"
        assert np.allclose(
            hf_logits, tf_logits, atol=1e-3
        ), f"difference: {np.max(hf_logits - tf_logits)}"

    def test_inference(self):
        model_id, hf_model_id = "vasudevgupta/gsoc-wav2vec2", "facebook/wav2vec2-base"
        self._test_inference(model_id, hf_model_id, test_graph_mode=False)

    def test_wav2vec2_robust(self):
        model_id, hf_model_id = "vasudevgupta/gsoc-wav2vec2-robust", "facebook/wav2vec2-large-robust"
        self._test_inference(model_id, hf_model_id, test_graph_mode=False)

    def test_wav2vec2_xlsr(self):
        model_id, hf_model_id = "vasudevgupta/gsoc-wav2vec2-xlsr-53", "facebook/wav2vec2-large-xlsr-53"
        self._test_inference(model_id, hf_model_id, test_graph_mode=False)

    def test_jit_and_graph_mode(self):
        model_id, hf_model_id = "vasudevgupta/gsoc-wav2vec2", "facebook/wav2vec2-base"
        self._test_inference(model_id, hf_model_id, test_graph_mode=True)

    @partial(requires_lib, lib=["transformers"])
    def test_feature_extractor(self):
        batch, hf_batch, _, _ = self._get_batches()
        tf_processor = Wav2Vec2Processor(is_tokenizer=False)
        hf_processor = HFWav2Vec2FeatureExtractor.from_pretrained(HF_MODEL_ID)

        tf_out = tf_processor(batch)
        hf_out = hf_processor(hf_batch.numpy().tolist())["input_values"]
        assert np.allclose(
            tf_out, hf_out, atol=0.01
        ), f"difference:, {np.max(hf_out - tf_out)}"

    def test_end2end(self):
        model_id = "vasudevgupta/gsoc-wav2vec2-960h"
        hf_model_id = "facebook/wav2vec2-base-960h"
        self._test_end2end(model_id, hf_model_id)

    @partial(requires_lib, lib=["torch", "transformers"])
    def _test_end2end(self, model_id, hf_model_id):
        # data loading
        b1 = tf.transpose(
            tf.audio.decode_wav(tf.io.read_file("data/sample.wav"))[0], perm=(1, 0)
        )
        b2 = tf.transpose(
            tf.audio.decode_wav(tf.io.read_file("data/SA2.wav"))[0], perm=(1, 0)
        )
        batch = tf.concat([b1[:, :40000], b2[:, :40000]], axis=0)

        # data processing
        tf_processor = Wav2Vec2Processor(is_tokenizer=False)
        hf_processor = HFWav2Vec2FeatureExtractor.from_pretrained(hf_model_id)

        hf_batch = hf_processor(batch.numpy().tolist())["input_values"]
        hf_batch = torch.tensor(hf_batch, dtype=torch.float)
        batch = tf_processor(batch)

        assert batch.shape == hf_batch.shape
        assert np.allclose(
            batch, hf_batch, atol=1e-5
        ), f"difference:, {np.max(batch - hf_batch)}"

        # model inference
        tf_model = Wav2Vec2ForCTC.from_pretrained(model_id, input_shape=batch.shape)
        hf_model = HFWav2Vec2ForCTC.from_pretrained(hf_model_id)

        if tf_model.config.is_robust:
            attention_mask = tf.ones(batch.shape)
            hf_attention_mask = torch.tensor(attention_mask.numpy())
        else:
            attention_mask = hf_attention_mask = None

        tf_out = tf_model(batch, attention_mask=attention_mask, training=False)
        with torch.no_grad():
            hf_out = hf_model(hf_batch, attention_mask=hf_attention_mask)["logits"]
        tf_out = tf_out.numpy()
        hf_out = hf_out.numpy()

        assert tf_out.shape == hf_out.shape
        assert np.allclose(
            hf_out, tf_out, atol=0.004
        ), f"difference:, {np.max(hf_out - tf_out)}"

        # decoding
        tf_tokenizer = Wav2Vec2Processor(
            is_tokenizer=True, vocab_path="data/vocab.json"
        )
        hf_tokenizer = HFWav2Vec2CTCTokenizer.from_pretrained(hf_model_id)

        tf_out = np.argmax(tf_out, axis=-1).squeeze()
        hf_out = np.argmax(hf_out, axis=-1).squeeze()

        tf_pred = [tf_tokenizer.decode(output) for output in tf_out.tolist()]
        hf_pred = hf_tokenizer.batch_decode(hf_out)
        assert tf_pred == hf_pred, f"{tf_pred} VS {hf_pred}"

    @partial(requires_lib, lib=["transformers", "torch"])
    def test_conversion_script(self):
        for hf_model_id in HF_MODEL_IDS:
            config = Wav2Vec2Config()
            tf_model, hf_model = get_tf_pretrained_model(
                config,
                hf_model_id,
                verbose=False,
                with_lm_head=True,
            )
            batch, hf_batch, _, _ = self._get_batches()
            tf_logits = tf_model(batch).numpy()
            with torch.no_grad():
                hf_logits = hf_model(hf_batch, return_dict=False)
                hf_logits = hf_logits[0].numpy()
            assert np.allclose(
                hf_logits, tf_logits, atol=0.004
            ), f"difference: {np.max(hf_logits - tf_logits)}"

    @partial(requires_lib, lib=["torch", "transformers"])
    def test_loss_autograph(self):
        """
        This is very important test and shows how model forward pass should be written.

        Note:
            1. `Wav2Vec2ForCTC.call()` & `CTCLoss.__call__` both works in eager mode.
            2. In graph mode, `Wav2Vec2ForCTC.call()` doesn't work with `jit_compile=False` while it works when `jit_compile=True`.
            3. In graph mode, `CTCLoss.__call__` doesn't work with `jit_compile=True` while it works when `jit_compile=False`.
        """

        @tf.function(jit_compile=True)
        def tf_forward(batch):
            return tf_model(batch, training=False)

        @tf.function
        def compute_loss(batch, labels):
            batch = tf_forward(batch)
            loss = loss_fn(labels, batch)
            return loss, batch

        batch, hf_batch, tf_labels, hf_labels = self._get_batches()

        tf_model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, input_shape=batch.shape)
        loss_fn = CTCLoss(tf_model.config, batch.shape)

        hf_model = HFWav2Vec2ForCTC.from_pretrained(HF_MODEL_ID)

        tf_loss, tf_logits = compute_loss(batch, labels=tf_labels)
        with torch.no_grad():
            hf_out = hf_model(hf_batch, labels=hf_labels)

        hf_loss = hf_out["loss"].numpy()
        tf_loss = tf_loss.numpy()

        assert (
            tf_logits.shape == hf_out["logits"].shape
        ), "Oops, logits shape is not matching"

        logits_difference = np.max(hf_out["logits"].numpy() - tf_logits.numpy())
        assert np.allclose(
            hf_out["logits"].numpy(), tf_logits.numpy(), atol=0.004
        ), f"difference: {logits_difference}"

        assert np.allclose(
            tf_loss, hf_loss, atol=1e-3
        ), f"difference: {np.max(tf_loss - hf_loss)}"

    @partial(requires_lib, lib=["torch"])
    def test_conv_weight_norm(self):
        bsz = 2
        seqlen = 128
        c_in = 32
        filters = 16
        kernal_size = 3
        padding = 1
        num_groups = 2

        np.random.seed(SEED)
        array = np.random.uniform(size=(bsz, seqlen, c_in))
        tf_tensor = tf.convert_to_tensor(array, dtype=tf.float32)

        # `nn.Conv1d` accepts (batch_size, channels, seqlen)
        torch_tensor = torch.tensor(array, dtype=torch.float32).transpose(2, 1)

        tf_layer = Conv1DWithWeightNorm(
            filters, kernal_size, padding=padding, groups=num_groups
        )
        tf_layer(tf_tensor)  # build tensorflow weights

        torch_layer = nn.Conv1d(
            c_in, filters, kernal_size, padding=padding, groups=num_groups
        )
        torch_layer = nn.utils.weight_norm(torch_layer, dim=2)

        # torch & tensorflow weights should be equal
        torch_layer.weight_v.data = torch.tensor(
            np.transpose(tf_layer.variables[1].numpy(), axes=(2, 1, 0))
        )
        torch_layer.bias.data = torch.tensor(tf_layer.variables[0].numpy())
        torch_layer.weight_g.data = torch.tensor(
            np.transpose(tf_layer.variables[2].numpy(), axes=(2, 1, 0))
        )

        # forward pass
        with torch.no_grad():
            torch_out = torch_layer(torch_tensor).transpose(2, 1).numpy()
        tf_out = tf_layer(tf_tensor).numpy()

        assert np.allclose(
            torch_out, tf_out, atol=1e-4
        ), f"Difference: {torch_out} vs {tf_out}"


class TFhubTester(unittest.TestCase):
    def _get_batch(self):
        tf.random.set_seed(SEED)
        batch = tf.random.normal((2, 246000))
        attention_mask = np.ones(batch.shape, dtype=np.float32)
        attention_mask[0, -1000:] = attention_mask[1, -132:] = 0.0
        attention_mask = tf.constant(attention_mask, dtype=tf.float32)
        return batch, attention_mask

    def _test_hub_model(self, hub_id, tf_model):
        batch, _ = self._get_batch()
        tfhub_model = hub.KerasLayer(hub_id, trainable=False)
        tfhub_out = tf.function(tfhub_model, jit_compile=True)(batch).numpy()
        out = tf_model(batch).numpy()
        assert np.allclose(tfhub_out, out, atol=1e-3), f"Difference: {tfhub_out} vs {out}"

    def _test_hub_robust_model(self, hub_id, tf_model):
        batch, attention_mask = self._get_batch()
        tfhub_model = hub.KerasLayer(hub_id, trainable=False)
        tfhub_out = tf.function(tfhub_model, jit_compile=True)((batch, attention_mask)).numpy()
        out = tf_model(batch, attention_mask=attention_mask).numpy()
        assert np.allclose(tfhub_out, out, atol=1e-2), f"Difference: {tfhub_out} vs {out}"

    def test_wav2vec2_base(self):
        hub_id = "https://tfhub.dev/vasudevgupta7/wav2vec2/1"
        tf_model = Wav2Vec2Model.from_pretrained("vasudevgupta/gsoc-wav2vec2")
        self._test_hub_model(hub_id, tf_model)

    def test_wav2vec2_base_960h(self):
        hub_id = "https://tfhub.dev/vasudevgupta7/wav2vec2-960h/1"
        tf_model = Wav2Vec2ForCTC.from_pretrained("vasudevgupta/gsoc-wav2vec2-960h")
        self._test_hub_model(hub_id, tf_model)

    def test_wav2vec2_xlsr_53(self):
        hub_id = "src/wav2vec2_xlsr_53" # "https://tfhub.dev/vasudevgupta7/wav2vec2-xlsr-53/1"
        tf_model = Wav2Vec2Model.from_pretrained("vasudevgupta/gsoc-wav2vec2-xlsr-53")
        self._test_hub_robust_model(hub_id, tf_model)

    def test_wav2vec2_robust(self):
        hub_id = "src/wav2vec2_robust" # "https://tfhub.dev/vasudevgupta7/wav2vec2-robust/1"
        tf_model = Wav2Vec2Model.from_pretrained("vasudevgupta/gsoc-wav2vec2-robust")
        self._test_hub_robust_model(hub_id, tf_model)
