# __author__ = "Vasudev Gupta"
# __author_email__ = "7vasudevgupta@gmail.com"

"""TensorFlow implementation of Wav2Vec2"""

import os

import tensorflow as tf
from huggingface_hub import ModelHubMixin

from .config import Wav2Vec2Config


class TransformerAttention(tf.keras.layer.Layer):
    """Attention layer from `Attention Is All You Need`"""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads

        self.q = tf.keras.layer.Dense(config.hidden_size)
        self.k = tf.keras.layer.Dense(config.hidden_size)
        self.v = tf.keras.layer.Dense(config.hidden_size)
        self.attn_fn = tf.keras.layers.Attention(dropout=config.dropout)
        self.projection = tf.keras.layers.Dense(config.hidden_size)

    def call(self, batch, padding_mask):
        q_out = self._prepare_either_qkv(self.q(batch))
        k_out = self._prepare_either_qkv(self.k(batch))
        v_out = self._prepare_either_qkv(self.v(batch))

        batch = self.attn_fn(
            [q_out, v_out, k_out],
            mask=[padding_mask, padding_mask],
            training=self.training,
        )
        batch = self.projection(batch)
        return batch

    def _prepare_either_qkv(self, tensor):
        bz, seqlen, hidden_size = tensor.shape
        head_size = hidden_size // self.num_heads
        tensor = tf.reshape(tensor, (bz, seqlen, self.num_heads, head_size))
        return tf.reshape(
            tf.transpose(tensor, perm=(0, 2, 1, 3)),
            (bz * self.num_heads, seqlen, head_size),
        )


class FeatureExtractorLayer(tf.keras.layers.Layer):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.is_gelu_approx = config.is_gelu_approx
        filters = config.filter_sizes
        kernal_sizes = config.kernal_sizes
        strides = config.strides

        self.conv_layer = tf.keras.layer.Conv1d(
            filters[layer_id], kernal_sizes[layer_id], strides=strides[layer_id]
        )

        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, batch):
        batch = self.conv_layer(batch)
        batch = self.layer_norm(batch)
        batch = tf.nn.gelu(batch, approximate=self.is_gelu_approx)
        return batch


class FeatureProjector(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.projection = tf.keras.layers.Dense(config.hidden_size)
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def call(self, batch):
        batch = self.layer_norm(batch)
        batch = self.projection(batch)
        return self.dropout(batch)


class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.is_gelu_approx = config.is_gelu_approx

        self.attention = TransformerAttention(config)
        self.intermediate = tf.keras.layers.Dense(self.intermediate_size)
        self.output = tf.keras.layers.Dense(self.hidden_size)
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def call(self, batch, padding_mask):
        batch = self.attention(batch, padding_mask)
        batch = tf.nn.gelu(self.intermediate(batch), approximate=self.is_gelu_approx)
        batch = self.dropout(self.output(batch))
        return batch



class TFKerasModel(tf.keras.Model):
    def save_pretrained(self, save_dir):
        self.config.save_pretrained(save_dir)
        self.save_weights(os.path.join(save_dir, "tf_model.h5"))

    def push_to_hub(self, directory, model_id):
        ModelHubMixin.push_to_hub(directory, model_id=model_id)

    @classmethod
    def from_pretrained(cls):
        raise NotImplementedError


class Wav2Vec2(TFKerasModel):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        assert isinstance(
            config, Wav2Vec2Config
        ), "`config` must be an instace of `Wave2Vec2Config`"

        num_feature_extractor_layers = len(config.filter_sizes)

        self.feature_extractor = [
            FeatureExtractorLayer(config, layer_id=i)
            for i in range(num_feature_extractor_layers)
        ]
        self.feature_projector = FeatureProjector(config)
        self.transformer = [
            TransformerLayer(config) for _ in range(len(config.num_layers))
        ]

    def call(self, batch, padding_mask):
        batch = self.feature_extractor(batch)
        batch = self.feature_projector(batch)
        batch = self.transformer(batch, padding_mask)
        return batch


class Wav2Vec2ForCTC(TFKerasModel):
    """Wave2Vec2 model with CTC/LM head"""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        assert isinstance(
            config, Wav2Vec2Config
        ), "`config` must be an instace of `Wave2Vec2Config`"
        self.config = config

        self.model = Wav2Vec2(config)
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.lm_head = tf.keras.layers.Dense(config.vocab_size)

    def call(self, batch):
        batch = self.dropout(self.model(batch))
        return self.lm_head(batch)
