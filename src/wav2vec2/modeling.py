# __author__ = "Vasudev Gupta"
# __author_email__ = "7vasudevgupta@gmail.com"

"""TensorFlow implementation of Wav2Vec2"""

import os
import subprocess

import numpy as np
import tensorflow as tf
from huggingface_hub import ModelHubMixin

# TODO: remove this dependacy at the very end
import tensorflow_addons as tfa

from .config import Wav2Vec2Config


class TransformerAttention(tf.keras.layers.Layer):
    """Attention layer from `Attention Is All You Need`"""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads

        self.q = tf.keras.layers.Dense(config.hidden_size)
        self.k = tf.keras.layers.Dense(config.hidden_size)
        self.v = tf.keras.layers.Dense(config.hidden_size)
        self.attn_fn = tf.keras.layers.Attention(dropout=config.dropout)
        self.projection = tf.keras.layers.Dense(config.hidden_size)

    def call(self, batch, padding_mask, training=False):
        bz, seqlen, hidden_size = batch.shape
        head_size = hidden_size // self.num_heads
        q_out = self._prepare_either_qkv(self.q(batch), head_size)
        k_out = self._prepare_either_qkv(self.k(batch), head_size)
        v_out = self._prepare_either_qkv(self.v(batch), head_size)

        batch = self.attn_fn(
            [q_out, v_out, k_out],
            mask=[padding_mask, padding_mask],
            training=training,
        )
        batch = tf.reshape(batch, (bz, self.num_heads, seqlen, head_size))
        batch = tf.transpose(batch, perm=(bz, seqlen, self.num_heads, head_size))
        batch = self.projection(batch)
        return batch

    def _prepare_either_qkv(self, tensor, head_size):
        bz, seqlen, _ = tensor.shape
        tensor = tf.reshape(tensor, (bz, seqlen, self.num_heads, head_size))
        tensor = tf.reshape(tensor, (bz * self.num_heads, seqlen, head_size))
        return tf.transpose(tensor, perm=(0, 2, 1, 3))


class FeatureExtractorLayer(tf.keras.layers.Layer):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.is_gelu_approx = config.is_gelu_approx
        conv_dim = config.filter_sizes[layer_id]
        kernal_size = config.kernal_sizes[layer_id]
        stride = config.strides[layer_id]

        self.layer_norm = None
        if layer_id == 0:
            self.conv_layer = tf.keras.layers.Conv1D(
                conv_dim,
                kernal_size,
                strides=stride,
                use_bias=config.conv_bias,
            )
            # TODO: correct axis during inference
            self.layer_norm = tfa.layers.GroupNormalization(conv_dim, axis=1)
        else:
            self.conv_layer = tf.keras.layers.Conv1D(
                conv_dim,
                kernal_size,
                strides=stride,
                use_bias=config.conv_bias,
            )

    def call(self, batch):
        batch = self.conv_layer(batch)
        if self.layer_norm is not None:
            batch = self.layer_norm(batch)
        batch = tf.nn.gelu(batch, approximate=self.is_gelu_approx)
        return batch


class FeatureProjection(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)
        self.projection = tf.keras.layers.Dense(config.hidden_size)
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def call(self, batch, training=False):
        batch = self.layer_norm(batch)
        batch = self.projection(batch)
        return self.dropout(batch, training=training)


class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.is_gelu_approx = config.is_gelu_approx

        self.attention = TransformerAttention(config)
        self.dropout = tf.keras.layers.Dropout(config.dropout)

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)
        self.intermediate = tf.keras.layers.Dense(config.intermediate_size)
        self.attn_output = tf.keras.layers.Dense(config.hidden_size)
        self.final_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps
        )

    def call(self, batch, padding_mask, training=False):
        # self_attn
        residual = batch
        batch = self.attention(batch, padding_mask, training=training)
        batch = self.dropout(batch, training=training)
        batch = self.layer_norm(batch + residual)

        # ffn
        residual = batch
        batch = tf.nn.gelu(self.intermediate(batch), approximate=self.is_gelu_approx)
        batch = self.dropout(batch, training=training)
        batch = self.dropout(self.attn_output(batch), training=training)
        batch = self.final_layer_norm(batch + residual)

        return batch


class PositionalConvEmbedding(tf.keras.layers.Layer):
    def __init__(self, config):
        self.is_gelu_apporx = config.is_gelu_approx
        # TODO: checkout padding in conv
        self.conv = tf.keras.layers.Conv1D(config.hidden_size, config.num_conv_pos_embeddings, strides=config.num_conv_pos_embeddings, groups=config.num_conv_pos_embedding_groups)
        # TODO: checkout weight norm
        # self.weight_norm =
        self.num_pad_remove = 1 if config.num_conv_pos_embeddings % 2 == 0 else 0

    def call(self, batch):
        batch = self.conv(batch)
        if self.num_pad_remove > 0:
            batch = batch[:, :, :, -self.num_pad_remove]
        batch = tf.nn.gelu(batch, approximate=self.is_gelu_approx)
        return batch


class Wav2Vec2Encoder(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.layer_drop = config.layer_drop

        # self.pos_conv_embed = PositionalConvEmbedding(config)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.layers = [TransformerLayer(config) for _ in range(config.num_layers)]

    def call(self, batch, padding_mask, training=False):
        # pos_embed = self.pos_conv_embed(batch)
        # batch += pos_embed
        batch = self.dropout(self.layer_norm(batch), training=training)
        for layer in self.layers:
            # layer_drop from [paper](https://arxiv.org/abs/1909.11556)
            drop_prob = np.random.uniform(0, 1)
            if training and (drop_prob < self.layer_drop):
                continue
            batch = layer(batch, padding_mask, training=training)
        return batch


class TFKerasModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # self.init()

    def save_pretrained(self, save_dir):
        self.config.save_pretrained(save_dir)
        self.save_weights(os.path.join(save_dir, "tf_model.h5"))

    def push_to_hub(self, directory, model_id):
        return ModelHubMixin.push_to_hub(directory, model_id=model_id)

    @classmethod
    def from_pretrained(cls, model_id, *args, **kwargs):
        """Model has to be in public repo"""

        save_dir = model_id.split("/")[-1]
        os.makedirs(save_dir, exist_ok=True)
        config_url = f"wget https://huggingface.co/{model_id}/resolve/main/config.json -P {save_dir}"
        model_url = f"wget https://huggingface.co/{model_id}/resolve/main/tf_model.h5 -P {save_dir}"

        try:
            for url in [config_url, model_url]:
                subprocess.run(url.split(), check=True, stderr=subprocess.PIPE)
        except:
            raise ValueError(f"Couldn't download model weights from {model_url}")

        config = Wav2Vec2Config.from_json(os.path.join(save_dir, "config.json"))
        model = cls(config, *args, **kwargs)
        model.load_weights(os.path.join(save_dir, "tf_model.h5"))
        return model

    def init(self, input_shape=(1, 128, 1)):
        """Build Model weights using dummy inputs"""
        dummy_input = tf.ones(input_shape, dtype=tf.int32)
        return self(dummy_input, dummy_input)


class Wav2Vec2Model(TFKerasModel):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        if not isinstance(config, Wav2Vec2Config):
            raise ValueError("`config` must be an instace of `Wave2Vec2Config`")

        num_feature_extractor_layers = len(config.filter_sizes)

        self.feature_extractor = [
            FeatureExtractorLayer(config, layer_id=i)
            for i in range(num_feature_extractor_layers)
        ]
        self.feature_projection = FeatureProjection(config)
        self.encoder = Wav2Vec2Encoder(config)

    def call(self, batch, padding_mask, training=False):
        for feature_extractor_layer in self.feature_extractor:
            batch = feature_extractor_layer(batch)
        # TODO: checkout transpose behavious here
        batch = self.feature_projection(batch, training=training)
        # TODO: apply spec-augment to batch later (useful for training only)
        batch = self.encoder(batch, padding_mask=padding_mask, training=training)
        return batch


class Wav2Vec2ForCTC(TFKerasModel):
    """Wave2Vec2 model with CTC/LM head"""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        if not isinstance(config, Wav2Vec2Config):
            raise ValueError("`config` must be an instace of `Wave2Vec2Config`")
        self.config = config

        self.model = Wav2Vec2Model(config)
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.lm_head = tf.keras.layers.Dense(config.vocab_size)

    def call(self, batch, padding_mask, training=False):
        batch = self.model(batch, padding_mask=padding_mask, training=training)
        batch = self.dropout(batch, training=training)
        batch = self.lm_head(batch)

        # TODO: implement loss
        loss = None

        return {"loss": loss, "logits": batch}
