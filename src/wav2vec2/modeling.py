# __author__ = "Vasudev Gupta"
# __author_email__ = "7vasudevgupta@gmail.com"

"""TensorFlow implementation of Wav2Vec2"""

import os
import subprocess
from dataclasses import replace

import numpy as np
import tensorflow as tf
from huggingface_hub import ModelHubMixin

from .config import Wav2Vec2Config
from .tensorflow_addons import Conv1DWithWeightNorm, GroupNormalization


class TransformerAttention(tf.keras.layers.Layer):
    """Attention layer from `Attention Is All You Need`"""

    def __init__(self, config, name="attention"):
        super().__init__(name=name)
        self.num_heads = config.num_heads

        self.q = tf.keras.layers.Dense(config.hidden_size, name="q_proj")
        self.k = tf.keras.layers.Dense(config.hidden_size, name="k_proj")
        self.v = tf.keras.layers.Dense(config.hidden_size, name="v_proj")
        self.projection = tf.keras.layers.Dense(config.hidden_size, name="out_proj")

        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def call(self, batch, padding_mask, training=False):
        head_size = batch.shape[2] // self.num_heads
        q_out = self._prepare_either_qkv(self.q(batch), head_size)
        k_out = self._prepare_either_qkv(self.k(batch), head_size)
        v_out = self._prepare_either_qkv(self.v(batch), head_size)

        q_out = q_out * head_size ** (-0.5)

        # TODO: confirm padding_mask later
        if padding_mask is None:
            shape = batch.shape[:-1]
            padding_mask = tf.ones(shape, dtype=tf.bool)

        batch = self.get_context(
            q_out, k_out, v_out, padding_mask=padding_mask, training=training
        )
        batch = self.projection(batch)
        return batch

    @staticmethod
    def prepare_mask(padding_mask):
        mask_shape = padding_mask.shape + (padding_mask.shape[1],)
        attn_penalty = tf.constant(-10000, dtype=tf.float32)
        padding_mask = tf.broadcast_to(~padding_mask, mask_shape)
        return tf.cast(padding_mask, tf.float32) * attn_penalty

    def get_context(self, q_out, k_out, v_out, padding_mask=None, training=False):

        b, h, l, d = q_out.shape
        attn_scores = tf.matmul(q_out, k_out, transpose_b=True)  # "bhqd,bhkd->bhqk"

        if padding_mask is not None:
            attn_scores += self.prepare_mask(padding_mask)

        attn_scores = self.dropout(
            tf.nn.softmax(attn_scores, axis=-1), training=training
        )
        context = tf.matmul(attn_scores, v_out)  # "bhll,bhld->bhld"
        context = tf.transpose(context, perm=(0, 2, 1, 3))
        return tf.reshape(context, (b, l, h * d))

    def _prepare_either_qkv(self, tensor, head_size):
        bsz, seqlen, _ = tensor.shape
        tensor = tf.reshape(tensor, (bsz, seqlen, self.num_heads, head_size))
        return tf.transpose(
            tensor, perm=(0, 2, 1, 3)
        )  # -> bsz, num_heads, seqlen, head_size


class FeatureExtractorLayer(tf.keras.layers.Layer):
    def __init__(self, config, layer_id=0, name=None):
        super().__init__(name=name)
        self.is_gelu_approx = config.is_gelu_approx
        conv_dim = config.filter_sizes[layer_id]
        kernal_size = config.kernal_sizes[layer_id]
        stride = config.strides[layer_id]

        self.conv_layer = tf.keras.layers.Conv1D(
            conv_dim,
            kernal_size,
            strides=stride,
            use_bias=config.conv_bias,
            name="conv",
        )

        if layer_id == 0:
            self.layer_norm = GroupNormalization(
                conv_dim,
                axis=-1,
                name="layer_norm",
                epsilon=1e-5,
            )
        else:
            self.layer_norm = None

    def call(self, batch):
        batch = self.conv_layer(batch)
        if self.layer_norm is not None:
            batch = self.layer_norm(batch)
        batch = tf.nn.gelu(batch, approximate=self.is_gelu_approx)
        return batch


class FeatureProjection(tf.keras.layers.Layer):
    def __init__(self, config, name="feature_projection"):
        super().__init__(name=name)
        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layer_norm"
        )
        self.projection = tf.keras.layers.Dense(config.hidden_size, name="projection")
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def call(self, batch, training=False):
        batch = self.layer_norm(batch)
        batch = self.projection(batch)
        return self.dropout(batch, training=training)


class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, config, name=None):
        super().__init__(name=name)
        self.is_gelu_approx = config.is_gelu_approx

        self.attention = TransformerAttention(config, name="attention")
        self.dropout = tf.keras.layers.Dropout(config.dropout)

        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layer_norm"
        )
        self.intermediate = tf.keras.layers.Dense(
            config.intermediate_size, name="feed_forward/intermediate_dense"
        )
        self.attn_output = tf.keras.layers.Dense(
            config.hidden_size, name="feed_forward/output_dense"
        )
        self.final_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps,
            name="final_layer_norm",
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
    def __init__(self, config, name="pos_conv_embed"):
        super().__init__(name=name)
        self.is_gelu_approx = config.is_gelu_approx

        self.conv = Conv1DWithWeightNorm(
            config.hidden_size,
            config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
            name="conv",
        )
        self.is_padding_wrong = config.num_conv_pos_embeddings % 2 == 0

    def call(self, batch):
        batch = self.conv(batch)
        if self.is_padding_wrong:
            batch = batch[:, :-1, :]
        return tf.nn.gelu(batch, approximate=self.is_gelu_approx)


class Wav2Vec2Encoder(tf.keras.layers.Layer):
    def __init__(self, config, name="encoder"):
        super().__init__(name=name)
        self.layer_drop = config.layer_drop

        self.pos_conv_embed = PositionalConvEmbedding(config, name="pos_conv_embed")
        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=config.layer_norm_eps, name="layer_norm"
        )
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.layers = [
            TransformerLayer(config, name=f"layers/{i}")
            for i in range(config.num_layers)
        ]

    def call(self, batch, padding_mask, training=False):
        pos_embed = self.pos_conv_embed(batch)
        batch += pos_embed
        batch = self.dropout(self.layer_norm(batch), training=training)
        for layer in self.layers:
            # layer_drop from [paper](https://arxiv.org/abs/1909.11556)
            drop_prob = np.random.uniform(0, 1)
            if training and (drop_prob < self.layer_drop):
                continue
            batch = layer(batch, padding_mask, training=training)
        return batch


class TFKerasModel(tf.keras.Model):
    def save_pretrained(self, save_dir):
        self.config.save_pretrained(save_dir)
        self.save_weights(os.path.join(save_dir, "tf_model.h5"))

    def push_to_hub(self, directory, model_id):
        return ModelHubMixin.push_to_hub(directory, model_id=model_id)

    @classmethod
    def from_pretrained(cls, model_id, **config_kwargs):
        """Model has to be in public repo"""

        save_dir = model_id.split("/")[-1]
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            config_url = f"wget https://huggingface.co/{model_id}/resolve/main/config.json -P {save_dir}"
            model_url = f"wget https://huggingface.co/{model_id}/resolve/main/tf_model.h5 -P {save_dir}"

            try:
                for url in [config_url, model_url]:
                    subprocess.run(url.split(), check=True, stderr=subprocess.PIPE)
            except:
                raise ValueError(f"Couldn't download model weights from {model_url}")
        else:
            print(f"Loading weights locally from `{save_dir}`")

        input_shape = config_kwargs.pop("input_shape")
        config = Wav2Vec2Config.from_json(os.path.join(save_dir, "config.json"))
        config = replace(config, **config_kwargs)
        model = cls(config, input_shape=input_shape)
        model.load_weights(os.path.join(save_dir, "tf_model.h5"))
        print("Total number of loaded variables:", len(model.variables))
        return model

    def _init(self, input_shape=None):
        """Build Model weights using dummy inputs"""
        # call this at the end only
        dummy_input = tf.ones(input_shape, dtype=tf.float32)
        return self(dummy_input)


class Wav2Vec2Model(TFKerasModel):
    def __init__(self, config: Wav2Vec2Config, name="wav2vec2"):
        super().__init__(name=name)
        if not isinstance(config, Wav2Vec2Config):
            raise ValueError("`config` must be an instace of `Wave2Vec2Config`")

        num_feature_extractor_layers = len(config.filter_sizes)

        self.feature_extractor = [
            FeatureExtractorLayer(
                config, layer_id=i, name=f"feature_extractor/conv_layers/{i}"
            )
            for i in range(num_feature_extractor_layers)
        ]
        self.feature_projection = FeatureProjection(config, name="feature_projection")
        self.encoder = Wav2Vec2Encoder(config, name="encoder")

    def call(self, batch, padding_mask=None, training=False):
        batch = tf.expand_dims(batch, axis=-1)
        for feature_extractor_layer in self.feature_extractor:
            batch = feature_extractor_layer(batch)
        batch = self.feature_projection(batch, training=training)

        # TODO: apply spec-augment to batch later (useful for training only)

        batch = self.encoder(batch, padding_mask=padding_mask, training=training)
        return batch


class Wav2Vec2ForCTC(TFKerasModel):
    """Wave2Vec2 model with CTC/LM head"""

    def __init__(
        self, config: Wav2Vec2Config, input_shape=(1, 2048), name="wav2vec-ctc"
    ):
        super().__init__(name=name)
        if not isinstance(config, Wav2Vec2Config):
            raise ValueError("`config` must be an instace of `Wave2Vec2Config`")
        self.config = config

        self.model = Wav2Vec2Model(config, name="wav2vec2")
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.lm_head = tf.keras.layers.Dense(config.vocab_size, name="lm_head")

        self._init(input_shape=input_shape)

    def call(self, batch, padding_mask=None, training=False):
        # batch - (bsz, seqlen)
        batch = self.model(batch, padding_mask=padding_mask, training=training)
        batch = self.dropout(batch, training=training)
        batch = self.lm_head(batch)

        # TODO: implement loss
        loss = None

        return {"loss": loss, "logits": batch}
