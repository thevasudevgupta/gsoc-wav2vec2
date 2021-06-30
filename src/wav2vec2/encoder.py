import tensorflow as tf

import numpy as np

from .tensorflow_addons import Conv1DWithWeightNorm


class TransformerAttention(tf.keras.layers.Layer):
    """Attention layer from `Attention Is All You Need`"""

    def __init__(self, hidden_size, num_heads, dropout=0.1, name="attention"):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout

        self.q = tf.keras.layers.Dense(hidden_size, name="q_proj")
        self.k = tf.keras.layers.Dense(hidden_size, name="k_proj")
        self.v = tf.keras.layers.Dense(hidden_size, name="v_proj")
        self.projection = tf.keras.layers.Dense(hidden_size, name="out_proj")

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, batch, training=False):
        head_size = batch.shape[2] // self.num_heads
        q_out = self._prepare_either_qkv(self.q(batch), head_size)
        k_out = self._prepare_either_qkv(self.k(batch), head_size)
        v_out = self._prepare_either_qkv(self.v(batch), head_size)

        q_out = q_out * head_size ** (-0.5)

        batch = self.get_context(q_out, k_out, v_out, training=training)
        batch = self.projection(batch)
        return batch

    def get_context(self, q_out, k_out, v_out, training=False):

        b, h, l, d = q_out.shape
        attn_scores = tf.matmul(q_out, k_out, transpose_b=True)  # "bhqd,bhkd->bhqk"

        attn_scores = self.dropout(
            tf.nn.softmax(attn_scores, axis=-1), training=training
        )
        context = tf.matmul(attn_scores, v_out)  # "bhll,bhld->bhld"
        context = tf.transpose(context, perm=(0, 2, 1, 3))
        return tf.reshape(context, (-1, l, h * d))

    def _prepare_either_qkv(self, tensor, head_size):
        bsz, seqlen, _ = tensor.shape
        tensor = tf.reshape(tensor, (-1, seqlen, self.num_heads, head_size))
        return tf.transpose(
            tensor, perm=(0, 2, 1, 3)
        )  # -> bsz, num_heads, seqlen, head_size

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
            }
        )
        return config


class TransformerLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size,
        num_heads,
        intermediate_size,
        layer_norm_eps=1e-5,
        is_gelu_approx=False,
        dropout=0.1,
        name=None,
    ):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.is_gelu_approx = is_gelu_approx
        self.dropout = dropout

        self.attention = TransformerAttention(
            hidden_size, num_heads, dropout=dropout, name="attention"
        )
        self.dropout = tf.keras.layers.Dropout(dropout)

        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps, name="layer_norm"
        )
        self.intermediate = tf.keras.layers.Dense(
            intermediate_size, name="feed_forward/intermediate_dense"
        )
        self.attn_output = tf.keras.layers.Dense(
            hidden_size, name="feed_forward/output_dense"
        )
        self.final_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps,
            name="final_layer_norm",
        )

    def call(self, batch, training=False):

        # self_attn
        residual = batch
        batch = self.attention(batch, training=training)
        batch = self.dropout(batch, training=training)
        batch = self.layer_norm(batch + residual)

        # ffn
        residual = batch
        batch = tf.nn.gelu(self.intermediate(batch), approximate=self.is_gelu_approx)
        batch = self.dropout(batch, training=training)
        batch = self.dropout(self.attn_output(batch), training=training)
        batch = self.final_layer_norm(batch + residual)

        return batch

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "intermediate_size": self.intermediate_size,
                "layer_norm_eps": self.layer_norm_eps,
                "is_gelu_approx": self.is_gelu_approx,
                "dropout": self.dropout,
            }
        )
        return config


class PositionalConvEmbedding(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size,
        num_conv_pos_embeddings,
        num_conv_pos_embedding_groups,
        is_gelu_approx=False,
        name="pos_conv_embed",
    ):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.is_gelu_approx = is_gelu_approx

        self.conv = Conv1DWithWeightNorm(
            hidden_size,
            num_conv_pos_embeddings,
            padding=num_conv_pos_embeddings // 2,
            groups=num_conv_pos_embedding_groups,
            name="conv",
        )
        self.is_padding_wrong = num_conv_pos_embeddings % 2 == 0

    def call(self, batch):
        batch = self.conv(batch)
        if self.is_padding_wrong:
            batch = batch[:, :-1, :]
        return tf.nn.gelu(batch, approximate=self.is_gelu_approx)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_conv_pos_embeddings": self.num_conv_pos_embeddings,
                "num_conv_pos_embedding_groups": self.num_conv_pos_embedding_groups,
                "is_gelu_approx": self.is_gelu_approx,
            }
        )
        return config


class Wav2Vec2Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        hidden_size,
        num_heads,
        num_layers,
        intermediate_size,
        num_conv_pos_embeddings,
        num_conv_pos_embedding_groups,
        layer_drop=0.1,
        dropout=0.1,
        layer_norm_eps=1e-5,
        is_gelu_approx=False,
        name="encoder",
    ):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.intermediate_size = intermediate_size
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.layer_drop = layer_drop
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.is_gelu_approx = is_gelu_approx

        self.pos_conv_embed = PositionalConvEmbedding(
            hidden_size,
            num_conv_pos_embeddings,
            num_conv_pos_embedding_groups,
            is_gelu_approx=is_gelu_approx,
            name="pos_conv_embed",
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps, name="layer_norm"
        )
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.layers = [
            TransformerLayer(
                hidden_size,
                num_heads,
                intermediate_size,
                layer_norm_eps=layer_norm_eps,
                is_gelu_approx=is_gelu_approx,
                dropout=dropout,
                name=f"layers/{i}",
            )
            for i in range(num_layers)
        ]

    def call(self, batch, training=False):
        batch = batch + self.pos_conv_embed(batch)
        batch = self.dropout(self.layer_norm(batch), training=training)
        for layer in self.layers:
            # layer_drop from [paper](https://arxiv.org/abs/1909.11556)
            drop_prob = np.random.uniform(0, 1)
            if training and (drop_prob < self.layer_drop):
                continue
            batch = layer(batch, training=training)
        return batch

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "intermediate_size": self.intermediate_size,
                "num_conv_pos_embeddings": self.num_conv_pos_embeddings,
                "num_conv_pos_embedding_groups": self.num_conv_pos_embedding_groups,
                "layer_drop": self.layer_drop,
                "dropout": self.dropout,
                "layer_norm_eps": self.layer_norm_eps,
                "is_gelu_approx": self.is_gelu_approx,
            }
        )
        return config
