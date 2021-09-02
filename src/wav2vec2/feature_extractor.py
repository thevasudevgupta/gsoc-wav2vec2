import tensorflow as tf

from .tensorflow_addons import GroupNormalization


class FeatureExtractorLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        filter_sizes,
        kernal_sizes,
        strides,
        conv_bias=False,
        is_gelu_approx=False,
        layer_id=0,
        feature_extractor_norm_type="group",
        name=None,
    ):
        super().__init__(name=name)
        self.filter_sizes = filter_sizes
        self.kernal_sizes = kernal_sizes
        self.strides = strides
        self.conv_bias = conv_bias
        self.is_gelu_approx = is_gelu_approx
        self.layer_id = layer_id
        self.feature_extractor_norm_type = feature_extractor_norm_type

        conv_dim = filter_sizes[layer_id]
        kernal_size = kernal_sizes[layer_id]
        stride = strides[layer_id]

        self.conv_layer = tf.keras.layers.Conv1D(
            conv_dim,
            kernal_size,
            strides=stride,
            use_bias=conv_bias,
            name="conv",
        )

        self.layer_norm = None
        if self.feature_extractor_norm_type == "group":
            if layer_id == 0:
                self.layer_norm = GroupNormalization(
                    conv_dim,
                    axis=-1,
                    name="layer_norm",
                    epsilon=1e-5,
                )
        elif self.feature_extractor_norm_type == "layer":
            # TODO: check value of axis
            self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5, name="layer_norm")
        else:
            raise NotImplementedError

    def call(self, batch):
        batch = self.conv_layer(batch)
        if self.layer_norm is not None:
            batch = self.layer_norm(batch)
        batch = tf.nn.gelu(batch, approximate=self.is_gelu_approx)
        return batch

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filter_sizes": self.filter_sizes,
                "kernal_sizes": self.kernal_sizes,
                "strides": self.strides,
                "conv_bias": self.conv_bias,
                "is_gelu_approx": self.is_gelu_approx,
                "layer_id": self.layer_id,
                "feature_extractor_norm_type": self.feature_extractor_norm_type,
            }
        )
        return config


class FeatureProjection(tf.keras.layers.Layer):
    def __init__(
        self, hidden_size, layer_norm_eps=1e-5, dropout=0.1, name="feature_projection"
    ):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout

        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=layer_norm_eps, name="layer_norm"
        )
        self.projection = tf.keras.layers.Dense(hidden_size, name="projection")
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, batch, training=False):
        batch = self.layer_norm(batch)
        batch = self.projection(batch)
        return self.dropout(batch, training=training)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_size": self.hidden_size,
                "layer_norm_eps": self.layer_norm_eps,
                "dropout": self.dropout,
            }
        )
        return config
