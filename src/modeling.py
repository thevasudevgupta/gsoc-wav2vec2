# __author__ = "Vasudev Gupta"
# __author_email__ = "7vasudevgupta@gmail.com"

"""TensorFlow implementation of Wave2Vec2"""

import tensorflow as tf


class Wave2Vec2Attention(tf.keras.layer.Layer):
    """Attention layer from `Attention Is All You Need`"""
    def __init__(self, config):
        assert config.hidden_size % config.num_heads == 0, "Hidden size must be perfect multiple of num_heads"
        head_size = config.hidden_size // config.num_heads
        # self.q = tf.keras.layer.Dense(head_size)
        # self.k = 
        self.self_attn = tf.keras.layers.Attention(dropout=config.dropout)
        self.intermediate = tf.keras.layers.Dense(self.intermediate_size)
        self.output = tf.keras.layers.Dense(self.hidden_size)

    def call(self, batch):
        batch = self.self_attn(batch)


class Model(tf.keras.Model):
    def save_to_disk(self, save_dir):
        self.config.save_to_disk(save_dir)
        self.save_weights(os.path.join(save_dir, "tf_model.h5"))


class Wave2Vec2(Model):
    """Wave2Vec2 model with no head"""
    def __init__(self, config):
        super().__init__()
        # feature extractor
        # feature projector
        # encoder
            # encoder layer
                # self-attention
                # ffn
        pass

class Wave2Vec2ForCTC(Model):
    """Wave2Vec2 model with CTC/LM head"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Wave2Vec2(config)
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.lm_head = tf.keras.layers.Dense(config.vocab_size)

    def call(self, batch):
        batch = self.dropout(self.model(batch))
        return self.lm_head(batch)
