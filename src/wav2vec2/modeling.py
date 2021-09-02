"""TensorFlow implementation of Wav2Vec2"""

import os
import logging
import subprocess
from dataclasses import replace
from typing import Optional

import tensorflow as tf

from huggingface_hub import ModelHubMixin

from .config import Wav2Vec2Config
from .encoder import Wav2Vec2Encoder
from .feature_extractor import FeatureExtractorLayer, FeatureProjection
from .spec_augment import apply_spec_augmentation

logger = logging.getLogger(__name__)


class TFKerasModel(tf.keras.Model):
    def save_pretrained(self, save_dir):
        """
        This method will save model weights and config in `save_directory`.
        """
        self.config.save_pretrained(save_dir)
        self.save_weights(os.path.join(save_dir, "tf_model.h5"))

    def push_to_hub(self, directory: str, model_id: str):
        """
        Use this method to push your model weights to HuggingFace Hub.

        Args:
            directory (:obj: `str`):
                directory where model weights are prensent.
            model_id (:obj: `str`):
                Name of the repositary in HuggingFace Hub you want to push to.
        """
        return ModelHubMixin.push_to_hub(directory, model_id=model_id)

    @classmethod
    def from_pretrained(cls, model_id, **config_kwargs) -> tf.keras.Model:
        """
        This will load model weights from the dictionary specified or download it from HuggingFace Hub
        if weights are not available locally.

        Args:
            model_id (:obj: `str`):
                Directory where weights are present or model_id if needs to be downloaded from HuggingFace Hub.
            config_kwargs (:obj: `dict`)
                Extra arguments will be passed to `Wav2Vec2Config`.

        Returns:
            Instance of `tf.keras.Model` initialized from trained weights.
        """

        save_dir = model_id
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            config_url = f"wget https://huggingface.co/{model_id}/resolve/main/config.json -P {save_dir}"
            model_url = f"wget https://huggingface.co/{model_id}/resolve/main/tf_model.h5 -P {save_dir}"

            print(
                f"Downloading model weights from https://huggingface.co/{model_id} ... ",
                end="",
            )
            try:
                for url in [config_url, model_url]:
                    subprocess.run(url.split(), check=True, stderr=subprocess.PIPE)
            except:
                raise ValueError(
                    f"Couldn't download model weights from https://huggingface.co/{model_id}"
                )
            print("Done")
        else:
            print(f"Loading weights locally from `{save_dir}`")

        input_shape = config_kwargs.pop("input_shape", (1, 2048))
        config = Wav2Vec2Config.from_json(os.path.join(save_dir, "config.json"))
        config = replace(config, **config_kwargs)
        model = cls(config, input_shape=input_shape)
        model.load_weights(os.path.join(save_dir, "tf_model.h5"))
        print("Total number of loaded variables:", len(model.variables))
        return model

    def _init(self, input_shape=None, is_robust=False, for_export=False):
        """Build Model weights using dummy inputs"""
        # call this at the end only
        if input_shape is None:
            input_shape = (1, 2048)
        dummy_input = tf.ones(input_shape, dtype=tf.float32)
        attention_mask = tf.ones(input_shape) if is_robust else None

        if for_export:
            self((dummy_input, attention_mask))
        else:
            try:
                # this operation doesn't work on CPU
                self.predict(dummy_input, attention_mask=attention_mask)
            except:
                # this operation will hang the TPU VM, hence prefer `.predict`
                self(dummy_input, attention_mask=attention_mask)


class Wav2Vec2Model(TFKerasModel):
    def __init__(self, config: Wav2Vec2Config, input_shape=(1, 246000), name="wav2vec2"):
        super().__init__(name=name)
        if not isinstance(config, Wav2Vec2Config):
            raise ValueError("`config` must be an instace of `Wave2Vec2Config`")

        self.config = config
        self.hidden_size = config.hidden_size
        self.is_robust = config.is_robust
        self.kernal_sizes = config.kernal_sizes
        self.strides = config.strides

        # spec-augmentation
        self.apply_spec_augment = config.apply_spec_augment
        self.mask_time_prob = config.mask_time_prob
        self.mask_time_length = config.mask_time_length

        num_feature_extractor_layers = len(config.filter_sizes)

        self.feature_extractor = [
            FeatureExtractorLayer(
                config.filter_sizes,
                config.kernal_sizes,
                config.strides,
                conv_bias=config.conv_bias,
                is_gelu_approx=config.is_gelu_approx,
                feature_extractor_norm_type=config.feature_extractor_norm_type,
                layer_id=i,
                name=f"feature_extractor/conv_layers/{i}",
            )
            for i in range(num_feature_extractor_layers)
        ]
        self.feature_projection = FeatureProjection(
            config.hidden_size,
            layer_norm_eps=config.layer_norm_eps,
            dropout=config.dropout,
            name="feature_projection",
        )
        self.encoder = Wav2Vec2Encoder(
            config.hidden_size,
            config.num_heads,
            config.num_layers,
            config.intermediate_size,
            config.num_conv_pos_embeddings,
            config.num_conv_pos_embedding_groups,
            survival_prob=config.survival_prob,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
            is_gelu_approx=config.is_gelu_approx,
            attention_norm_type=config.attention_norm_type,
            name="encoder",
        )

        if input_shape is not None:
            self._init(input_shape=input_shape, is_robust=config.is_robust)

    def build(self, input_shape):
        self.masked_spec_augment = self.add_weight(
            name="masked_spec_embed",
            shape=(self.hidden_size,),
            initializer="uniform",
            trainable=True,
        )

    def call(self, batch, attention_mask: Optional[tf.Tensor] = None, training=False):
        """
        Args:
            batch (:obj: `tf.Tensor`) of shape (batch_size, seqlen):
                Sound tensor obtained from `Wav2Vec2Processor.__call__`.
            attention_mask (:obj: `tf.Tensor`, `optional`) of shape (batch_size, seqlen):
                Don't pass `attention_mask` when working with checkpoints based on `wav2vec2-base`
                otherwise you should pass this argument.
            training (:obj: `bool`, `optional`):
                Whether to use model for training.

        Returns:
            Logits from the model of shape (batch_size, seqlen, hidden_dim).
        """
        if self.is_robust and attention_mask is None:
            logger.warning("You should pass `attention_mask` when working with Wav2Vec2 new checkpoints")
        elif not self.is_robust and attention_mask is not None:
            logger.warning("You should not pass `attention_mask` when working with checkpoints based on `wav2vec2-base`")

        batch = tf.expand_dims(batch, axis=-1)
        for feature_extractor_layer in self.feature_extractor:
            batch = feature_extractor_layer(batch)
        batch = self.feature_projection(batch, training=training)

        if training and self.apply_spec_augment:
            batch = apply_spec_augmentation(
                batch,
                self.masked_spec_augment,
                self.mask_time_prob,
                self.mask_time_length,
            )

        if attention_mask is not None:
            input_length = tf.reduce_sum(attention_mask, axis=-1)
            for kernal_size, stride in zip(self.kernal_sizes, self.strides):
                input_length = 1 + (input_length - kernal_size) // stride

            attention_mask = tf.sequence_mask(input_length, maxlen=batch.shape[1])

        batch = self.encoder(batch, attention_mask=attention_mask, training=training)
        return batch

    def freeze_feature_extractor(self):
        """This will freeze the feature extractor layers (Recommended to use for fine-tuning)."""
        for i in range(len(self.feature_extractor)):
            self.feature_extractor[i].trainable = False


class Wav2Vec2ForCTC(TFKerasModel):
    """Wave2Vec2 model with a CTC head."""

    def __init__(
        self, config: Wav2Vec2Config, input_shape=(1, 246000), name="wav2vec2-ctc"
    ):
        super().__init__(name=name)
        if not isinstance(config, Wav2Vec2Config):
            raise ValueError("`config` must be an instace of `Wave2Vec2Config`.")
        self.config = config
        self.pad_id = config.pad_id

        self.model = Wav2Vec2Model(config, input_shape=None, name="wav2vec2")
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        self.lm_head = tf.keras.layers.Dense(config.vocab_size, name="lm_head")

        self._init(input_shape=input_shape, is_robust=config.is_robust)

    def freeze_feature_extractor(self):
        """This will freeze the feature extractor layers (Recommended to use for fine-tuning)."""
        self.model.freeze_feature_extractor()

    def call(self, batch: tf.Tensor, attention_mask: Optional[tf.Tensor] = None, training=False):
        """
        Args:
            batch (:obj: `tf.Tensor`) of shape (batch_size, seqlen):
                Sound tensor obtained from `Wav2Vec2Processor.__call__`.
            attention_mask (:obj: `tf.Tensor`, `optional`) of shape (batch_size, seqlen):
                Don't pass `attention_mask` when working with checkpoints based on `wav2vec2-base`
                otherwise you should pass this argument.
            training (:obj: `bool`, `optional`):
                Whether to use model for training.
        Returns:
            Logits from the model of shape (batch_size, seqlen, vocab_size).
        """
        batch = self.model(batch, attention_mask=attention_mask, training=training)
        batch = self.dropout(batch, training=training)
        batch = self.lm_head(batch)
        return batch
