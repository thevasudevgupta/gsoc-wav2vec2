# __author__ = "Vasudev Gupta"
# __author_email__ = "7vasudevgupta@gmail.com"

import numpy as np
import tensorflow as tf
import transformers
from tqdm.auto import tqdm

from wav2vec2 import Wav2Vec2Config, Wav2Vec2ForCTC

PREFIX = "wav2vec-ctc/"
SUFFIX = ":0"
MAPPING = (
    ("layer_norm.weight", "layer_norm/gamma"),
    ("layer_norm.bias", "layer_norm.beta"),
    ("weight", "kernel"),
    (".", "/"),
)

# TODO: fix later
KEYS_TO_IGNORE = [
    "wav2vec2.masked_spec_embed",
    "wav2vec2.encoder.pos_conv_embed.conv.bias",
    "wav2vec2.encoder.pos_conv_embed.conv.weight_g",
    "wav2vec2.encoder.pos_conv_embed.conv.weight_v",
]


def replace(k):
    for hf_v, tf_v in MAPPING:
        k = k.replace(hf_v, tf_v)
    return PREFIX + k + SUFFIX


def get_tf_pretrained_model(config, hf_model_id: str):
    from tensorflow.python.keras import backend as K

    tf_model = Wav2Vec2ForCTC(config)
    hf_model = transformers.Wav2Vec2ForCTC.from_pretrained(hf_model_id)

    hf_state_dict = hf_model.state_dict()

    tf_variables = tf_model.variables
    tf_variables_dict = {}
    for v in tf_variables:
        tf_variables_dict[v.name] = v

    tf_weights = []
    extra_keys = []
    for k in tqdm(hf_state_dict, desc="hf -> tf"):
        if k in KEYS_TO_IGNORE:
            continue
        new_k = replace(k)
        print(k, "->", new_k)
        array = hf_state_dict[k].numpy()
        if "kernel" in new_k:
            array = np.transpose(array)

        if new_k not in tf_variables_dict.keys():
            extra_keys.append(k)
            print(f"SKIPPING {k}")
            continue

        tf_weights.append((tf_variables_dict[new_k], array))

    print("EXTRA KEYS:", extra_keys)

    K.batch_set_value(tf_weights)
    # let's check if forward working properly
    tf_model(tf.ones((1, 1024), dtype=tf.float32))

    return tf_model


if __name__ == "__main__":
    config = Wav2Vec2Config()
    tf_model = get_tf_pretrained_model(config, "facebook/wav2vec2-base-960h")
    tf_model.save_pretrained("wav2vec2-base-960h")

