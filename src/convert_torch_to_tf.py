import tensorflow as tf
import transformers

import numpy as np
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

# fill-in PyTorch keys to ignore below
KEYS_TO_IGNORE = []

SPECIAL_MAPPING = {
    "wav2vec2.encoder.pos_conv_embed.conv.weight_g": "wav2vec-ctc/wav2vec2/encoder/pos_conv_embed/conv/weight_g:0",
    "wav2vec2.encoder.pos_conv_embed.conv.weight_v": "wav2vec-ctc/wav2vec2/encoder/pos_conv_embed/conv/weight_v:0",
}


def replace(k: str) -> str:
    """
    Converts PyTorch state_dict keys to TensorFlow varible name
    """
    for hf_v, tf_v in MAPPING:
        k = k.replace(hf_v, tf_v)
    return PREFIX + k + SUFFIX


def get_tf_pretrained_model(
    config: Wav2Vec2Config, hf_model_id: str, verbose: bool = False
) -> Wav2Vec2ForCTC:
    """
    Converts HF PyTorch weights to TensorFlow compatible weights

    Args:
        config (:obj: `Wav2Vec2Config`):
            Configuration of TF model
        hf_model_id (:obj: `str`):
            model_id of HuggingFace PyTorch model

    Returns:
        Instance of `Wav2Vec2ForCTC` loaded with pre-trained weights
    """

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
        new_k = SPECIAL_MAPPING[k] if k in SPECIAL_MAPPING.keys() else replace(k)
        if verbose:
            print(k, "->", new_k)

        if new_k not in tf_variables_dict.keys():
            extra_keys.append(k)
            print(f"SKIPPING {k}")
            continue

        array = hf_state_dict[k].numpy()

        # transpose the PyTorch weights for correct loading in TF-2
        # Weights corresponding to `SPECIAL_MAPPING` are 3D array while other weights are 2D
        # so we need to separate weights first & do special transpose on 3D weights
        if k in SPECIAL_MAPPING.keys():
            array = np.transpose(array, axes=(2, 1, 0))
        elif "kernel" in new_k:
            array = np.transpose(array)

        tf_weights.append((tf_variables_dict[new_k], array))

    print("EXTRA KEYS:\n", extra_keys)

    tf.keras.backend.batch_set_value(tf_weights)

    return tf_model


if __name__ == "__main__":
    config = Wav2Vec2Config()
    tf_model = get_tf_pretrained_model(
        config, "facebook/wav2vec2-base-960h", verbose=True
    )
    tf_model.save_pretrained("wav2vec2-base-960h")
