import argparse
from typing import Union

import tensorflow as tf
import transformers

import numpy as np
from tqdm.auto import tqdm
from wav2vec2 import Wav2Vec2Config, RobustWav2Vec2Config, Wav2Vec2ForCTC, Wav2Vec2Model


SUFFIX = ":0"
MAPPING = (
    ("layer_norm.weight", "layer_norm/gamma"),
    ("layer_norm.bias", "layer_norm.beta"),
    ("weight", "kernel"),
    (".", "/"),
)

# fill-in PyTorch keys to ignore below
KEYS_TO_IGNORE = []

ACCEPTABLE_HF_IDS = ["facebook/wav2vec2-base-960h", "facebook/wav2vec2-base", "facebook/wav2vec2-large-robust", "facebook/wav2vec2-large-xlsr-53"]

PREFIX_WITH_HEAD = "wav2vec2-ctc/"
SPECIAL_MAPPING_WITH_HEAD = {
    "wav2vec2.encoder.pos_conv_embed.conv.weight_g": f"{PREFIX_WITH_HEAD}wav2vec2/encoder/pos_conv_embed/conv/weight_g:0",
    "wav2vec2.encoder.pos_conv_embed.conv.weight_v": f"{PREFIX_WITH_HEAD}wav2vec2/encoder/pos_conv_embed/conv/weight_v:0",
}

PREFIX_WITHOUT_HEAD = "wav2vec2/"
SPECIAL_MAPPING_WITHOUT_HEAD = {
    "encoder.pos_conv_embed.conv.weight_g": f"{PREFIX_WITHOUT_HEAD}encoder/pos_conv_embed/conv/weight_g:0",
    "encoder.pos_conv_embed.conv.weight_v": f"{PREFIX_WITHOUT_HEAD}encoder/pos_conv_embed/conv/weight_v:0",
}


def replace(k: str, prefix) -> str:
    """
    Converts PyTorch state_dict keys to TensorFlow varible name.
    """
    for hf_v, tf_v in MAPPING:
        k = k.replace(hf_v, tf_v)
    return prefix + k + SUFFIX


def get_tf_pretrained_model(
    config: Wav2Vec2Config,
    hf_model_id: str,
    verbose=False,
    with_lm_head=True,
) -> Union[Wav2Vec2ForCTC, Wav2Vec2Model]:
    """
    Converts HuggingFace PyTorch weights to TensorFlow compatible weights.

    Args:
        config (:obj: `Wav2Vec2Config`):
            Configuration of TF model.
        hf_model_id (:obj: `str`):
            model_id of HuggingFace PyTorch model.
        with_lm_head (:obj: `bool`, default=True):
            Whether to return Wav2Vec2ForCTC or Wav2Vec2Model

    Returns:
        Instance of `Wav2Vec2ForCTC` loaded with pre-trained weights.
    """
    assert hf_model_id in ACCEPTABLE_HF_IDS, f"{hf_model_id} is not acceptable"

    if with_lm_head:
        tf_model = Wav2Vec2ForCTC(config)
        prefix = PREFIX_WITH_HEAD
        hf_model = transformers.Wav2Vec2ForCTC.from_pretrained(hf_model_id)
    else:
        tf_model = Wav2Vec2Model(config)
        tf_model._init(input_shape=(1, 2048))
        prefix = PREFIX_WITHOUT_HEAD
        hf_model = transformers.Wav2Vec2Model.from_pretrained(hf_model_id)

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

        if k in SPECIAL_MAPPING_WITH_HEAD or k in SPECIAL_MAPPING_WITHOUT_HEAD:
            new_k = (
                SPECIAL_MAPPING_WITH_HEAD[k]
                if with_lm_head
                else SPECIAL_MAPPING_WITHOUT_HEAD[k]
            )
        else:
            new_k = replace(k, prefix=prefix)

        if new_k not in tf_variables_dict.keys():
            extra_keys.append(k)
            print(f"SKIPPING {k}")
            continue

        if verbose:
            print(k, "->", new_k)

        array = hf_state_dict[k].numpy()

        # transpose the PyTorch weights for correct loading in TF-2
        # Weights corresponding to `SPECIAL_MAPPING` are 3D array while other weights are 2D
        # so we need to separate weights first & do special transpose on 3D weights
        if k in SPECIAL_MAPPING_WITH_HEAD or k in SPECIAL_MAPPING_WITHOUT_HEAD:
            array = np.transpose(array, axes=(2, 1, 0))
        elif "kernel" in new_k:
            array = np.transpose(array)

        tf_weights.append((tf_variables_dict[new_k], array))

    print("EXTRA KEYS:\n", extra_keys)

    tf.keras.backend.batch_set_value(tf_weights)

    return tf_model, hf_model


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_model_id",
        type=str,
        default="facebook/wav2vec2-base",
        help="Model ID of HuggingFace wav2vec2 which needs to be converted into TensorFlow",
    )
    parser.add_argument(
        "--with_lm_head",
        action="store_true",
        help="Whether to use `Wav2Vec2Model` or `Wav2Vec2ForCTC` from `wav2vec2/modeling.py`",
    )
    parser.add_argument(
        "--is_robust",
        action="store_true",
        help="Whether to pass `Wav2Vec2Config` or `RobustWav2Vec2Config` from `wav2vec2/config.py`",
    )
    return parser


if __name__ == "__main__":

    args = get_parser().parse_args()

    config = Wav2Vec2Config() if not args.is_robust else RobustWav2Vec2Config()
    tf_model, _ = get_tf_pretrained_model(
        config, args.hf_model_id, verbose=True, with_lm_head=args.with_lm_head
    )

    model_id = "tf-" + args.hf_model_id.split("/")[-1]
    tf_model.save_pretrained(model_id)
    print(f"TF model `{tf_model}` saved in `{model_id}`")
