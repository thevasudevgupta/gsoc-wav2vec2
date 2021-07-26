import tensorflow as tf

from convert_torch_to_tf import get_tf_pretrained_model
from wav2vec2 import Wav2Vec2Config

import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_model_id",
        default="facebook/wav2vec2-base",
        type=str,
        help="Model ID of HuggingFace wav2vec2 which needs to be converted into TensorFlow",
    )
    parser.add_argument(
        "--with_lm_head",
        action="store_true",
        help="Whether to use `Wav2Vec2Model` or `Wav2Vec2ForCTC` from `wav2vec2/modeling.py`",
    )
    parser.add_argument(
        "--verbose",
        default=False,
        type=bool,
        help="Whether to display specific layers conversion while running conversion script",
    )
    parser.add_argument(
        "--saved_model_dir",
        default="saved-model/",
        type=str,
        help="Where to save the obtained `saved-model`",
    )
    parser.add_argument(
        "--seqlen",
        default=246000,
        type=int,
        help="With what sequence length should model be exported",
    )
    return parser


if __name__ == "__main__":

    args = get_parser().parse_args()

    # spec_augmentation can't be supported with saved_model for now
    config = Wav2Vec2Config(apply_spec_augment=False)
    model, _ = get_tf_pretrained_model(
        config, args.hf_model_id, verbose=args.verbose, with_lm_head=args.with_lm_head
    )

    input_signature = [tf.TensorSpec((None, args.seqlen), tf.float32, name="speech")]
    model.__call__ = tf.function(model.__call__, input_signature=input_signature)
    tf.saved_model.save(model, args.saved_model_dir)

    print(f"Your `TF SavedModel ({model})` saved in {args.saved_model_dir}")
