import tensorflow as tf

from convert_torch_to_tf import get_tf_pretrained_model
from wav2vec2 import Wav2Vec2Config

import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_id", default="facebook/wav2vec2-base", type=str)
    parser.add_argument("--with_lm_head", default=False, type=bool)
    parser.add_argument("--verbose", default=False, type=bool)
    parser.add_argument("--saved_model_dir", default="saved-model/", type=str)
    parser.add_argument("--seqlen", default=246000, type=int)
    return parser


if __name__ == '__main__':

    args = get_parser().parse_args()

    # spec_augmentation can't be supported with saved_model for now
    config = Wav2Vec2Config(apply_spec_augment=False)
    model, _ = get_tf_pretrained_model(config, args.hf_model_id, verbose=args.verbose, with_lm_head=args.with_lm_head)

    input_signature = [tf.TensorSpec((None, args.seqlen), dtype=tf.float32)]

    @tf.function(input_signature=input_signature)
    def infer_function(batch):
        return model(batch, training=False)

    @tf.function(input_signature=input_signature)
    def train_function(batch):
        return model(batch, training=True)

    signatures = {"infer": infer_function, "train": train_function}
    tf.saved_model.save(model, args.saved_model_dir, signatures=signatures)
