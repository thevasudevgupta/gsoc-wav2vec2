import argparse
import tensorflow as tf

from wav2vec2 import Wav2Vec2Model, Wav2Vec2ForCTC


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
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
    parser.add_argument(
        "--is_robust",
        action="store_true",
        help="Whether to pass `Wav2Vec2Config` or `RobustWav2Vec2Config` from `wav2vec2/config.py`",
    )
    return parser


class RobustModelExporter(Wav2Vec2Model):
    def call(self, inputs, training=False):
        speech, attention_mask = inputs
        return super().call(speech, attention_mask=attention_mask, training=training)

    def _init(self, *args, **kwargs):
        kwargs["for_export"] = True
        return super()._init(*args, **kwargs)


class RobustModelForCTCExporter(Wav2Vec2ForCTC):
    def call(self, inputs, training=False):
        speech, attention_mask = inputs
        return super().call(speech, attention_mask=attention_mask, training=training)

    def _init(self, *args, **kwargs):
        kwargs["for_export"] = True
        return super()._init(*args, **kwargs)


if __name__ == "__main__":
    args = get_parser().parse_args()

    # spec_augmentation can't be supported with saved_model for now
    if not args.is_robust:
        ModelClass = Wav2Vec2ForCTC if args.with_lm_head else Wav2Vec2Model
    else:
        ModelClass = RobustModelForCTCExporter if args.with_lm_head else RobustModelExporter

    model = ModelClass.from_pretrained(args.model_id, apply_spec_augment=False, input_shape=(1, args.seqlen))
    tf.saved_model.save(model, args.saved_model_dir)

    print(f"Your `TF SavedModel ({model}, {model.config})` saved in {args.saved_model_dir}")
