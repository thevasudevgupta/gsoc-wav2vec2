"""Run this script to launch training"""

from dataclasses import dataclass, replace

import tensorflow as tf

from data_utils import LibriSpeechDataLoader, LibriSpeechDataLoaderArgs
from wav2vec2 import CTCLoss, Wav2Vec2ForCTC


@dataclass
class TrainingArgs:
    lr: float = 7e-5
    max_epochs: int = 2
    batch_size: int = 8
    apply_spec_augment: bool = False

    audio_maxlen: int = 50000
    labels_maxlen: int = 128

    seed: int = 0

    train_dir: str = "../data/LibriSpeech/test-clean"
    val_dir: str = "../data/LibriSpeech/test-clean"
    test_dir: str = "../data/LibriSpeech/test-clean"

    # TODO: change this to pre-trained config later
    model_id: str = "vasudevgupta/tf-wav2vec2-base-960h"

    # wandb args
    project_name: str = "gsoc-wav2vec2"
    run_name: str = "finetuning"


def main(args):

    print("######### preparing train set #########")
    tr_data_args = LibriSpeechDataLoaderArgs(
        data_dir=args.train_dir,
        batch_size=args.batch_size,
        audio_maxlen=args.audio_maxlen,
        audio_pad_id=0,
        labels_maxlen=args.labels_maxlen,
        labels_pad_id=0,
    )
    tr_dataset = LibriSpeechDataLoader(tr_data_args)(seed=args.seed)

    print("######### preparing validation set #########")
    val_data_args = replace(tr_data_args, data_dir=args.val_dir)
    val_dataset = LibriSpeechDataLoader(val_data_args)(seed=None)

    print("######### preparing test set #########")
    test_data_args = replace(val_data_args, data_dir=args.test_dir)
    test_dataset = LibriSpeechDataLoader(test_data_args)(seed=None)

    print("######### initializing training state #########")
    input_shape = (args.batch_size, args.audio_maxlen)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss_tracker = tf.keras.metrics.Mean(name="loss")
    callbacks = [] # TODO: fill this

    print("######### preparing model #########")
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_id,
        input_shape=input_shape,
        apply_spec_augment=args.apply_spec_augment,
        layer_drop=0, # TODO: layer-drop is not implemented correctly
    )
    model.freeze_feature_extractor()  # for finetuning
    model.summary()

    print("######### initiating training #########")
    loss_fn = CTCLoss(model.config, input_shape)
    model.compile(optimizer=optimizer, loss_fn=loss_fn, loss_tracker=loss_tracker)

    # train it like any other TF model; just call `model.fit(...)`
    model.fit(
        tr_dataset,
        validation_data=val_dataset,
        epochs=args.max_epochs,
        callbacks=callbacks,
        verbose="auto",
    )

    print("######### preparing for evaluation #########")
    results = model.evaluate(test_dataset, return_dict=True)
    print("RESULTS:\n", results)


if __name__ == "__main__":
    args = TrainingArgs()
    tf.random.set_seed(args.seed)
    main(args)
