"""
Run this script to launch training

EXAMPLE:
    >>> TPU_NAME=gsoc-project python3 main.py

    >>> # for running dummy training on TPUs
    >>> DUMMY_DATA_PATH=gs://gsoc-librispeech-us/dev-clean/dev-clean-0.tfrecord TPU_NAME=gsoc-project python3 main.py
"""

import os
from dataclasses import asdict, dataclass, field, replace
from typing import List

import tensorflow as tf
import wandb

import numpy as np
from data_utils import LibriSpeechDataLoader, LibriSpeechDataLoaderArgs
from training_utils import fetch_callbacks, is_gpu_available, is_tpu_available
from wav2vec2 import CTCLoss, Wav2Vec2Config, Wav2Vec2ForCTC


TPU_NAME = os.getenv("TPU_NAME", "none")
DATA_BUCKET_NAME = os.getenv("DATA_BUCKET_NAME", "gsoc-librispeech-us")
CKPT_BUCKET_NAME = os.getenv("CKPT_BUCKET_NAME", "gsoc-checkpoints-us")
DUMMY_DATA_PATH = os.getenv("DUMMY_DATA_PATH", "none")


@dataclass
class TrainingArgs:
    # main hparams
    stage1_lr: float = 1e-3
    stage1_epochs: int = 15

    stage2_lr1: float = 1e-4
    stage2_transition_epochs: int = 10
    stage2_lr2: float = 5e-5
    stage2_epochs: int = 15

    batch_size_per_device: int = 32
    logging_steps: int = 16

    # regularization
    apply_spec_augment: bool = True
    survival_prob: float = 1

    # try to keep everything multiple of 128 on TPUs
    # 246000 is 768 for the transformer layer
    audio_maxlen: int = 246000
    labels_maxlen: int = 256

    seed: int = 42
    from_tfrecords: bool = True

    # For training, we converted complete data into multiple tfrecords
    # these tfrecords are further stored in `DATA_BUCKET_NAME`
    # note tfrecords from different splits are stored in different directories in same bucket
    # for more information on data prepartion, please checkout `readme.md`
    train_tfrecords: List[str] = field(
        repr=False,
        default_factory=lambda: [
            f"gs://{DATA_BUCKET_NAME}/train-clean-100/",
            f"gs://{DATA_BUCKET_NAME}/train-clean-360/",
            f"gs://{DATA_BUCKET_NAME}/train-other-500/",
        ]
    )
    # similarly dev data is stored in dev-clean & dev-other directory in same bucket
    val_tfrecords: List[str] = field(
        repr=False,
        default_factory=lambda: [
            f"gs://{DATA_BUCKET_NAME}/dev-clean/",
            # f"gs://{DATA_BUCKET_NAME}/dev-other/",
        ]
    )
    # similarly test data is stored in test-clean & test-other directory in same bucket
    test_tfrecords: List[str] = field(
        repr=False,
        default_factory=lambda: [
            f"gs://{DATA_BUCKET_NAME}/test-clean/",
            # f"gs://{DATA_BUCKET_NAME}/test-other/",
        ]
    )

    train_dir: str = "../data/LibriSpeech/test-clean/"
    val_dir: str = "../data/LibriSpeech/test-clean/"
    test_dir: str = "../data/LibriSpeech/test-clean/"

    model_id: str = "gs://gsoc-weights/tf-wav2vec2-base"
    ckpt_path: str = f"gs://{CKPT_BUCKET_NAME}/experiment"

    # wandb args
    project_name: str = "gsoc-wav2vec2"

    def __post_init__(self):

        if DUMMY_DATA_PATH != "none":
            self.train_dir = self.val_dir = self.test_dir = None
            self.train_tfrecords = tf.io.gfile.glob(DUMMY_DATA_PATH)
            self.test_tfrecords = self.val_tfrecords = self.train_tfrecords
            assert self.from_tfrecords
        else:
            if self.from_tfrecords:
                self.train_dir = self.val_dir = self.test_dir = None

                train_tfrecords = [
                    f"{record}*.tfrecord" for record in self.train_tfrecords
                ]
                self.train_tfrecords = tf.io.gfile.glob(train_tfrecords)

                val_tfrecords = [f"{record}*.tfrecord" for record in self.val_tfrecords]
                self.val_tfrecords = tf.io.gfile.glob(val_tfrecords)

                test_tfrecords = [
                    f"{record}*.tfrecord" for record in self.test_tfrecords
                ]
                self.test_tfrecords = tf.io.gfile.glob(test_tfrecords)

                assert (
                    len(self.train_tfrecords) > 0
                    and len(self.val_tfrecords) > 0
                    and len(self.test_tfrecords) > 0
                )
            else:
                self.train_tfrecords = self.val_tfrecords = self.test_tfrecords = None


def build_model(args):
    model_config = Wav2Vec2Config(apply_spec_augment=args.apply_spec_augment, survival_prob=args.survival_prob)
    model = Wav2Vec2ForCTC(model_config, input_shape=(1, args.audio_maxlen))
    print(f"loading model from {args.model_id}")
    model.load_weights(f"{args.model_id}/tf_model")
    return model


def main(args):
    # on TPUs, we need to connect to TPU cluster first
    # then TensorFlow will be able to detect TPUs
    if TPU_NAME != "none":
        print("############ INITIATING TPU ############")
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(TPU_NAME)
        tf.config.experimental_connect_to_cluster(resolver)
        print("##############################################")

    if is_tpu_available():
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("All devices: ", tf.config.list_logical_devices("TPU"))
        strategy = tf.distribute.TPUStrategy(resolver)
    elif is_gpu_available():
        print("All devices: ", tf.config.list_logical_devices("GPU"))
        strategy = tf.distribute.MirroredStrategy()
    else:
        print("All devices: ", tf.config.list_logical_devices("CPU"))
        raise NotImplementedError

    global_batch_size = strategy.num_replicas_in_sync * args.batch_size_per_device
    print("Training with global batch size of", global_batch_size)
    print(args, end="\n\n")

    print("######### Preparing dataset #########")
    tr_data_args = LibriSpeechDataLoaderArgs(
        data_dir=args.train_dir,
        from_tfrecords=args.from_tfrecords,
        tfrecords=args.train_tfrecords,
        batch_size=global_batch_size,
        audio_maxlen=args.audio_maxlen,
        audio_pad_id=0,
        labels_maxlen=args.labels_maxlen,
        labels_pad_id=0,
    )
    tr_dataset = LibriSpeechDataLoader(tr_data_args)
    tr_dataset = tr_dataset(seed=args.seed, drop_remainder=True)

    val_data_args = replace(
        tr_data_args, data_dir=args.val_dir, tfrecords=args.val_tfrecords
    )
    val_dataset = LibriSpeechDataLoader(val_data_args)
    val_dataset = val_dataset(seed=None, drop_remainder=True)

    test_data_args = replace(
        val_data_args, data_dir=args.test_dir, tfrecords=args.test_tfrecords
    )
    test_dataset = LibriSpeechDataLoader(test_data_args)
    test_dataset = test_dataset(seed=None, drop_remainder=True)

    # `CTCLoss` needs to know raw-speech input shape in advance
    # Hence, defining it here
    model_input_shape = (args.batch_size_per_device, args.audio_maxlen)
    # NOTE: here we are using `batch_size_per_device` instead of `global_batch_size`
    # since loss will be calculated over each microbatch & will get summed

    with strategy.scope():
        print("######### Preparing model #########")
        model = build_model(args)

        # `division_factor` should be passed else loss will be summed
        # it will help us in distributed training over several processes
        loss = CTCLoss(
            model.config, model_input_shape, division_factor=global_batch_size
        )

        # training is divided into 2 stages, hence we will compile model twice & call .fit(...) twice

        print("######################### STAGE-1 #########################")
        # for 1st stage, we will just train the LM head (i.e. top most dense layer) untill the convergence
        # this will ensure pre-trained weights don't get penalized because of randomly initialized LM head

        print("######## FREEZING THE BACKBONE (i.e all pretrained weights) ########")
        # till `stage1_epochs`, we will train only `lm_head`
        model.layers[0].trainable = False
        model.summary()

        optimizer = tf.keras.optimizers.Adam(learning_rate=args.stage1_lr)
        model.compile(optimizer=optimizer, loss=loss)

        try:
            history = model.fit(
                tr_dataset,
                validation_data=val_dataset,
                epochs=args.stage1_epochs,
                callbacks=fetch_callbacks(args, is_stage2=False),
                verbose="auto",
            )
            print(history.history)
        except KeyboardInterrupt:
            print("Interrupting through KEYBOARD")

        print("###########################################################")

        print("######################### STAGE-2 #########################")
        # In 2nd stage, we will fine-tune the complete model except the feature extraction layers
        # It's recommended to freeze all the feature extraction layers during fine-tuning stage

        model.trainable = True
        print("############## FREEZING THE FEATURE_EXTRACTION LAYERS ##############")
        for i in range(len(model.layers[0].layers) - 2):
            model.layers[0].layers[i].trainable = False
        model.summary()

        optimizer = tf.keras.optimizers.Adam(learning_rate=args.stage2_lr1)
        model.compile(optimizer=optimizer, loss=loss)

        try:
            history = model.fit(
                tr_dataset,
                validation_data=val_dataset,
                epochs=args.stage2_epochs,
                callbacks=fetch_callbacks(args, is_stage2=True),
                verbose="auto",
            )
            print(history.history)
        except KeyboardInterrupt:
            print("Interrupting through KEYBOARD")

        print("###########################################################")

    print("\n######### Running evaluation #########")
    results = model.evaluate(test_dataset, return_dict=True)
    print(results)


if __name__ == "__main__":

    # setting up args for training (supports wandb sweep for distributed hparams tuning)
    args = TrainingArgs()
    wandb.init(project=args.project_name, config=asdict(args))
    args.ckpt_path = os.path.join(args.ckpt_path + f"-{wandb.run.id}")

    # setting up seed for reproducible runs
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # start training
    main(args)
