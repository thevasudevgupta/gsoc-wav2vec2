"""
Run this script to launch training

EXAMPLE:
    >>> TPU_NAME=gsoc-project python3 main.py

    >>> # for running dummy training on TPUs
    >>> DUMMY_DATA_PATH=gs://gsoc-librispeech/dev-clean/dev-clean-0.tfrecord TPU_NAME=gsoc-project python3 main.py
"""

import os
from dataclasses import asdict, dataclass, field, replace
from typing import List

import tensorflow as tf
import wandb

import numpy as np
from data_utils import LibriSpeechDataLoader, LibriSpeechDataLoaderArgs
from training_utils import fetch_callbacks, is_gpu_available, is_tpu_available
from wav2vec2 import CTCLoss, Wav2Vec2ForCTC, Wav2Vec2Config


TPU_NAME = os.getenv("TPU_NAME", "none")
DATA_BUCKET_NAME = os.getenv("DATA_BUCKET_NAME", "gsoc-librispeech-us")
CKPT_BUCKET_NAME = os.getenv("CKPT_BUCKET_NAME", "gsoc-checkpoints-us")
DUMMY_DATA_PATH = os.getenv("DUMMY_DATA_PATH", "none")


@dataclass
class TrainingArgs:

    # main hparams
    lr1: float = 1e-4
    lr2: float = 2e-5
    lr3: float = 4e-5
    transition_epoch1: int = 10
    transition_epoch2: int = 20
    max_epochs: int = 30
    batch_size_per_device: int = 64

    logging_steps: int = 16
    trainable_transition_epoch: int = 10

    # regularization
    apply_spec_augment: bool = False
    survival_prob: float = 1

    # try to keep everything multiple of 128 on TPUs
    # 246000 is 768 for the transformer layer
    audio_maxlen: int = 246000
    labels_maxlen: int = 256

    seed: int = 42
    from_tfrecords: bool = True

    train_tfrecords: List[str] = field(
        default_factory=lambda: [
            f"gs://{DATA_BUCKET_NAME}/train-clean-100/",
            # f"gs://{DATA_BUCKET_NAME}/train-clean-360/",
            # f"gs://{DATA_BUCKET_NAME}/train-other-500/",
        ]
    )
    val_tfrecords: List[str] = field(
        default_factory=lambda: [
            f"gs://{DATA_BUCKET_NAME}/dev-clean/",
            # f"gs://{DATA_BUCKET_NAME}/dev-other/",
        ]
    )
    test_tfrecords: List[str] = field(
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
    run_name: str = "finetuning"

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


def build_model(saved_model_path, args, model_config, model_input_shape, division_factor):
    model = Wav2Vec2ForCTC(Wav2Vec2Config(apply_spec_augment=args.apply_spec_augment, survival_prob=args.survival_prob))
    model.predict(tf.random.uniform(1, args.audio_maxlen))
    model.load_weights(saved_model_path)

    print(model.summary())
    print("######## FREEZING ########")
    if args.trainable_transition_epoch > 0:
        # till `trainable_transition_epoch`, we will train only `lm_head`
        model.layers[0].trainable = False
        print(model.layers[0])
    else:
        # during fine-tuning, it's recommended to freeze the feature extraction layer from the pre-trained weights
        for i in range(len(model.layers[0].layers) - 2):
            model.layers[0].layers[i].trainable = False
            print(model.layers[0].layers[i])
    print("#########################")

    # `division_factor` should be passed else loss will be summed
    # it will help us in distributed training over several processes
    loss_fn = CTCLoss(
        model_config, model_input_shape, division_factor=division_factor
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr1)
    model.compile(optimizer=optimizer, loss=loss_fn)
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
        strategy = None

    if strategy is not None:
        global_batch_size = strategy.num_replicas_in_sync * args.batch_size_per_device
    else:
        global_batch_size = args.batch_size_per_device
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

    print("######### Initializing training state #########")

    # `CTCLoss` needs to know raw-speech input shape in advance
    # Hence, defining it here
    model_input_shape = (args.batch_size_per_device, args.audio_maxlen)
    # NOTE: here we are using `batch_size_per_device` instead of `global_batch_size`
    # since loss will be calculated over each microbatch & will get summed

    print("######### Preparing model #########")
    if strategy is not None:
        with strategy.scope():
            model = build_model(saved_model_path, args, model.config, model_input_shape, global_batch_size)
    else:
        model = build_model(saved_model_path, args, model.config, model_input_shape, global_batch_size)

    print("######### Training #########")
    try:
        history = model.fit(
            tr_dataset,
            validation_data=val_dataset,
            epochs=args.max_epochs,
            callbacks=fetch_callbacks(args),
            verbose="auto",
        )
        print(history.history)
    except KeyboardInterrupt:
        print("Interrupting through KEYBOARD")

    print("\n######### Running evaluation #########")
    results = model.evaluate(test_dataset, return_dict=True)
    print(results)


if __name__ == "__main__":

    # setting up args for training (supports wandb sweep for distributed hparams tuning)
    args = TrainingArgs()
    wandb.init(project=args.project_name, config=asdict(args))
    args.ckpt_path = os.path.join(args.ckpt_path + f"-{wandb.run.id}", "saved-model")

    # setting up seed for reproducible runs
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # start training
    main(args)
