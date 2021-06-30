"""Run this script to launch training"""

from dataclasses import dataclass, replace, field

from typing import List
import tensorflow as tf
import wandb

from data_utils import LibriSpeechDataLoader, LibriSpeechDataLoaderArgs
from wav2vec2 import CTCLoss, Wav2Vec2ForCTCTrainer
from training_utils import fetch_callbacks
from training_utils import LocalTPUClusterResolver, is_gpu_available, is_tpu_available

import os
ON_COLAB_TPU = os.getenv("ON_COLAB_TPU", "false")

@dataclass
class TrainingArgs:

    # lr related args
    lr: float = 5e-5
    transition_epoch: int = 1

    max_epochs: int = 5
    batch_size_per_device: int = 1
    apply_spec_augment: bool = True
    layer_drop: float = 0.0

    audio_maxlen: int = 50000
    labels_maxlen: int = 128

    seed: int = 42

    from_tfrecords: bool = True

    train_tfrecords: List[str] = field(default_factory=lambda: ["gs://gsoc-librispeech/test/test-clean.tfrecord"])
    val_tfrecords: List[str] = field(default_factory=lambda: ["gs://gsoc-librispeech/test/test-clean.tfrecord"])
    test_tfrecords: List[str] = field(default_factory=lambda: ["gs://gsoc-librispeech/test/test-clean.tfrecord"])

    train_dir: str = "../data/LibriSpeech/test-clean"
    val_dir: str = "../data/LibriSpeech/test-clean"
    test_dir: str = "../data/LibriSpeech/test-clean"

    model_id: str = "vasudevgupta/tf-wav2vec2-base"

    save_path: str = "tf_model.h5"
    seed: int = 0

    # wandb args
    project_name: str = "gsoc-wav2vec2"
    run_name: str = "finetuning"

    def __post_init__(self):
        if self.from_tfrecords:
            self.train_dir = self.val_dir = self.test_dir = None
        else:
            self.train_tfrecords = self.val_tfrecords = self.test_tfrecords = None

def main(args):
    # setting up args for sweep
    wandb.init(project="gsoc-wav2vec2", config=args.__dict__)
    logging_dict = dict(wandb.config)
    args = replace(args, **logging_dict)

    # on colab, we need to connect to TPU cluster first
    # then TensorFlow will be able to detect TPUs
    if ON_COLAB_TPU == "true":
      print("############ INITIATING COLAB TPU ############")
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
      tf.config.experimental_connect_to_cluster(resolver)
      print("##############################################")

    jit_compile = True
    if is_tpu_available():
        jit_compile = None
        if not ON_COLAB_TPU == "true":
          resolver = LocalTPUClusterResolver()
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("All devices: ", tf.config.list_logical_devices('TPU'))
        strategy = tf.distribute.TPUStrategy(resolver)
    elif is_gpu_available():
        print("All devices: ", tf.config.list_logical_devices('GPU'))
        strategy = tf.distribute.MirroredStrategy()
    else:
        print("All devices: ", tf.config.list_logical_devices('CPU'))
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

    val_data_args = replace(tr_data_args, data_dir=args.val_dir, tfrecords=args.val_tfrecords)
    val_dataset = LibriSpeechDataLoader(val_data_args)
    val_dataset = val_dataset(seed=None, drop_remainder=True)

    test_data_args = replace(val_data_args, data_dir=args.test_dir, tfrecords=args.test_tfrecords)
    test_dataset = LibriSpeechDataLoader(test_data_args)
    test_dataset = test_dataset(seed=None, drop_remainder=True)

    print("######### Initializing training state #########")

    # `CTCLoss` needs to know raw-speech input shape in advance
    # Hence, defining it here
    model_input_shape = (args.batch_size_per_device, args.audio_maxlen)
    # NOTE: here we are using `batch_size_per_device` instead of `global_batch_size`
    # since loss will be calculated over each microbatch & will get summed

    with strategy.scope():
        print("######### Preparing model #########")
        model = Wav2Vec2ForCTCTrainer.from_pretrained(
            args.model_id,
            jit_compile=jit_compile,
            input_shape=model_input_shape,
            apply_spec_augment=args.apply_spec_augment,
            layer_drop=args.layer_drop,
        )

        # during fine-tuning, we need to freeze the feature extraction layer with pre-trained weights
        model.freeze_feature_extractor()

        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

        # `division_factor` should be passed else loss will be summed
        # it will help us in distributed training over several processes
        loss_fn = CTCLoss(model.config, model_input_shape, division_factor=global_batch_size)

        model.compile(
            optimizer=optimizer,
            steps_per_execution=None,
            loss_fn=loss_fn,
        )
    model.summary()
    print("######### Initiating training #########")
    model.fit(
        tr_dataset.repeat(args.max_epochs),
        # validation_data=val_dataset.take(2),
        epochs=1,
        callbacks=fetch_callbacks(args),
        verbose="auto",
    )

    # print("######### Preparing for evaluation #########")
    # results = model.evaluate(test_dataset.take(32), return_dict=True)
    # print("RESULTS:\n", results)


if __name__ == "__main__":
    args = TrainingArgs()
    tf.random.set_seed(args.seed)
    main(args)
