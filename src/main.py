# __author__ = "Vasudev Gupta"
# __author_email__ = "7vasudevgupta@gmail.com"

"""Run this script to launch training"""

import tensorflow as tf

from trainer import Trainer
from wav2vec2 import Wav2Vec2ForCTC
from functools import partial

from .data_utils import DataLoader

MODEL_ID = "wav2vec2-base"


if __name__ == "__main__":

    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.freeze_feature_extractor() # for finetuning
    tr_dataset_fn = partial(DataLoader(data_dir="timit/data/TRAIN"), shuffle=True)
    val_dataset_fn = partial(DataLoader(data_dir="timit/data/TEST"), shuffle=False)

    T = Trainer(lr=2e-4, max_epochs=2, use_tpu=False)
    optimizer = tf.keras.optimizers.Adam(lr=T.lr)
    T.setup(model, optimizer, restore_ckpt=False)

    # let's do training
    T.train(tr_dataset_fn, val_dataset_fn)
