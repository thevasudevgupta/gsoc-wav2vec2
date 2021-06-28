import argparse

import tensorflow as tf

import numpy as np
from data_utils import (
    LABEL_DTYPE,
    SPEECH_DTYPE,
    LibriSpeechDataLoader,
    LibriSpeechDataLoaderArgs,
)
from tqdm.auto import tqdm


def create_tfrecord(speech_tensor, label_tensor):
    speech_bytes = speech_tensor.numpy().astype(SPEECH_DTYPE).tobytes()
    label_bytes = label_tensor.numpy().astype(LABEL_DTYPE).tobytes()

    assert np.allclose(
        np.frombuffer(speech_bytes, dtype=SPEECH_DTYPE), speech_tensor, atol=1e-4
    )
    assert np.allclose(
        np.frombuffer(label_bytes, dtype=LABEL_DTYPE), label_tensor, atol=1e-4
    )

    feature = {
        "speech": tf.train.Feature(bytes_list=tf.train.BytesList(value=[speech_bytes])),
        "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_bytes])),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "CLI to convert .flac dataset into .tfrecords format"
    )
    parser.add_argument(
        "--data_dir", default="../data/librispeech/test-clean", type=str
    )
    parser.add_argument("-f", "--record_filename", default="data.tfrecords", type=str)

    args = parser.parse_args()

    data_args = LibriSpeechDataLoaderArgs(data_dir=args.data_dir)
    dataloader = LibriSpeechDataLoader(data_args)
    dataset = dataloader.build_and_fetch_dataset()

    with tf.io.TFRecordWriter(args.record_filename) as writer:
        for speech, label in tqdm(
            dataset, total=len(dataloader), desc="Making tfrecords ... "
        ):
            speech, label = tf.squeeze(speech), tf.squeeze(label)
            tf_record = create_tfrecord(speech, label)
            writer.write(tf_record)
    print(f"All tfrecords are created in `{args.record_filename}`")

    # shift below code to tests
    # from data_utils import read_tfrecord
    # dataset = tf.data.TFRecordDataset(args.record_filename)
    # dataset = dataset.map(read_tfrecord)
    # for speech, label in dataset:
    #     print(speech, label)
