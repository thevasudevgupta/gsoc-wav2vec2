import json
import os
import re
import subprocess
from itertools import groupby

import tensorflow as tf


class Wav2Vec2Processor:
    def __init__(
        self, is_tokenizer, do_normalize=True, vocab_path="./vocab.json"
    ):
        # whether to use as `feature_extractor` or `tokenizer`

        self.is_tokenizer = is_tokenizer
        self.do_normalize = do_normalize
        self.vocab_path = vocab_path

        if self.is_tokenizer:
            self._setup_vocab()

            self.token_to_id_mapping = self.get_vocab()
            self.id_to_token_mapping = {
                v: k for k, v in self.token_to_id_mapping.items()
            }
            self.unk_token = "<unk>"
            self.unk_id = self.token_to_id_mapping[self.unk_token]

            self.dimiliter_token = "|"
            self.dimiliter_id = self.token_to_id_mapping[self.dimiliter_token]

            special_tokens = ["<pad>"]
            self.special_ids = [self.token_to_id_mapping[k] for k in special_tokens]

    def _setup_vocab(self):
        """This method will download & setup the vocab file if it's not on the `vocab_path`"""
        if not os.path.isfile(self.vocab_path):
            url = "https://github.com/vasudevgupta7/gsoc-wav2vec2/raw/main/data/vocab.json"

            print(f"Downloading `vocab.json` from {url} ... ", end="")
            try:
                subprocess.run(
                    ["wget", url], stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
            except:
                raise ValueError(f"Couldn't download `vocab.json` from {url}")
            print("DONE")

            self.vocab_path = "./vocab.json"

    def __call__(self, input_values):
        """
        if is_tokenizer:
            input_values (:obj: `str`):
                Single string you want to encode to ids
        else:
            input_values (:obj: `tf.Tensor`):
                Tensor which needs to be fed into `model.call()`
        """
        if self.is_tokenizer:
            input_values = self._tokenize(input_values)
            input_values = [
                self.token_to_id_mapping.get(k, self.unk_id) for k in input_values
            ]
        else:
            if self.do_normalize:
                input_values = self._normalize(input_values)
        return input_values

    def decode(self, input_ids: list, skip_special_tokens=True, group_tokens=True):
        """
        Use this method to decode your ids back to string.

        Args:
            input_ids (:obj: `list`):
                input_ids you want to decode to string.
            skip_special_tokens (:obj: `bool`, `optional`):
                Whether to remove special tokens (like `<pad>`) from string.
            group_tokens (:obj: `bool`, `optional`):
                Whether to group repeated characters.
        """
        if group_tokens:
            input_ids = [t[0] for t in groupby(input_ids)]
        if skip_special_tokens:
            input_ids = [k for k in input_ids if k not in self.special_ids]
        tokens = [self.id_to_token_mapping.get(k, self.unk_token) for k in input_ids]
        tokens = [k if k != self.dimiliter_token else " " for k in tokens]
        return "".join(tokens).strip()

    def _tokenize(self, string: str):
        string = re.sub("-", " ", string)
        string = re.sub("[^A-Z' ]", "", string.upper())
        return list(string.replace(" ", self.dimiliter_token))

    def get_vocab(self):
        with open(self.vocab_path, "r") as f:
            vocab = json.load(f)
        return vocab

    def _normalize(self, x):
        """You must call this before padding."""
        # -> (1, seqlen)
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        var = tf.math.reduce_variance(x, axis=-1, keepdims=True)
        return tf.squeeze((x - mean) / tf.sqrt(var + 1e-5))


if __name__ == "__main__":
    """Testing Area"""

    feature_extractor = Wav2Vec2Processor(is_tokenizer=False)
    batch, _ = tf.audio.decode_wav(tf.io.read_file("../data/sample.wav"))
    batch = tf.transpose(batch, perm=(1, 0))
    batch = tf.concat([batch, batch], axis=0)

    out = feature_extractor(batch)
    print(out)

    print("\n\n")

    tokenizer = Wav2Vec2Processor(is_tokenizer=True)
    ids = tokenizer("vasudev guptaa is a data scientist.")
    print(ids)
    print(tokenizer.decode(ids))
    print(tokenizer.decode(ids, group_tokens=False))

    ids = tokenizer("how is life gooing? what's up.. yayy i got results. it's awe-some")
    print(ids)
    print(tokenizer.decode(ids))
    print(tokenizer.decode(ids, group_tokens=False))
