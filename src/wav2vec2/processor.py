import json
from dataclasses import dataclass

import tensorflow as tf

VOCAB_PATH = "../data/vocab.json"


@dataclass
class Wav2Vec2Processer:
    is_tokenizer: bool  # whether to use as `feature_extractor` or `tokenizer`
    do_normalize: bool = True

    def __post_init__(self):
        if self.is_tokenizer:
            self.token_to_id_mapping = self.get_vocab()
            self.id_to_token_mapping = {
                v: k for k, v in self.token_to_id_mapping.items()
            }
            self.unk_token = "<unk>"
            self.unk_id = self.token_to_id_mapping[self.unk_token]

            self.dimiliter_token = "|"
            self.dimiliter_id = self.token_to_id_mapping[self.dimiliter_token]

    def __call__(self, input_values):
        """
        if is_tokenizer:
            input_values: `str`

        else:
            input_values: `tf.Tensor`
        """
        if self.is_tokenizer:
            input_values = input_values.upper()
            input_values = self._tokenize(input_values)
            input_values = [
                self.token_to_id_mapping.get(k, self.unk_id) for k in input_values
            ]
        else:
            if self.do_normalize:
                input_values = self._normalize(input_values)
        return input_values

    def decode(self, input_ids: list):
        ids = [self.id_to_token_mapping.get(k, self.unk_token) for k in input_ids]
        ids = [k if k != self.dimiliter_token else " " for k in ids]
        return "".join(ids)

    def _tokenize(self, string: str):
        return list(string.replace(" ", self.dimiliter_token))

    def get_vocab(self):
        if not self.is_tokenizer:
            raise NotImplementedError
        with open(VOCAB_PATH, "r") as f:
            vocab = json.load(f)
        return vocab

    def _normalize(self, x):
        # -> bsz, seqlen
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        std = tf.math.reduce_std(x, axis=1, keepdims=True)
        return (x - mean) / (std + 1e-5)


if __name__ == "__main__":
    """Testing Area"""

    feature_extractor = Wav2Vec2Processer(is_tokenizer=False)
    batch, _ = tf.audio.decode_wav(tf.io.read_file("../data/sample.wav"))
    batch = tf.transpose(batch, perm=(1, 0))
    batch = tf.concat([batch, batch], axis=0)

    out = feature_extractor(batch)
    print(out)

    print("\n\n")

    tokenizer = Wav2Vec2Processer(is_tokenizer=True)
    ids = tokenizer("vasudev gupta is a data scientist.")
    print(ids)
    print(tokenizer.decode(ids))
