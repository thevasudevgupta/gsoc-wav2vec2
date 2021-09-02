import json
import os
from dataclasses import asdict, dataclass, field


@dataclass
class Wav2Vec2Config:
    vocab_size: int = 32
    dropout: int = 0.1
    hidden_size: int = 768
    num_heads: int = 12
    num_layers: int = 12
    intermediate_size: int = 3072
    is_gelu_approx: bool = False
    layer_norm_eps: float = 1e-5
    survival_prob: float = 1.0
    pad_id: int = 0

    # positional embedding
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16

    # feature extractor
    filter_sizes: list = field(
        default_factory=lambda: [512, 512, 512, 512, 512, 512, 512]
    )
    kernal_sizes: list = field(default_factory=lambda: [10, 3, 3, 3, 3, 2, 2])
    strides: list = field(default_factory=lambda: [5, 2, 2, 2, 2, 2, 2])
    conv_bias: bool = False

    # spec augmentation arguments
    apply_spec_augment: bool = True
    mask_time_prob: float = 0.05
    mask_time_length: int = 10

    attention_norm_type: str = "postnorm"
    feature_extractor_norm_type: bool = "group"
    is_robust: bool = False

    def __post_init__(self):
        if not (len(self.filter_sizes) == len(self.kernal_sizes) == len(self.strides)):
            raise ValueError(
                "Length of filter_sizes, kernal_sizes, strides must match."
            )
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("Hidden size must be perfect multiple of num_heads.")

        assert self.feature_extractor_norm_type in ["group", "layer"], "Only `group` / `layer` are supported"
        assert self.attention_norm_type in ["prenorm", "postnorm"], "Only `prenorm` / `postnorm` are supported"

    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(asdict(self), f)

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)


@dataclass
class RobustWav2Vec2Config(Wav2Vec2Config):
    attention_norm_type: str = "prenorm"
    feature_extractor_norm_type: str = "layer"
    is_robust: bool = True
    conv_bias: bool = True

    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_heads: int = 16
    num_layers: int = 24
