# __author__ = "Vasudev Gupta"
# __author_email__ = "7vasudevgupta@gmail.com"

import os
import json
from dataclasses import dataclass, asdict


@dataclass
class Wave2Vec2Config:
    vocab_size: int = 32
    dropout: int = .1
    hidden_size: int = 768
    intermediate_size: int = 3072

    def save_to_disk(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(asdict(self), f)


if __name__ == "__main__":
    """Testing area"""

    config = Wave2Vec2Config()
    print(config)
    config.save_to_disk("dummy")
