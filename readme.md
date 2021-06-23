# GSoC'21 @ TensorFlow

This repositary hosts my work on the project with [TensorFlow](https://github.com/tensorflow/tensorflow) as a part of [GSoC'21](https://summerofcode.withgoogle.com/).

| Mentors | [Jaeyoun Kim](https://github.com/jaeyounkim), [Morgan Roff](https://github.com/MorganR), [Sayak Paul](https://github.com/sayakpaul) |
|---------|---------|

## Notebooks

| Description                               | Link                                      |
|-------------------------------------------|-------------------------------------------|
| Wav2Vec2 **evaluation on LibriSpeech dataset** using original fine-tuned checkpoint | <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/librispeech-evaluation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| TensorFlow Wav2Vec2 for **Automatic Speech Recognition** | <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/wav2vec2-inference.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |

## Using this Repositary

**Running tests**

```shell
# first install `torch` & `transformers`
pip3 install torch transformers

# run this from root of this repositary
pytest -sv tests
```
