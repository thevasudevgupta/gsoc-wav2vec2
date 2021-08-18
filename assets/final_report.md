
# Final Report

## Objectives

* Implement `Wav2Vec2` in `TensorFlow` and **release the pre-trained model** for the TensorFlow community
* Fine-tune Wav2Vec2 model to achieve state-of-art results on speech-related tasks like Automatic Speech Recognition
* Work on Colab Notebooks and tutorials to make it easier for the community to use state-of-the-art speech models

## Acknowledgement

I would like to thank [**Google Summer of Code**](https://summerofcode.withgoogle.com) and [**TensorFlow**](https://www.tensorflow.org) for giving me this opportunity. I am grateful to my mentors [**Sayak Paul**](https://github.com/sayakpaul), [**Morgan Roff**](https://github.com/MorganR) & [**Jaeyoun Kim**](https://github.com/jaeyounkim) for their continuous support and guidance. I would like to also thank [**TPU Research Cloud**](https://sites.research.google/trc/) for providing me with high-performance TPUs that allowed me to train the models at scale.

## Milestones achieved

* Implemented Wav2Vec2 model in TensorFlow 2
* Made scripts for converting LibriSpeech dataset to TFRecords and for efficient data loading from GCS
* Implemented training script for training on multi-TPUs/GPUs
* Made conversion script for converting pre-trained PyTorch Wav2Vec2 checkpoints into TensorFlow compatible format
* Setup tests for ensuring one-to-one mapping of the original and converted model
* Trained Wav2Vec2 on 300 GBs of LibriSpeech tfrecords on TPU v3-8
* Exported pre-trained & fine-tuned TFSavedModel to TFHub
* Made notebook demonstrating fine-tuning of pre-trained Wav2Vec2 model
* Made notebook showing how to evaluate & benchmark fine-tuned Wav2Vec2 on LibriSpeech test dataset
* Made notebook demonstrating export of Wav2Vec2 model to ONNX (Open Neural Network Exchange) and benchmarking Wav2Vec2 inference on CPU
* Deployed Wav2Vec2 on HuggingFace spaces using gradio

For details on running my training scripts, please refer to [`README`](https://github.com/vasudevgupta7/gsoc-wav2vec2) of my repository.

## Pull Requests / Commits

Here is the list of commits/PR made by me during GSoC'21:

| Description | Repository | Link |
|-------------|------------|------|
| Implement & train Wav2Vec2 model in TensorFlow | [`vasudevgupta7/gsoc-wav2vec2`](https://github.com/vasudevgupta7/gsoc-wav2vec2) | [`Commits`](https://github.com/vasudevgupta7/gsoc-wav2vec2/commits?author=vasudevgupta7) |
| Export fine-tuned Wav2Vec2 model to TFHub | [`tensorflow/tfhub.dev`](https://github.com/tensorflow/tfhub.dev) | [`#68`](https://github.com/tensorflow/tfhub.dev/pull/68) |
| Export pre-trained Wav2Vec2 model to TFHub | [`tensorflow/tfhub.dev`](https://github.com/tensorflow/tfhub.dev) | [`#65`](https://github.com/tensorflow/tfhub.dev/pull/65) |
| Add notebook for demonstrating Wav2Vec2 fine-tuning | [`tensorflow/hub`](https://github.com/tensorflow/hub) | [`#788`](https://github.com/tensorflow/hub/pull/788) |

## Notebooks

The following table summarizes the notebooks, I made during my GSoC tenure:

| Notebook | Description |
|------------|-------------|
| <a href="https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/wav2vec2_saved_model_finetuning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | This notebook gives you a template to fine-tune a pre-trained Wav2Vec2 SavedModel |
| <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/wav2vec2_onnx.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | This notebook demonstrates conversion of TF Wav2Vec2 model to ONNX and compares the latency of ONNX exported model & TF model on CPU |
| <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/librispeech_evaluation_WER_3.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | This notebook demonstrates Wav2Vec2 evaluation (without any padding) on LibriSpeech data |
| <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/librispeech_evaluation_WER_6.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | This notebook demonstrates Wav2Vec2 SavedModel evaluation (with constant padding upto 246000 length) on LibriSpeech data |
| <a href="https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/wav2vec2-inference.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | This notebook shows a small demo of how to use Wav2Vec2 for inference for ASR task |

## Results

| Checkpoint | WER (with no padding) | WER with constant padding to 246000 |
|-------------|------------------------|-------------------------------------|
| [`vasudevgupta/gsoc-wav2vec2-960h`](https://huggingface.co/vasudevgupta/gsoc-wav2vec2-960h) | 3.3% | 6% | 
| [`vasudevgupta/finetuned-wav2vec2-960h`](https://huggingface.co/vasudevgupta/finetuned-wav2vec2-960h) | 5.6% | 6.7% |

## Latency Comparison

| Description | Latency |
|---------------------|-----------|
| ONNX exported model | 0.84 secs |
| JIT-compiled model | 2.85 secs |

*Note: Above table is obtained by benchmarking on Colab CPU. Please refer [this notebook](https://colab.research.google.com/github/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/wav2vec2_onnx.ipynb) for reproducing above table*

## Parting thoughts

The last 2-3 months were full of lots of learning and coding. GSoC helped me get into the speech domain and motivated me to explore more about the TensorFlow ecosystem. I am thankful to my mentors for their continuous & timely feedback. I am looking forward to contributing more to the TensorFlow community and other awesome open source projects out there.

## Future Plans (after GSoC'21)

* Distil Wav2Vec2 using knowledge distillation or other strategies and convert distilled Wav2Vec2 into TFLite.
* Deploy TFLite converted model on mobile devices.
