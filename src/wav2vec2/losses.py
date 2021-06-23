import tensorflow as tf


class CTCLoss:
    def __init__(self, config, input_shape):
        self.kernal_sizes = config.kernal_sizes
        self.strides = config.strides
        self.pad_id = config.pad_id
        self.loss_reduction = config.loss_reduction

        self.input_shape = input_shape

    def __call__(self, hidden_states, labels):
        """
        This methods wraps up `tf.nn.ctc_loss` and returns the ctc-loss for batch

        Args:
            hidden_states (:obj: `tf.Tensor`):
                This is the output of LM head of `Wav2Vec2ForCTC.call(...)`
            labels (:obj: `tf.Tensor`):
                This is batch of tokenized text labels

        Returns:
            loss (:obj: `tf.Tensor`):
                This is the summation/mean of CTC loss of the batch. Mean/Summation will be decided by
                `loss_reduction` parameter in your config
        """
        input_length = tf.ones(self.input_shape[0]) * self.input_shape[1]
        logit_length = self._get_logit_length(input_length)

        label_mask = tf.cast(labels != self.pad_id, tf.int32)
        label_length = tf.reduce_sum(label_mask, axis=-1)

        loss = tf.nn.ctc_loss(
            labels=labels,
            logits=hidden_states,
            label_length=label_length,
            logit_length=logit_length,
            logits_time_major=False,
            blank_index=self.pad_id,
            name="ctc-loss",
        )

        if self.loss_reduction == "sum":
            loss = tf.reduce_sum(loss)
        else:
            loss = tf.reduce_mean(loss)

        return loss

    def _get_logit_length(self, input_length):
        """
        This will return length of the sequence at the end of convolutional layers
        i.e. seqlen fed to transformer encoder.
        """
        kernal_sizes = self.kernal_sizes
        strides = self.strides
        for kernal_size, stride in zip(kernal_sizes, strides):
            input_length = 1 + (input_length - kernal_size) // stride
        return input_length
