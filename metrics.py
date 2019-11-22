import tensorflow as tf

def masked_softmax_cross_entropy(preds, labels, mask, multi_label=True):
    """Softmax cross-entropy loss with masking."""
    print(preds)
    if not multi_label:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=preds, labels=labels)
    else:
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask, multi_label=True):
    """Accuracy with masking."""
    accuracy_all = None
    if not multi_label:
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    else:
        # pass
        preds = tf.where(preds >= 0.5, tf.ones(tf.shape(preds)), tf.zeros(tf.shape(preds)))
        correct_prediction = tf.equal(preds, labels)
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        correct_prediction = tf.reduce_mean(correct_prediction)

    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)
