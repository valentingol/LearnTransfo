import time

import tensorflow as tf

from datasets.scripts.fra_eng import datasets_fra_eng
from transformer.text.tokenizer import TokenizerBert
from transformer.architecture.transfo import TransformerNLP
from transformer.train.metrics import MaskedAccuracy
from transformer.train.metrics import MaskedSparseCategoricalCrossentropy
from transformer.train.optimizer import ScheduleLR

if __name__ == '__main__':
    # get dataset for french to english traduction
    _, _, full_dataset = datasets_fra_eng()
    full_dataset = full_dataset.shuffle(buffer_size=len(full_dataset))
    len_ds = len(full_dataset)

    # build tokenizer
    fr_dataset = full_dataset.map(lambda fr, _: fr)
    en_dataset = full_dataset.map(lambda _, en: en)
    fr_tokenizer = TokenizerBert(lower_case=True)
    en_tokenizer = TokenizerBert(lower_case=True)
    fr_tokenizer.build_tokenizer(fr_dataset)
    en_tokenizer.build_tokenizer(en_dataset)

    # prepare dataset
    full_dataset = full_dataset.cache()
    full_dataset = full_dataset.batch(32)
    full_dataset = full_dataset.prefetch(2)

    # create transformer
    in_vocab_size = len(fr_tokenizer.vocab)
    out_vocab_size = len(en_tokenizer.vocab)
    transfo = TransformerNLP(n_layers=12, d_model=768, n_heads=12, d_ff=1072,
                             dropout=0.1, in_vocab_size=in_vocab_size,
                             out_vocab_size=out_vocab_size,
                             max_seq_len=40)

    # training set-up
    schedule_lr = ScheduleLR(d_model=transfo.d_model)
    opt = tf.keras.optimizers.Adam(schedule_lr, beta_1=0.9, beta_2=0.98,
                                   epsilon=1e-9)
    loss_function = MaskedSparseCategoricalCrossentropy()
    acc_function = MaskedAccuracy()

    # training function
    input_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int64)] * 3
    @tf.function(input_signature=input_signature)
    def train(fr_tokens, en_tokens, labels):
        with tf.GradientTape() as tape:
            proba, _ = transfo(fr_tokens, en_tokens, training=True)
            # enventually cut to maximum_length to match proba shape
            labels = labels[..., :tf.shape(proba)[-2]]
            loss = loss_function(labels, proba)
        grads = tape.gradient(loss, transfo.trainable_variables)
        opt.apply_gradients(zip(grads, transfo.trainable_variables))
        acc = acc_function(labels, proba)
        return loss, acc

    # training loop
    mean_loss = tf.keras.metrics.Mean()
    mean_acc = tf.keras.metrics.Mean()
    for i, (fr_txt, en_txt) in enumerate(full_dataset):
        fr_tokens, en_tokens = fr_tokenizer(fr_txt), en_tokenizer(en_txt)
        labels = en_tokens[:, 1:]
        en_tokens = en_tokens[:, :-1]

        loss, acc = train(fr_tokens, en_tokens, labels)
        loss, acc = mean_loss(loss), mean_acc(acc)

        if i == 0: start = time.time()
        if i % 100 == 0 and i > 0:
            current_time_epoch = time.time() - start
            time_epoch = current_time_epoch * len_ds / (i+1)
            remaining_time = time_epoch - current_time_epoch
            print('batch', i, '/', len_ds)
            print(f'loss = {loss.numpy():.3f}, acc = {acc.numpy():.3f}')
            print(f'estimated remaining time: {int(remaining_time // 60)}min '
                  f'{remaining_time % 60:.1f}sec')
            mean_loss.reset_state(), mean_acc.reset_state()
