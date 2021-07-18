import pathlib

import tensorflow as tf
import tensorflow_datasets as tfds



def datasets_fra_eng(val_prop=0.2, buffer_size=None):
    Dataset = tf.data.Dataset
    zip_path = tf.keras.utils.get_file('fra-eng.zip',
        origin=('http://storage.googleapis.com/download.'
                'tensorflow.org/data/fra-eng.zip'),
        extract=True)
    file_path = pathlib.Path(zip_path).parent/'fra.txt'

    text = file_path.read_text(encoding='utf-8')

    lines = text.splitlines()
    pairs = [line.split('\t') for line in lines]

    inp = [inp for (_, inp) in pairs]
    targ = [targ for (targ, _) in pairs]
    n_sequences = len(inp)

    buffer_size = n_sequences if buffer_size is None else buffer_size

    full_dataset = Dataset.from_tensor_slices((inp, targ)).shuffle(buffer_size)
    test_dataset = full_dataset.take(int(n_sequences * val_prop))
    train_dataset = full_dataset.skip(int(n_sequences * val_prop))
    return train_dataset, test_dataset, full_dataset

if __name__ == '__main__':
    train_dataset, _, _ = datasets_fra_eng()
    train_fr = train_dataset.map(lambda fr, _: fr)
    train_en = train_dataset.map(lambda _, en: en)






