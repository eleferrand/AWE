import io, os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from python_speech_features import delta
from python_speech_features import mfcc
import scipy.io.wavfile as wav

class Dataset(object):
    """Creat data class."""

    def __init__(self, wav_path, max_length=None):
        """Initialize dataset."""

        self.feature_dim = 39

        data, words = self.get_data(wav_path)
        self.data = data
        self.words = words
        uwords = np.unique(words)
        word2id = {v: k for k, v in enumerate(uwords)}
        self.id2word = {k: v for k, v in enumerate(uwords)}
        ids = [word2id[w] for w in words]

        self.ids = np.array(ids, dtype=np.int32)

#         self.num_classes = len(self.id_counts)
        self.num_examples = len(self.ids)
        if max_length == None:
            self.max_length = np.max([len(self.data[x]) for x in range(0, self.num_examples-1)])
        else:
            self.max_length = max_length
    def get_data(self, wav_path):
        data = []
        labels = []
        for elt in os.listdir(wav_path):
            (rate, signal) = wav.read(wav_path+elt)
            mfcc_static = mfcc(signal, rate)
            mfcc_deltas = delta(mfcc_static, 2)
            mfcc_delta_deltas = delta(mfcc_deltas, 2)

            features = np.hstack([mfcc_static, mfcc_deltas, mfcc_delta_deltas])
            features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

            data.append(features)
            labels.append(elt)
            data_len = len(labels)

        return data, labels
    def get_max_len(self):
        return self.max_length
    
    def batch(self, batch_size):

        for batch_ind in range(0, self.num_examples, batch_size):
            temp_tens = []
            temp_label = []
            if batch_ind+batch_size>self.num_examples:
                end= batch_ind+batch_size- self.num_examples
            else:
                end = batch_ind+batch_size
            for data_ind in range(batch_ind, end):
                max_length = np.max([len(self.data[x]) for x in range(0, self.num_examples-1)])

                paddings = tf.constant([[0, max_length-len(self.data[data_ind])], [0, 0]])
                padded = tf.pad(self.data[data_ind], paddings, "CONSTANT") 
                temp_tens.append(padded)
                temp_label.append(self.words[data_ind])
            if len(temp_tens)>1:
                yield (tf.reshape(temp_tens, shape=(len(temp_tens),max_length, 39, 1)), tf.convert_to_tensor(temp_label, dtype=object))

def _normalize_img(img, label):
    img = tf.cast(img, tf.float32) / 255.
    return (img, label)

# train_dataset, test_dataset = tfds.load(name="mnist", split=['train', 'test'], as_supervised=True)

# Build your input pipelines
# train_dataset = train_dataset.shuffle(1024).batch(32)
# train_dataset = train_dataset.map(_normalize_img)

# test_dataset = test_dataset.batch(32)
# test_dataset = test_dataset.map(_normalize_img)
# print(len(list(train_dataset)))

train_dataset = Dataset("/home/getalp/leferrae/post_doc/corpora/cv-corpus-12.0-2022-12-07/pt/trainWords/")
test_dataset = Dataset("/home/getalp/leferrae/post_doc/corpora/cv-corpus-12.0-2022-12-07/pt/devWords/", max_length = train_dataset.get_max_len())
train_dataset = train_dataset.batch(32)
test_dataset = test_dataset.batch(32)



model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(194,39,1)),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=None), # No activation on final dense layer
    tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings

])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tfa.losses.TripletSemiHardLoss())

history = model.fit(
    train_dataset,
    epochs=5)