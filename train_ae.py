import tensorflow as tf
from tensorflow.keras.layers import  GRU, Dense, Reshape, LSTM
import numpy as np
import os
import _pickle as pickle
import argparse
import scipy.io.wavfile as wav
from python_speech_features import delta
from python_speech_features import mfcc
from tensorflow.keras.models import Model
import math
from tqdm import tqdm
from scipy.spatial.distance import cosine
import sys

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)
tf.compat.v1.logging.set_verbosity(40)

def eval_QbE(model, max_length):
    queries = {}
    test = {}
    results = {}
    # rootTest = "/home/getalp/leferrae/post_doc/classifier_zing/wordsTest/"
    # rootQueries = "/home/getalp/leferrae/post_doc/corpora/guinee_casa/queries/"
    rootTest = "/home/getalp/leferrae/post_doc/corpora/cv-corpus-12.0-2022-12-07/pt/testWords/"
    rootQueries = "/home/getalp/leferrae/post_doc/corpora/cv-corpus-12.0-2022-12-07/pt/queries/"
    temp_emb = []
    temp_labels = []
    for elt in tqdm(os.listdir(rootQueries)):
        if ".wav" in elt:
            (rate, signal) = wav.read(rootQueries+elt)
            mfcc_static = mfcc(signal, rate)
            mfcc_deltas = delta(mfcc_static, 2)
            mfcc_delta_deltas = delta(mfcc_deltas, 2)
            features = np.hstack([mfcc_static, mfcc_deltas, mfcc_delta_deltas])
            features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
            if len(features)>max_length:
                tooMuch = len(features)-max_length
                startTrunc = tooMuch//2
                endTruc = tooMuch//2+tooMuch%2
                padded = tf.constant(features[startTrunc:-endTrunc])
            else:
                pad_bef = (max_length-len(features))//2
                pad_aft = (max_length-len(features))//2+(max_length-len(features))%2
                paddings = tf.constant([[pad_bef, pad_aft], [0, 0]])
                padded = tf.pad(features, paddings, "CONSTANT") 
            temp_emb.append(padded)
            temp_labels.append(elt.replace(".wav", ""))
            
    embeddings = model.make_emb(tf.reshape(temp_emb, shape=(len(temp_emb),max_length, 39)))
                                
    for i, emb in enumerate(embeddings):
        queries[temp_labels[i]] = emb
    
    temp_emb = []
    temp_labels = []
    for elt in tqdm(os.listdir(rootTest)):
        (rate, signal) = wav.read(rootTest+elt)
        mfcc_static = mfcc(signal, rate)
        mfcc_deltas = delta(mfcc_static, 2)
        mfcc_delta_deltas = delta(mfcc_deltas, 2)
        temp_labels.append(elt)
        features = np.hstack([mfcc_static, mfcc_deltas, mfcc_delta_deltas])
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
        if len(features)>max_length:
            tooMuch = len(features)-max_length
            startTrunc = tooMuch//2
            endTrunc = tooMuch//2+tooMuch%2
            padded = tf.constant(features[startTrunc:-endTrunc])
        else:
            pad_bef = (max_length-len(features))//2
            pad_aft = (max_length-len(features))//2+(max_length-len(features))%2
            paddings = tf.constant([[pad_bef, pad_aft], [0, 0]])
            padded = tf.pad(features, paddings, "CONSTANT") 
        temp_emb.append(padded)
#     embeddings = model.make_emb(tf.reshape(temp_emb, shape=(len(temp_emb),max_length, 39)))
    embeddings = model.make_emb(tf.reshape(temp_emb, shape=(len(temp_emb),max_length, 39)))
    for i, emb in enumerate(embeddings):
        elt = temp_labels[i]
        queries[elt.replace(".wav", "")] = emb
        name = elt.split("_")[0]
        test[elt.replace(".wav", "")] = {"embedding" : emb, "label" : name}
    for testWord in test:
        scores = []
        for query in queries:
            score = cosine(test[testWord]["embedding"][0], queries[query][0])
            scores.append((query, score))
        scores.sort(key=lambda x : x[1])
        results[testWord] = scores
    top1 = 0
    top2 = 0
    top5 = 0
    tot = 0
    for elt in results:
        tot+=1
        name = elt.split("_")[0]
        list2 = [x[0] for x in results[elt][:2]]
        list5 = [x[0] for x in results[elt][:5]]

        if name == results[elt][0][0]:
            top1+=1
        if name in list2:
            top2+=1
        if name in list5:
            top5+=1
    print("\n",top1/tot*100, top2/tot*100, top5/tot*100)

def print_progress_bar(index, total, label):
    n_bar = 50  # Progress bar width
    progress = index / total
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * progress):{n_bar}s}] {int(100 * progress)}%  {label}")
    sys.stdout.flush()

class Dataset(object):
    """Creat data class."""

    def __init__(self, wav_path, max_length=None):
        """Initialize dataset."""

        self.feature_dim = 39
        data, words, refs = self.get_data(wav_path)
        self.data = data
        self.words = words
        self.refs = refs
        uwords = np.unique(words)
        word2id = {v: k for k, v in enumerate(uwords)}
        self.id2word = {k: v for k, v in enumerate(uwords)}
        ids = [word2id[w] for w in words]

        self.ids = np.array(ids, dtype=np.int32)

        self.num_examples = len(self.ids)
        if max_length == None:
            self.max_length = np.max([len(self.data[x]) for x in range(0, self.num_examples-1)])
        else:
            self.max_length = max_length
    def get_data(self, wav_path):
        data = []
        labels = []
        refs = []
        for elt in os.listdir(wav_path):
            if ".wav" in elt:
                (rate, signal) = wav.read(wav_path+elt)
                mfcc_static = mfcc(signal, rate)
                mfcc_deltas = delta(mfcc_static, 2)
                mfcc_delta_deltas = delta(mfcc_deltas, 2)

                features = np.hstack([mfcc_static, mfcc_deltas, mfcc_delta_deltas])
                features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
                word = elt.split("_")[0]
                data.append(features)
                labels.append(word)
                refs.append(elt)
                data_len = len(labels)

        return data, labels, refs

    def batch(self, batch_size):
        self.tot_batch = self.num_examples//batch_size-1
        for batch_ind in range(0, self.num_examples, batch_size):
            temp_tens = []
            temp_label = []
            if batch_ind+batch_size>self.num_examples:
                end= batch_ind+batch_size- self.num_examples
            else:
                end = batch_ind+batch_size
            for data_ind in range(batch_ind, end):
                if len(self.data[data_ind])>self.max_length:
                    tooMuch = len(self.data[data_ind])-self.max_length
                    startTrunc = tooMuch//2
                    endTrunc = tooMuch//2+tooMuch%2
                    padded = tf.constant(self.data[data_ind][startTrunc:-endTrunc])
                else:
                    pad_bef = (self.max_length-len(self.data[data_ind]))//2
                    pad_aft = (self.max_length-len(self.data[data_ind]))//2+(self.max_length-len(self.data[data_ind]))%2
                    paddings = tf.constant([[pad_bef, pad_aft], [0, 0]])
                    padded = tf.pad(self.data[data_ind], paddings, "CONSTANT") 
                temp_tens.append(padded)
                temp_label.append(self.words[data_ind])
#             yield (tf.convert_to_tensor(temp_tens, dtype=tf.float32), tf.convert_to_tensor(temp_label, dtype=object))
            if len(temp_tens)>1:
                yield (tf.reshape(temp_tens, shape=(len(temp_tens),self.max_length, 39)))
    def batch_pairs(self, batch_size):
        
        x_inp = []
        y_inp = []
        x_ref = []
        y_ref = []
        for ind_x, tens_x in enumerate(self.data):
            for ind_y, tens_y in enumerate(self.data):
                if self.refs[ind_x] != self.refs[ind_y] and self.words[ind_x] == self.words[ind_y]:

                    x_inp.append(tens_x)
                    y_inp.append(tens_y)
                    x_ref.append(self.refs[ind_x])
                    y_ref.append(self.refs[ind_y])
        self.tot_batch_pair = len(x_inp)//batch_size-1
        for batch_ind in range(0, len(x_inp), batch_size):
            temp_tens = []
            temp_label = []
            final_x = []
            final_y = []
            if batch_ind+batch_size>len(x_inp):
                end= batch_ind+batch_size- len(x_inp)
            else:
                end = batch_ind+batch_size
            for data_ind in range(batch_ind, end):
                if len(x_inp[data_ind])>self.max_length:
                    tooMuch = len(x_inp[data_ind])-self.max_length
                    startTrunc = tooMuch//2
                    endTrunc = tooMuch//2+tooMuch%2
                    padded_x = tf.constant(x_inp[data_ind][startTrunc:-endTrunc])
                    final_x.append(padded_x)
                else:
                    pad_bef = (self.max_length-len(x_inp[data_ind]))//2
                    pad_aft = (self.max_length-len(x_inp[data_ind]))//2+(self.max_length-len(x_inp[data_ind]))%2
                    paddings = tf.constant([[pad_bef, pad_aft], [0, 0]])
                    padded_x = tf.pad(x_inp[data_ind], paddings, "CONSTANT") 
                    final_x.append(padded_x)

                if  len(y_inp[data_ind])>self.max_length:

                    tooMuch = len(y_inp[data_ind])-self.max_length
                    startTrunc = tooMuch//2
                    endTrunc = tooMuch//2+tooMuch%2
                    padded_y = tf.constant(y_inp[data_ind][startTrunc:-endTrunc])
                    final_y.append(padded_y)

                else:
                    pad_bef = (self.max_length-len(y_inp[data_ind]))//2
                    pad_aft = (self.max_length-len(y_inp[data_ind]))//2+(self.max_length-len(y_inp[data_ind]))%2
                    paddings = tf.constant([[pad_bef, pad_aft], [0, 0]])

                    padded_y = tf.pad(y_inp[data_ind], paddings, "CONSTANT") 
                    final_y.append(padded_y)

            if len(final_x)>1:
                yield (tf.reshape(final_x, shape=(len(final_x),self.max_length, 39)),tf.reshape(final_y, shape=(len(final_y),self.max_length, 39)))

    
    def get_batch_nb(self):
        return self.tot_batch

    def get_batch_pair_nb(self):
        return self.tot_batch_pair

    def get_max_len(self):
        return self.max_length
    
    def batch_2(self):
        final = []
        
        for data_ind in range(0, self.num_examples):

            max_length = np.max([len(self.data[x]) for x in range(0, self.num_examples-1)])

            paddings = tf.constant([[0, max_length-len(self.data[data_ind])], [0, 0]])
            
            padded = tf.pad(self.data[data_ind], paddings, "CONSTANT") 
            
            final.append(padded)

        x_inp = np.asarray([x for x in final if np.mean(x) !=0 and not math.isnan(np.mean(x))])
        return x_inp



train_dataset = Dataset("/home/getalp/leferrae/post_doc/corpora/cv-corpus-12.0-2022-12-07/pt/trainWords/")
test_dataset = Dataset("/home/getalp/leferrae/post_doc/corpora/cv-corpus-12.0-2022-12-07/pt/devWords/", max_length = train_dataset.get_max_len())

class Autoencoder(Model):
  def __init__(self, latent_dim, output_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      LSTM(latent_dim, return_sequences=True,activation="tanh"),
      LSTM(latent_dim, return_sequences=True,activation="tanh"),
      LSTM(latent_dim),
    ])
    self.decoder = tf.keras.Sequential([
      Dense(latent_dim),
      Dense(latent_dim),
      Dense(latent_dim),
      Dense(output_dim[0]*output_dim[1]),
      Reshape(output_dim)
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
  def make_emb(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return encoded

loss = tf.keras.losses.MeanSquaredError()

def grad(model, inputs, outputs):
    with tf.GradientTape() as tape:
        out = model(inputs)
        loss_value = loss(outputs, out)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), inputs, out

autoencoder = Autoencoder(400, (train_dataset.get_max_len(),39))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
# autoencoder.save_weights("./models/AE_{}_ckpt/ae_rand_model.ckpt".format(rep), overwrite=True)
global_step = tf.Variable(0)
n_epochs = 15

autoencoder.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
autoencoder.build((None,train_dataset.get_max_len(), 39))
autoencoder.summary()

eval_QbE(autoencoder, train_dataset.get_max_len())

for epoch in range(10):
    print("\nEpoch: ", epoch)

    batch_id = 0
    val_losses=[]
    
    for index, x in enumerate(test_dataset.batch(batch_size=50)):
        loss_value, grads, inputs, reconstruction = grad(autoencoder, x, x)
        print_progress_bar(index, test_dataset.get_batch_nb(), loss_value)
        val_losses.append(loss_value)
    print("\nvalidation loss : {}".format(np.mean(val_losses)))
    for index, x in enumerate(train_dataset.batch(batch_size=16)):
        
        loss_value, grads, inputs, reconstruction = grad(autoencoder, x, x)
        optimizer.apply_gradients(zip(grads, autoencoder.trainable_variables),
                            global_step)
        print_progress_bar(index, train_dataset.get_batch_nb()+1, loss_value)
        batch_id +=1
    eval_QbE(autoencoder, train_dataset.get_max_len())

autoencoder.save_weights("./ae_model.ckpt", overwrite=True)

for epoch in range(30):
    batch_id = 0
    val_losses=[]
    for index, tup_d in enumerate(train_dataset.batch_pairs(batch_size=32)):
        x,y = tup_d[0], tup_d[1]

        loss_value, grads, inputs, reconstruction = grad(autoencoder, x, y)
        optimizer.apply_gradients(zip(grads, autoencoder.trainable_variables),
                            global_step)
        print_progress_bar(index, train_dataset.get_batch_pair_nb()+1, loss_value)
        batch_id +=1
    for index, tup_d in enumerate(test_dataset.batch_pairs(batch_size=100)):
        x,y = tup_d[0], tup_d[1]
        loss_value, grads, inputs, reconstruction = grad(autoencoder, x, y)
        print_progress_bar(index, test_dataset.get_batch_pair_nb(), loss_value)
        val_losses.append(loss_value)
    print("\nvalidation loss : {}".format(np.mean(val_losses)))
    eval_QbE(autoencoder, train_dataset.get_max_len())
autoencoder.save_weights("./cae_model.ckpt", overwrite=True)
