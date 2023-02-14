import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, Embedding
import matplotlib.pyplot as plt 


def prepare_dataset(seq):
     input_vector = seq[:-1]
     target_vector = seq[1:]
     return input_vector, target_vector

raw = open("expression-examples.txt", 'rb').read()
text = raw.decode(encoding='utf-8')


vocabulary = np.array(sorted(set(text)))
tokenizer = {char:i for i,char in enumerate(vocabulary)} 

for i in range(20):
     char = vocabulary[i]
     token = tokenizer[char]

vector = np.array([tokenizer[char] for char in text])
vector = tf.data.Dataset.from_tensor_slices(vector)
sequences = vector.batch(100, drop_remainder=True) 

dataset = sequences.map(prepare_dataset) 

for inp, tar in dataset.take(1):
     print(inp.numpy())
     print(tar.numpy())
     inp_text = ''.join(vocabulary[inp])
     tar_text = ''.join(vocabulary[tar])
     print(repr(inp_text))
     print(repr(tar_text)) 

AUTOTUNE = tf.data.AUTOTUNE
 # buffer size 10000
 # batch size 64
data = dataset.batch(64, drop_remainder=True).repeat()
data = data.prefetch(AUTOTUNE)
 # steps per epoch is number of batches available
STEPS_PER_EPOCH = len(sequences)//64 
for inp, tar in data.take(1):
     print(inp.numpy().shape)
     print(tar.numpy().shape) 

model = keras.Sequential([
     # Embed len(vocabulary) into 64 dimensions
     Embedding(len(vocabulary), 64, batch_input_shape=[64,None]),
     # LSTM RNN layers
     LSTM(512, return_sequences=True, stateful=True),
     LSTM(512, return_sequences=True, stateful=True),
     # Classification head
     Dense(len(vocabulary))
 ])

model.summary()      

for example_inp, example_tar in data.take(1):
     example_pred = model(example_inp)
     print(example_tar.numpy().shape)
     print(example_pred.shape) 

model.compile(optimizer='adam', 
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

history = model.fit(data, 
                     epochs=5, 
                     steps_per_epoch=STEPS_PER_EPOCH) 

model.reset_states() 
model.save('expression-model.keras')


sample = 'lambda a, b, c, d:'
sample_vector = [tokenizer[s] for s in sample]
predicted = sample_vector
 # convert into tensor of required dimensions
sample_tensor = tf.expand_dims(sample_vector, 0) 
 # broadcast to first dimension to 64 
sample_tensor = tf.repeat(sample_tensor, 64, axis=0)

 # predict next 1000 characters
 # temperature is a sensitive variable to adjust prediction
temperature = 0.6

for i in range(1000):
     pred = model(sample_tensor)
     # reduce unnecessary dimensions
     pred = pred[0].numpy()/temperature
     pred = tf.random.categorical(pred, num_samples=1)[-1,0].numpy()
     predicted.append(pred)
     sample_tensor = predicted[-99:]
     sample_tensor = tf.expand_dims([pred],0)
     # broadcast to first dimension to 64 
     sample_tensor = tf.repeat(sample_tensor, 64, axis=0)

 # convert the integers back to characters
pred_char = [vocabulary[i] for i in predicted]
generated = ''.join(pred_char)
print(generated)