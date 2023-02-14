#%%
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, Embedding
import matplotlib.pyplot as plt 

raw = open("expression-examples.txt", 'rb').read()
text = raw.decode(encoding='utf-8')


vocabulary = np.array(sorted(set(text)))
tokenizer = {char:i for i,char in enumerate(vocabulary)} 

model = keras.models.load_model('expression-model.keras')
model.summary()      

sample = 'lambda a, b, c, d:'
sample_vector = [tokenizer[s] for s in sample]
predicted = sample_vector
sample_tensor = tf.expand_dims(sample_vector, 0) 
sample_tensor = tf.repeat(sample_tensor, 64, axis=0)

temperature = 0.6

for i in range(10000):
     pred = model(sample_tensor)
     pred = pred[0].numpy()/temperature
     pred = tf.random.categorical(pred, num_samples=1)[-1,0].numpy()
     predicted.append(pred)
     sample_tensor = predicted[-99:]
     sample_tensor = tf.expand_dims([pred],0)
     sample_tensor = tf.repeat(sample_tensor, 64, axis=0)

 # convert the integers back to characters
pred_char = [vocabulary[i] for i in predicted]
generated = ''.join(pred_char)

#print(generated)
expressions = generated.splitlines()
all=0
error =0
for line in expressions:
     all +=1
     print(line)
     try:
        func = eval(line)
        result = func(19, 22, 32, 14)
        print(result)
     except: # catch *all* exceptions
        print("Error in Expression")
        error += 1
     
print (error, "error from overall ", all)     
     

# %%
