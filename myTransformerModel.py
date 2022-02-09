#resources used heavily
#https://medium.com/analytics-vidhya/understanding-positional-encoding-in-transformers-def92aca1dfe
#https://github.com/keras-team/keras-io/blob/master/examples/nlp/text_classification_with_transformer.py



import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from tensorflow.keras.layers import Conv2D


#positional encodings from  on https://medium.com/analytics-vidhya/understanding-positional-encoding-in-transformers-def92aca1dfe




class PositionalEncodings(layers.Layer):#remembers, thse are sub_classes of layers.Layer cass (why super)
    def __init__(self, sequenceLength, d_model):
        super(PositionalEncodings, self).__init__()
        self.sequenceLength =sequenceLength
        self.d_model = d_model

    def get_angles(self, pos, i, embedding_size):
        #// divide -> int
        angles_times_pos = pos / np.power(10000., (2*(i//2))/np.float32(embedding_size))
        #formula from paper
        return angles_times_pos

    def call(self, inputs):
        #inputs are m by sequenceLength by modelDimension (m, t, k)
        sequenceLength = self.sequenceLength
        #print(sequenceLength)
        d_model = self.d_model
        #newaxis explicitly makes it column vector
        #it is used to make sure shape is correct
        """
        print(np.arange(sequenceLength)[:, np.newaxis])
        print(np.arange(d_model)[np.newaxis, :])
        print(d_model)
       
        print("getting angles")
        """
        angles = self.get_angles(np.arange(sequenceLength)[:,np.newaxis], np.arange(d_model)[np.newaxis,:], d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        #print(angles.shape)
        positionalEncodings = angles[np.newaxis, ...]
        """
        print(positionalEncodings.shape)
        print("input shape")
        print(inputs.shape)
        print("add and return")
        """
        return inputs + tf.cast(positionalEncodings, tf.float32)




#used the following as a template: https://github.com/keras-team/keras-io/blob/master/examples/nlp/text_classification_with_transformer.py



class embeddings(layers.Layer):
    def __init__(self, sequenceLength, d_model):
        super(embeddings, self).__init__()
        #if working wih NLP or something, probably would want token embeddings
        
        self.positional_embeddings = PositionalEncodings(sequenceLength, d_model)

    def call(self, x):
        #if working w/ NLP or something, would probably want ot have token_embeddings
        positions = self.positional_embeddings(x)
        return positions

#possible problem: setting variable not in init, if getting error try just returning set_encodeVals
class Transformer(layers.Layer):#should rlly be encoder
    def __init__(self, key_query_val_input_dim, num_heads, ff_dim, dropoutRate):
        super(Transformer, self).__init__()
        self.multiHeadEncoder = layers.MultiHeadAttention(num_heads = num_heads, key_dim = key_query_val_input_dim)
        self.ffEncoder = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(key_query_val_input_dim),]

            )

        #above applies dense -> ff_dim w/ relu then matrix multiplication to input dimension so can add and such

        self.layernorm1Encoder = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2Encoder = layers.LayerNormalization(epsilon = 1e-6)
        self.dropout1Encoder = layers.Dropout(dropoutRate)
        self.dropout2Encoder = layers.Dropout(dropoutRate)

        self.encodeVals = None
        
    def set_encodeVals(self, inputs, bool_train):
        attention = self.multiHeadEncoder(inputs, inputs)
        attention = self.dropout1Encoder(attention, training=bool_train)

        out1 = self.layernorm1Encoder(inputs+attention)
   
        feed_forward = self.ffEncoder(out1)
       
        ff = self.dropout2Encoder(feed_forward)
 
        self.encodeVals = self.layernorm2Encoder(out1+ff)

    def call(self, inputs, training):#NOTE, THIS DOESN'T IMPLEMENT DECODER
        self.set_encodeVals(inputs, training)
        return self.encodeVals
        
    
        
#alright, let's do some testing

def model(inputSize, maxLength, embeddingDimensions, num_heads, ff_dim):
    inputs = layers.Input(shape=(maxLength,inputSize))
    #embeddings_layer = embeddings(maxLength, inputSize, embeddingDimensions)
    embeddings_layer = embeddings(maxLength, inputSize)
    x = embeddings_layer(inputs)

    
    #transformer_block = Transformer(embeddingDimensions, num_heads, ff_dim, 0.2)
    print("x shape pre transformer")
    print(x.shape)
    transformer_block = Transformer(inputSize, num_heads, ff_dim, 0.2)
    x = transformer_block(x)

    x = tf.keras.layers.Reshape((maxLength, -1, 1))(x)
    #x = Conv2D(2, kernel_size=(512,1),strides=(64,1),padding="valid")(x)

    
    x = tf.keras.layers.Flatten()(x)

    
    x = layers.Dense(512, activation="relu")(x)
    x_ = layers.Dense(208, activation ="relu")(x)
    x = layers.Activation("relu")(layers.Dense(104)(x_)+layers.Dense(104)(x))

    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(8, activation="softmax")(x)
    """
    #below might not be best for heartbeat analysis, I think it's used for sentiment w/ NLP, keep for now so can g=check params
    #below was 1d average pooling
    print(x.shape)
    x = tf.keras.layers.Reshape((maxLength, -1, 1))(x)
    print(x.shape)
    x = Conv2D(8, kernel_size=(32,1),strides=(8,1),padding="valid")(x)
    x = tf.keras.layers.Flatten()(x)

    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(2, activation="softmax")(x)


    """

    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

myModel = model(32, 200, 32, 2, 32)
print(myModel.summary())

