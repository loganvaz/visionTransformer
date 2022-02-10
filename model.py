import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
#from keras.models import Sequential
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation, Reshape, Lambda, Input, Concatenate
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, ZeroPadding2D
import os
from tensorflow.keras.layers import ELU



from myTransformerModel import Transformer as Encoder
from myTransformerModel import embeddings as emb
#def __init__(self, key_query_val_input_dim, num_heads, ff_dim, dropoutRate)
"""

model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, kernel_initializer='he_normal'))
model.add(ELU(alpha=elu_alpha))
"""


def transformerEncoder(inputShape, L, kqv_input_dim, num_heads, ff_dim, drop_rate):#TODO (might be able to import it)
    embed_layer = emb(inputShape[0],inputShape[1])
    X_in = Input(shape = inputShape)
    print("is")
    print(inputShape)
    X = X_in
    
    X = embed_layer(X)

    out2 = Dense(256)(Flatten()(X))
    out2 = ELU(1.0)(out2)
    process1 = Dense(512)

    for l in range(L):
        enc = Encoder(kqv_input_dim, num_heads, ff_dim, drop_rate)
        X = enc(X)
        out2_l = process1(Flatten()(X))
        out2 = out2 + Dense(256)(out2_l)
        out2 = ELU(1.0)(out2)
    
    model = Model(inputs = X_in, outputs = [X, out2])
    return model
    #make transformer encoder model w/ positional encodings as described in paper
def convolutions(inputShape):#TODO

    

    X_in = Input(shape = inputShape)
    
    X = Conv2D(64, kernel_size =7, strides = 2, padding = 'valid')(X_in)
    X = BatchNormalization(axis = 3)(X)
    X = ELU(alpha=1.0)(X)

    X = MaxPooling2D((2,2), strides = (2,2))(X)

    X = Conv2D(128, kernel_size =5, strides = 2, padding = 'valid')(X)
    X = BatchNormalization(axis = 3)(X)
    X = ELU(alpha=1.0)(X)

    X = MaxPooling2D((2,2), strides = (2,2))(X)

    X = Conv2D(8, kernel_size = 1, strides = 1, padding = 'valid')(X)
    #linearly project it down for each 'pixel'



    model = Model(inputs = X_in, outputs = X)

    return model

    
    #make model, return after the two-three convs (see notebook, we want to reduce size somewhat but not completely)
def skip_connect(inputShape, numStart, mid, drop_rate):
    X_in = Input(shape = inputShape)

    X = X_in

    mid = Dense(mid)(X)
    mid = ELU(1.0)(mid)
    mid = Dropout(drop_rate)(mid)

    final = Dense(numStart)(mid)
    final = Dropout(drop_rate)(final) + X
    final = ELU(1.0)(final)

    model = Model(inputs = X_in, outputs = final)

    return model

    

    

    
def feed_forward(inputShape, outputNum):
    drop_rate = 0.1
    
    X_in = Input(shape = inputShape)
    flat = Flatten()(X_in)

    X = Dense(2048)(flat)
    X = ELU(1.0)(X)
    X = Dropout(drop_rate)(X)

    X_0 = Dense(128)(X)

    connect_1 = skip_connect(X.shape[1:], 2048, 2048,drop_rate)

    X = connect_1(X)
    print(connect_1.summary())

    connect_2 = skip_connect(X.shape[1:], 2048, 1024,drop_rate)
    X = connect_2(X)

    X_ = X

    X = Dense(1024)(X)
    X = ELU(1.0)(X)
    X = Dropout(drop_rate)(X)

    connect_3 = skip_connect(X.shape[1:], 1024, 1024,drop_rate)
    X = connect_3(X)

    connect_4 = skip_connect(X.shape[1:], 1024, 512, drop_rate)
    X = connect_4(X)


    X = Concatenate()([X, Dense(256)(X_), X_0])

    predictions = X
    #predictions = Dense(outputNum, activation = 'softmax')(X)

    model = Model(inputs = X_in, outputs = predictions)

    return model

    

    

    
def imageTransformer(inputShape, outputNum):
    #TODO: define transformer params up here
    numLayers = 4#continue transformer params here (num_heads, FF size, dropout, etc.)
    kqv_input_dim = 512
    num_heads = 16
    ff_dim = 2048
    drop_rate = 0.1

    ###end transformer stuff


    
    X_in = Input(shape = inputShape)

    convs = convolutions(inputShape)

    print("convs summary")
    print(convs.summary())

    X = convs(X_in)

    X_patch = tf.image.extract_patches(images = X, sizes = [1, 8, 8, 1], rates = [1,1,1,1], strides = [1,6,6,1], padding = 'VALID', name = 'image_extraction')

    print("patch shape")
    print(X_patch.shape)
    #print(X_patch.shape)
    X_patch_flattened = tf.reshape(X_patch,(-1,X_patch.shape[1]*X_patch.shape[2], X_patch.shape[3]))#X_patch.reshape(X_patch.shape[0], -1, X_patch.shape[3]) numpy works so tf should too
    #print(X_patch_flattened.shape)
    print("flattened shape")
    print(X_patch_flattened.shape)

    inputShape2 = X_patch_flattened.shape[1:]#tf.shape(X_patch_flattened)
    print("sh2")
    print(X_patch_flattened.shape)
    print(inputShape2)

    transformer_block = transformerEncoder(inputShape2, numLayers, kqv_input_dim, num_heads, ff_dim, drop_rate)

    print("Transformer summart")
    print(transformer_block.summary())

    print("shape is")
    print(X_patch_flattened.shape)

    X, add_end = transformer_block(X_patch_flattened)

    
    #inputShape3 = tf.shape(X)
    inputShape3 = X.shape[1:]
    
    vanilla = feed_forward(inputShape3, outputNum)
    print("vanilla summary")
    print(vanilla.summary())

    output= vanilla(X)

    output = Concatenate()([output, add_end])

    connect = skip_connect(output.shape[1:], output.shape[1], output.shape[1], 0.2)

    output = connect(output)

    output = Dense(outputNum, activation = 'softmax')(output)

    model = Model(inputs = X_in , outputs = output)

    return model

    

    

    


    #encoder = transformerEncoder(inputShape, numLayers,)#TODO: add other params, see other computer

    #fully connected, res layers -> prediction, TODO

    
    
    
print("kenobi")
X = imageTransformer((512,512,3), 10)

    
