import tensorflow as tf
from tensorflow.keras import layers, activations, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow import Tensor




class RecommendationSystem(Model):
    def __init__(self, num_sku, num_user, embedding_dim = 100,
                sku2vec_path = './models/5_gram_sku2vec_v2.h5', 
                *args, **kwargs):
        super().__init__(*args, **kwargs)
        sku2vec = Sequential(layers.Embedding(num_sku, embedding_dim))
        sku2vec.load_weights(sku2vec_path)
        self.sku2vec_layer = sku2vec
        self.user2vec_1 = Sequential(layers.Embedding(num_user, 100), name='usr2vec1')
        self.user2vec_2 = Sequential(layers.Embedding(num_user, 100))
        self.user2vec_3 = Sequential(layers.Embedding(num_user, 100))

        self.diff_layer = layers.Lambda(lambda x: K.abs(x[0] - x[1]))
        self.concat_layer = layers.Concatenate()
        #self.reshape = layers.Reshape((-1,))
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1, activation='relu')
        self.output_layer = layers.Dense(1, activation='sigmoid')


    def call(self, input_tensors):
        user, sku = input_tensors

        sku_vec = self.sku2vec_layer(sku)
        usr_vec1 = self.user2vec_1(user)
        usr_vec2 = self.user2vec_2(user)
        usr_vec3 = self.user2vec_3(user)

        taste_diff_1 = self.diff_layer([sku_vec, usr_vec1])
        taste_diff_2 = self.diff_layer([sku_vec, usr_vec2])
        taste_diff_3 = self.diff_layer([sku_vec, usr_vec3])


        concat = self.concat_layer([taste_diff_1, taste_diff_2, taste_diff_3])
        #concat = self.reshape(concat)
        flat = self.flatten(concat)
        print(flat.shape)
        output = self.output_layer(flat)

        return output


def build_recommendation_model(num_user, num_sku, path = './models/5_gram_sku2vec_v2.h5'):
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.activations import sigmoid
    from tensorflow.keras.layers import Embedding, Input, Lambda, Concatenate, Dense, Flatten, Reshape
    from tensorflow.keras import backend as K

    def build_seq2vec(num, embedding_dim, path=False, trainable=True):
        input_ = Input(shape=(1,))
        output = Embedding(num, embedding_dim, trainable=trainable)(input_)
        model = Model(inputs=input_, outputs=output)
        if path:
            model.load_weights(path)
        return model
    

    sku2vec = build_seq2vec(num_sku, 100, path)
    sku2vec.load_weights(path)

    user2vec_1 = build_seq2vec(num_user, 100)
    user2vec_2 = build_seq2vec(num_user, 100)
    user2vec_3 = build_seq2vec(num_user, 100)

    diff_layer = Lambda(lambda x: K.abs(x[0] - x[1]))
    concat_layer = Concatenate(axis=-1)
    dense_output = Dense(1, activation='sigmoid')
    means_layer = Lambda(lambda x: (x[0] + x[1] + x[2])/3)

    recommen_sku = Input(shape=(1))
    recommen_user = Input(shape=(1))

    sku_vec = sku2vec(recommen_sku)
    user_vec1 = user2vec_1(recommen_user)
    user_vec2 = user2vec_2(recommen_user)
    user_vec3 = user2vec_3(recommen_user)

    interest_diff_1 = Flatten()( Lambda(lambda x: K.abs(x[0] - x[1]))([sku_vec, user_vec1]))
    interest_diff_2 = Flatten()( Lambda(lambda x: K.abs(x[0] - x[1]))([sku_vec, user_vec2]))
    interest_diff_3 = Flatten()( Lambda(lambda x: K.abs(x[0] - x[1]))([sku_vec, user_vec3]))

    concat = concat_layer([interest_diff_1, interest_diff_2, interest_diff_3])
    flat = Flatten()(concat)
    output = dense_output(flat)


    recommendation_system = Model(inputs = [recommen_user, recommen_sku], outputs=output)
    return recommendation_system

if __name__ == "__main__":
    test = RecommendationSystem(32195, 4000, 100, '5_gram_sku2vec_v2.h5')
    test([[1], [2]])
    #print(test([[1], [2]]))