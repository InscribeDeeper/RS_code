# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers, activations, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow import Tensor

def negative_log_likelihood(y_true, y_pred):
    return -K.mean(y_pred)


class CrossInnerProduct(layers.Layer):
    '''
    compute inner product between center item and other item
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        '''
        把center和windows以及negative sampling里每一个向量做内积
        
        input:
        ======================================
        center_vec: tf.Tensor ----> with shape [data_size, embedding_dim, 1]
                centor item vector after embedding
        windows_vecs: tf.Tensor ----> with shape [data_size, embedding_dim, window_size]
                a list of context item vector
        neg_vecs: tf.Tensor ----> with shape [data_size, embedding_dim, negative_sampling_size]
                a list of embedding vector of negative sampling item

        output:
        ======================================
        output: tf.Tensor
                inner product between center item vector and other input vector
        '''
        from tensorflow import stack, transpose
        center_vec, windows_vecs, neg_vecs = inputs

        data_size = center_vec.shape[0] # size of data [数据集大小]
        win_size = windows_vecs.shape[1]  # size of window [n gram 中 window大小]
        neg_size = neg_vecs.shape[1] # number of negative sampling [负采样数量]

        output1 = []
        output2 = []
        if not data_size is None:
            for i in range(data_size):
                inner_product_center_window = K.dot(windows_vecs[i,:,:], transpose(center_vec[i,:,:]))
                inner_product_center_negative = -K.dot(neg_vecs[i,:,:], transpose(center_vec[i,:,:]))
            
                output1.append(inner_product_center_window)
                output2.append(inner_product_center_negative)
        else:
            output1 = K.dot(windows_vecs, transpose(center_vec, perm=(0,2,1)))
            output2 = -K.dot(neg_vecs, transpose(center_vec, perm=(0,2,1)))
        

        return K.concatenate([stack(output1), stack(output2)], axis = 1)


class CrossInnerProductWithBuyer(layers.Layer):
    '''
    compute inner product between center item and other item
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        '''
        把center和windows以及negative sampling里每一个向量做内积
        
        input:
        ======================================
        center_vec: tf.Tensor ----> with shape [data_size, embedding_dim, 1]
                centor item vector after embedding
        windows_vecs: tf.Tensor ----> with shape [data_size, embedding_dim, window_size]
                a list of context item vector
        neg_vecs: tf.Tensor ----> with shape [data_size, embedding_dim, negative_sampling_size]
                a list of embedding vector of negative sampling item

        output:
        ======================================
        output: tf.Tensor
                inner product between center item vector and other input vector
        '''
        from tensorflow import stack, transpose
        center_vec, windows_vecs, neg_vecs, buy_vec = inputs

        data_size = center_vec.shape[0] # size of data [数据集大小]
        win_size = windows_vecs.shape[1]  # size of window [n gram 中 window大小]
        neg_size = neg_vecs.shape[1] # number of negative sampling [负采样数量]

        output1 = []
        output2 = []
        output3 = []
        if not data_size is None:
            for i in range(data_size):
                inner_product_center_window = K.dot(windows_vecs[i,:,:], transpose(center_vec[i,:,:]))
                inner_product_center_negative = -K.dot(neg_vecs[i,:,:], transpose(center_vec[i,:,:]))
                inner_product_center_buy = K.dot(center_vec[i,:,:], transpose(buy_vec[i,:,:]))
            
                output1.append(inner_product_center_window)
                output2.append(inner_product_center_negative)
                output3.append(inner_product_center_buy)
        else:
            output1 = K.dot(windows_vecs, transpose(center_vec, perm=(0,2,1)))
            output2 = -K.dot(neg_vecs, transpose(center_vec, perm=(0,2,1)))
            output3 = K.dot(buy_vec, transpose(center_vec, perm=(0,2,1)))
        
        return K.concatenate([stack(output1), stack(output2), stack(output3)], axis = 1)

    
#    def __call__(self, *args):
#        output = self.call(*args)
#        return output



class BinomialProbability(layers.Layer):
    '''
    Compute negative log likelihood (Binomial) using inner product.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    def call(self, inner_product_result):
        '''
        Compute negative log likelihood and add it as loss
        
        input:
        ======================================
        inner_product_result: tf.Tensor 
                inner product between center vector and other vector
                with shape [size of data, window_size + negative_sampling_size + 1, 1]

        output:
        ======================================
        output: tf.Tensor
                Negative log likelihood of generating window
        '''
        from tensorflow.keras.activations import sigmoid
        _ = 1e-10
        binomial_probability = sigmoid(inner_product_result)
        log_likelihood = K.log(binomial_probability+_)
        #minus_log_likelihood = K.sum(log_likelihood)
        #self.add_loss(minus_log_likelihood, inputs = inner_product_result)
        print(log_likelihood)
        print(log_likelihood.shape)
        return log_likelihood 



class Sequence2Vector(Model):
    def __init__(self, num_items, embedding_dim, penalty = 1e-4, name=''):
        super().__init__(name=name)

        self.embedding_layer = layers.Embedding(input_dim=num_items, 
                                                output_dim=embedding_dim,
                                                embeddings_regularizer = regularizers.l2(penalty))
        self.inner_product_layer = CrossInnerProduct()
        self.binomial_probability = BinomialProbability()

    def call(self, inputs):

        x_center, x_positive, x_negative = inputs

        v_center = self.to_vector(x_center)
        v_pos = self.to_vector(x_positive)
        v_neg = self.to_vector(x_negative)
        
        inner_product_result = self.inner_product_layer([v_center, v_pos, v_neg])
        likelihood = self.binomial_probability(inner_product_result)

        return likelihood


    
    def to_vector(self, input_tensor):
        '''
        convert item into vector
        '''
        if not isinstance(input_tensor, Tensor):
            input_tensor = K.variable(input_tensor)
        embedding_vector = self.embedding_layer(input_tensor)
        return embedding_vector


class Sequence2VectorWithBuyer(Sequence2Vector):
    def __init__(self, num_items, embedding_dim, penalty = 1e-3, name=''):
        super().__init__(num_items, embedding_dim, penalty)
        self.inner_product_layer = CrossInnerProductWithBuyer()

        

    def call(self, inputs):
    
        x_center, x_positive, x_negative, x_buy = inputs

        v_center = self.to_vector(x_center)
        v_pos = self.to_vector(x_positive)
        v_neg = self.to_vector(x_negative)
        v_buy = self.to_vector(x_buy)
        
        inner_product_result = self.inner_product_layer([v_center, v_pos, v_neg, v_buy])
        likelihood = self.binomial_probability(inner_product_result)

        return likelihood    






    if __name__ == '__main__':

        '''
        测试内积层
        '''
        from keras.layers import Embedding
        e = Embedding(input_dim=100, output_dim=5)
        # 初始话内积层
        test_layer = CrossInnerProduct()
        likli = BinomialProbability()
        # 做一个center向量
        center = e(K.variable([[1], [3]]))
        vec1s = e(K.variable([[3,1,4], [1,1,4]]))
        vec2s = e(K.variable([[2,5,1], [3,2,1]]))
        buy1 = e(K.variable([[7], [0]]))

        # windows宽度为1
        windowsvec = [K.variable([[2,2,2,2.0]]),
                        K.variable([[0,2.0,1,0]])]

        # 随便输入两个负采样向量
        negvec = [K.variable([[2,0,0,2]]),
                        K.variable([[0,2,2.0,0]])]

        # 购买商品的向量
        buy = K.variable([[1,1,1,.01]])

        # 测试计算结果
        result = test_layer(center, vec1s, vec2s, buy1)
        prob = likli(result)
        print(K.eval(result))
        print(result.shape)
        print(K.eval(prob))

        center = [[0], [10], [20], [0]]
        windows = [[1,2,3,4], [11,12,13,14], [21,22,23,23], [0,1,1,3]]
        negtive = [[11,21,31,14], [21,2,23,4], [1,2,13,13], [10,21,21,13]]
        buy = [[1], [11], [21], [1]]

        model = Sequence2Vector(40, 2)
        model.compile(optimizer='adam', loss = empty_loss)
        model.fit(x=[center, windows, negtive, buy], y=[[1],[1],[1],[1]], epochs=10)
        print(model.to_vector([0]))
        print(model.to_vector([1]))

