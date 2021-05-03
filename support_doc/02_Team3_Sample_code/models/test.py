from sequence2vector import Sequence2VectorWithBuyer, CrossInnerProduct
from sequence2vector import negative_log_likelihood

import numpy as np
import tensorflow.keras.backend as K

if __name__ == "__main__":
    center = np.array([[0], [10], [20], [0]], dtype=int)
    windows = np.array([[1,2,3,4], [11,12,13,14], [21,22,23,23], [0,1,1,3]], dtype=int)
    negtive = np.array([[11,21,31,14], [21,2,23,4], [1,2,13,13], [11,21,21,13]], dtype=int)
    buy = np.array([[1], [11], [21], [1]], dtype=int)

    model = Sequence2VectorWithBuyer(40, 3)
    model.compile(optimizer='adam', loss = negative_log_likelihood)
    model.fit(x=[center, windows, negtive, buy], y=np.array([1,1,1,1]), epochs=3000, batch_size=2)

    test1 = model.to_vector(np.array([0]))
    test2 = model.to_vector(np.array([1]))
    test3 = model.to_vector(np.array([11]))
    test4 = model.to_vector(np.array([14]))

    print(K.eval(test1))
    print(K.eval(test2))
    print(K.eval(test3))
    print(K.eval(K.dot(test1, K.transpose(test3))))
    print(K.eval(K.dot(test1, K.transpose(test2))))
    print(K.eval(K.dot(test3, K.transpose(test4))))
    print(K.eval(K.dot(test4, K.transpose(test2))))