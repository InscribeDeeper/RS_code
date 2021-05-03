from sklearn.base import BaseEstimator


class AList:
    '''
    A custom-designed list
    '''
    def __init__(self, *l):
        '''
        initialize a list
        
        input: a sequence of object

        >>> AList(1,2,3)
        >>> "A list which has [1,2,3]"
        '''
        self.store = [element for element in l]
    
    def __repr__(self):
        return 'A list which has {}'.format(self.store)

    def __getitem__(self, i):
        '''
        index the ith element of the list
        '''
        return self.store[i]

    def __call__(self, k):
        '''
        multiply every element of list wich k
        '''
        return [k*element for element in self.store]

    def __add__(self, list2):
        '''
        vector addition
        '''
        l = len(list2)
        return [self.store[i] + list2[i] 
                for i in range(l)]


class CustomEstimator(BaseEstimator):
    def __init__(self, w, **kwargs):
        super(BaseEstimator, self).__init__(**kwargs) # Use the initialization method of BaseEstimator
        self.w_ = w
    

    def predict(self, x):
        from numpy import dot
        return dot(self.w_, x)


if __name__ == '__main__':

    test = AList(1,2,3)
    print(test)
    print(test[1])
    print(test(10))
    print(test+[10,11,12])


    model = CustomEstimator([1,2,3])
    print(model.predict([1,2,3]))




