import pandas as pd

class SkuManager:
    def __init__(self, map_path = "./models/sku_map.pkl"):
        self.__map_path = map_path

        self.__map = pd.read_pickle(self.__map_path)
        self.map = self.__map.copy()
        self.inverse_map = self.__map.copy()

        self.map.index = self.map['origin_sku_ID']
        self.map = self.map['sku_ID'].to_dict()

        self.inverse_map.index = self.inverse_map['sku_ID']
        self.inverse_map = self.inverse_map['origin_sku_ID'].to_dict()

        self.__num = max(self.inverse_map) + 1

    def get_label_by_ID(self, ID):
        if ID == 'NONE':
            return self.__num
        return self.map[ID]

    def get_ID_by_label(self, label):
        if label == self.__num:
            return 'NONE'
        return self.inverse_map[label]
    
    def get_num_sku(self):
        return self.__num

class UserManager:
    def __init__(self, label_encoder_path = './models/5_gram_user_label_encoder.pk',
                user_map_path = './models/user_map.csv'):
        from sklearn.externals import joblib
        self.label_encoder = joblib.load(label_encoder_path)
        self.map = pd.read_csv(user_map_path, index_col=0)

    def get_label_by_encoding(self, encoding):
        return self.label_encoder.transform([encoding])
        
    def get_encoding_by_label(self, label):
        return self.label_encoder.inverse_transform([label])

    def get_labels_by_encodings(self, encodings):
        return self.label_encoder.transform(encodings)

    def get_encodings_by_labels(self, labels):
        return self.label_encoder.inverse_transform(labels)

    def get_encoding_by_ID(self, ID):
        return self.map['attrs_combined'][ID]

    def get_ID_by_encoding(self, encoding):
        pass


if __name__ == '__main__':

    test = SkuManager()
    user = UserManager()

    print(test.get_label_by_ID('c8af744670'))
    print(test.get_ID_by_label(25276))

    print(user.get_encoding_by_label(1))
    print(user.get_label_by_encoding('-1__F__-1__1__-1__U__26-35__'))