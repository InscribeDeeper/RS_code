# -*- coding: utf-8 -*-

import torch
from torch import nn
from tqdm import tqdm

#device = torch.device("cuda:0")

class Sequence2Vector(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding1 = nn.Embedding(embedding_dim=embedding_dim, num_embeddings=num_embeddings)

    
    def forward(self, input_tensor1, input_tensor2):

        center_embedding = self.to_vector(input_tensor1)
        context_embedding = self.to_vector(input_tensor2)

        length = center_embedding.shape[0]
        output = []

        for i in tqdm(range(length)):
            inner_product = torch.matmul(context_embedding[i, :, :], center_embedding[i, :, :].transpose(1,0))
            output.append(inner_product)
        print(output); print(inner_product.shape); print(input_tensor1.shape)
        return torch.stack(output)


    def to_vector(self, input_tensor):
        return self.embedding1(input_tensor)




if __name__ == "__main__":
    print('run')
#    x = torch.LongTensor([1,2,3])
    x = torch.LongTensor([[2], [1], [3]])
    y = torch.LongTensor([[2, 2, 2], [1, 1, 1], [3, 1, 1]])

    model = Sequence2Vector(4, 2)
    X = model.to_vector(x)
    Y = model.to_vector(y)
    print(X)
 #   print(Y)


    print(torch.sum(model(x, y), axis = 0))

