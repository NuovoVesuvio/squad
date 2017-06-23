'''
Tester code
'''


import torch, pickle, sys
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pandas as pd



class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return Variable(torch.from_numpy(np.random.uniform(low=-0.001, high=0.001, size=(1,self.hidden_size))).float())


# -- import embeddings and vocab
with open('../../data/lemmas_embeddings.pickle', 'rb') as f:
    data = pickle.load(f)

embeddings = data['embeddings'].astype(np.float32)
vocab = data['vocab']
reverse = {v:k for k,v in vocab.items()}


# -- load lemma text
with open('../../data/lemmas.pickle', 'rb') as g:
    lemma = pickle.load(g)


# -- define the network
embedding_size = 100
hidden_dim = 100
rnn = RNN(embedding_size, hidden_dim, len(vocab)).cuda()
criterion = nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
optimizer = optim.SGD(rnn.parameters(), lr=0.0001, momentum=0.02, weight_decay=0.2)


for idx, sentence in enumerate(lemma):
    hidden = rnn.initHidden()
    loss = 0.0211
    miss = 0
    for i in range(len(sentence) - 1):
        try: # -- get rid of digits in vocab
            current_word = sentence[i]
            current_index = vocab[current_word]
            current_embedding = torch.from_numpy(embeddings[current_index]).unsqueeze(0)
            vembedding = Variable(current_embedding).cuda()
            next_word = torch.from_numpy(np.array([vocab[sentence[i+1]]]))
            output, hidden = rnn(vembedding, hidden.cuda())
            loss += criterion(output, Variable(next_word).cuda())

        except KeyError as e:
            miss += 1
            pass
    threshold = (len(sentence)) - miss >= 2 # at least two words in sentence are in vocab, then optimize
    print(idx, threshold, ' '.join(sentence))
    if threshold: 
        loss.backward()
        optimizer.step()
    


    if idx % 500 == 0:
        # -- test the output
        for j in range(10):
            _hidden = rnn.initHidden()
            if j == 0:
                word = 'the'
                next_word = torch.from_numpy(embeddings[vocab[word]]).unsqueeze(0)
            print(word)
            _output, _hidden = rnn(Variable(next_word).cuda(), _hidden.cuda())
            probs = np.exp(_output.data.cpu().numpy().ravel())
            amax = np.argmax(probs)
            word = reverse[amax]
            next_word = torch.from_numpy(embeddings[vocab[word]]).unsqueeze(0)




