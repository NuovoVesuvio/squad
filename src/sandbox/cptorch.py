'''
Tester code
'''


import torch, pickle, sys, argparse
from termcolor import colored
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

    def initHidden(self, value):
        return Variable(torch.from_numpy(np.random.uniform(low=-value, high=value, size=(1,self.hidden_size))).float())


def getNorm(i2o):
    return np.linalg.norm(i2o, ord='fro')



def loaddata(embeddings='../../data/lemmas_embeddings.pickle', text='../../data/lemmas.pickle'):
    # -- import embeddings and vocab
    with open(embeddings, 'rb') as f:
        data = pickle.load(f)

    # -- load lemma text
    with open(text, 'rb') as g:
        lemma = pickle.load(g)

    return data, lemma


def train(embedding_size, hidden_dim, gradclip, initalrange, lr, momentum, weight_decay):
    # -- load the data()
    data, lemma = loaddata()
    embeddings = data['embeddings'].astype(np.float32)
    vocab = data['vocab']
    reverse = {v:k for k,v in vocab.items()}

    # -- initalize network
    rnn = RNN(embedding_size, hidden_dim, len(vocab)).cuda()
    criterion = nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
    optimizer = optim.SGD(rnn.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # -- train the network 
    for idx, sentence in enumerate(lemma):
        hidden = rnn.initHidden(initalrange)
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
                torch.nn.utils.clip_grad_norm(rnn.parameters(), gradclip)

            except KeyError as e:
                miss += 1
                pass
        threshold = (len(sentence)) - miss >= 2 # at least two words in sentence are in vocab, then optimize
        print(idx, threshold, ' '.join(sentence))
        if threshold: 
            loss.backward()
            optimizer.step()
        
        # -- evaluate with predicited sentences
        psent = []    
        if idx % 100 == 0:
            # -- test the output
            for j in range(10):
                _hidden = rnn.initHidden(initalrange )
                if j == 0:
                    word = 'the'
                    next_word = torch.from_numpy(embeddings[vocab[word]]).unsqueeze(0)
                psent.append(word)
                _output, dummy_hidden = rnn(Variable(next_word).cuda(), _hidden.cuda())
                probs = np.exp(_output.data.cpu().numpy().ravel())
                amax = np.argmax(probs)
                word = reverse[amax]
                next_word = torch.from_numpy(embeddings[vocab[word]]).unsqueeze(0)
            print(colored(','.join(psent), 'red'))
            i2o = rnn.i2o.weight.grad.data.cpu().numpy()
            i2h = rnn.i2h.weight.grad.data.cpu().numpy()
            o2o = rnn.o2o.weight.grad.data.cpu().numpy()
            norm_i2o = getNorm(i2o)
            norm_i2h = getNorm(i2h)
            norm_o2o = getNorm(o2o)
            print(colored('Norms: %.2f,\t%.2f,\t%.2f' % (norm_i2o, norm_i2h, norm_o2o), 'red'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RNN training and network')
    parser.add_argument('--embedding-size', dest='embedding_size', default=100, type=int, help='embedding size')
    parser.add_argument('--hidden-dim', dest='hidden_dim', default=100, type=int, help='hidden layer size')
    parser.add_argument('--gradclip', dest='gradclip', default=0.5, type=float, help='gradient clip coeffient')
    parser.add_argument('--initalrange', dest='initalrange', default=0.0001, type=float, help='weight initalization range')
    parser.add_argument('--lr', dest='lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.5, help='momentum')
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=0.2)
    args = parser.parse_args()
    train(args.embedding_size, args.hidden_dim, args.gradclip, args.initalrange, args.lr, args.momentum, args.weight_decay)








