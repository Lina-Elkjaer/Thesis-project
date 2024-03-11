import torch 
import torch.nn as nn
from torch.autograd import Variable

#https://cnvrg.io/pytorch-lstm/
#https://machinelearningmastery.com/building-a-binary-classification-model-in-pytorch/

class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc1 =  nn.Linear(hidden_size, 128) #fully connected first layer
        self.relu = nn.ReLU()
        self.fc_last = nn.Linear(128, num_classes) #fully connected last layer
        self.sigmoid = nn.Sigmoid()
        #self.softmax = nn.Softmax(dim = 1) 
    

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda() #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda() #internal state
        
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next

        #dense layer
        out = self.relu(hn)
        out = self.fc1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc_last(out)

        #output through softmax to get predictions between 0 and 1 
        out = self.sigmoid(out)

        return out

