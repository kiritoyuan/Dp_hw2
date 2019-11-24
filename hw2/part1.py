#!/usr/bin/env python3
"""
part1.py

UNSW COMP9444 Neural Networks and Deep Learning

ONLY COMPLETE METHODS AND CLASSES MARKED "TODO".

DO NOT MODIFY IMPORTS. DO NOT ADD EXTRA FUNCTIONS.
DO NOT MODIFY EXISTING FUNCTION SIGNATURES.
DO NOT IMPORT ADDITIONAL LIBRARIES.
DOING SO MAY CAUSE YOUR CODE TO FAIL AUTOMATED TESTING.
"""

import torch

class rnn(torch.nn.Module):

    def __init__(self):
        super(rnn, self).__init__()

        self.ih = torch.nn.Linear(64, 128)
        self.hh = torch.nn.Linear(128, 128)
        
    def rnnCell(self, input, hidden):
        """
        TODO: Using only the above defined linear layers and a tanh
              activation, create an Elman RNN cell.  You do not need
              to include an output layer.  The network should take
              some input (inputDim = 64) and the current hidden state
              (hiddenDim = 128), and return the new hidden state.
        """
                # out, hiddenState = torch.nn.RNN(input, hidden)
        
        # input = self.ih(input.view(input.size(0) , input.size(1), -1))
        # rnn = torch.nn.RNN(64,128,2)
        # _, hiddenState = rnn(input,hidden)
        # hidden = self.hh(hiddenState)
        # return hidden
        x = self.ih(input)
        hid = self.hh(hidden)
        # tanh activation: tanh(wight_input * input + weight_hidden*hidden + bias)
        tanh = torch.nn.functional.tanh(x + hid + 0)
        return tanh

    def forward(self, input):
        hidden = torch.zeros(128)
        """
        TODO: Using self.rnnCell, create a model that takes as input
              a sequence of size [seqLength, batchSize, inputDim]
              and passes each input through the rnn sequentially,
              updating the (initally zero) hidden state.
              Return the final hidden state after the
              last input in the sequence has been processed.
        """
        # batchSize = input.size(1)
        
        # hidden0 = self.rnnCell(input, hidden)

        for i in input :
            x = self.rnnCell(i, hidden)
        return x
        

class rnnSimplified(torch.nn.Module):

    def __init__(self):
        super(rnnSimplified, self).__init__()
        """
        TODO: Define self.net using a single PyTorch module such that
              the network defined by this class is equivalent to the
              one defined in class "rnn".
        """
        self.net = torch.nn.RNN(64,128,num_layers=1)



    def forward(self, input):
        hidden = torch.zeros(128)
        out, hidden = self.net(input, hidden)
            
        return hidden

def lstm(input, hiddenSize):
    """
    TODO: Let variable lstm be an instance of torch.nn.LSTM.
          Variable input is of size [batchSize, seqLength, inputDim]
    """
    # hidden = (torch.zeros(input.size(0), input.size(1), hiddenSize), 
    # torch.zeros(input.size(0), input.size(1),hiddenSize))

    # lstm = torch.nn.LSTM(input.size(2),hiddenSize,2)
    # # lstm_out, (hn,cn) = lstm(input.view(len(input), input.size(0),-1), hidden)
    input_size = list(input.size())[1] #?to fix

    lstm = torch.nn.LSTM(input_size, hiddenSize, 1)
    
    return lstm(input)

def conv(input, weight):
    """
    TODO: Return the convolution of input and weight tensors,
          where input contains sequential data.
          The convolution should be along the sequence axis.
          input is of size [batchSize, inputDim, seqLength]
    """
    # conv1 = nn.Conv2d(input.size(1), 128,  kernel_size=5, stride=1)
    # for i in range(len(input)) :
    #     x = conv1(input[i])
    # input_size = input.size(2)
    # weight_size = len(weight)
    # conv = torch.nn.Conv1d(input_size, weight_size, kernel_size=8, padding=5)
    return torch.nn.Conv1d(input, weight)