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
        x = self.ih(input)
        hid = self.hh(hidden)
        # tanh activation: tanh(wight_input * input + weight_hidden*hidden + bias)
        tanh = torch.tanh(x + hid )
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
        for i in input :
            hidden = self.rnnCell(i, hidden)
        return hidden
        

class rnnSimplified(torch.nn.Module):

    def __init__(self):
        super(rnnSimplified, self).__init__()
        """
        TODO: Define self.net using a single PyTorch module such that
              the network defined by this class is equivalent to the
              one defined in class "rnn".
        """
        self.net = torch.nn.RNN(64,128)



    def forward(self, input):
        
        out, hidden = self.net(input)           
        return hidden

def lstm(input, hiddenSize):
    """
    TODO: Let variable lstm be an instance of torch.nn.LSTM.
          Variable input is of size [batchSize, seqLength, inputDim]
    """
    input_size = input.shape[-1]    #inputDim

    # lstm = torch.nn.LSTM(input_size, hiddenSize, 1)
    lstm = torch.nn.LSTM(input_size, hiddenSize, batch_first=True)
    
    return lstm(input)

def conv(input, weight):
    """
    TODO: Return the convolution of input and weight tensors,
          where input contains sequential data.
          The convolution should be along the sequence axis.
          input is of size [batchSize, inputDim, seqLength]
    """
    # src :https://pytorch.org/docs/stable/nn.functional.html
    #  torch.nn.functional.conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
    # weight tensor is of shape (out_channels, in_channels/groups, kernel_size)
    out_c, in_g, kernel_size = weight.shape
    in_channels = input.shape[1]    #inputDim
    conv = torch.nn.Conv1d(in_channels, out_c, kernel_size, bias=False)
    conv.weight.data = weight
    return conv(input)
