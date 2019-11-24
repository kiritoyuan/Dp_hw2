import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as topti
from torchtext import data
from torchtext.vocab import GloVe
from imdb_dataloader import IMDB


# Class for creating the neural network.
class Network(tnn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # according to pytorch document, batch_first should be true 
        # if the input and output tensors are provided as (batch, seq, feature). 
        self.lstm = torch.nn.LSTM(50, 100, batch_first=True)
        self.l1 = torch.nn.Linear(100, 64)
        self.l2 = torch.nn.Linear(64, 1)

    def forward(self, input, length):
        """
        DO NOT MODIFY FUNCTION SIGNATURE
        Create the forward pass through the network.
        """
        # src: https://pytorch.org/docs/stable/nn.html
        # h_n  hidden state 
        # c_n  hidden cell
        output, (h_n, c_n) = self.lstm(input)
        # LSTM(hidden dim = 100) -> Linear(64) -> ReLu-> Linear(1)
        x = tnn.functional.relu(self.l1(h_n))
        x = self.l2(x)

        return x.view(-1)


stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than','m','ll','the'})

punctuation = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''


class PreProcessing():
    def pre(x):
        """Called after tokenization"""
   
        # remove the stop_words 
        filtered_sentence = [w for w in x if not w in stop_words]
        filtered_sentence = []
        for w in x:
            if w not in stop_words:
                filtered_sentence.append(w)

        # remove string.punctuation
        x = [''.join(c for c in s if c not in punctuation) for s in filtered_sentence]
        # remove empty string
        x = [s for s in x if s]

      
        return x

    def post(batch, vocab):
        """Called after numericalization but prior to vectorization"""
        
        return batch
    # # this functin is used for data augmentation
    # def shuffle_tokenized(text):
    #     random.shuffle(text)
    #     newl=list(text)
    #     shuffled.append(newl)
    #     return text
    # # data augmentation
    # # given a tokenized text, return a list of text with 10 new sentence
    # # question how to implement this in main?
    # def data_augmentation(text): 
    #     augmented = []
    #     reps = []
    #     for i in range(11):
    #     #generate 11 new reviews
    #     shuffled = [text]
    #     shuffle_tokenized(shuffled[-1])
    #     for k in shuffled:
    #         '''create new review by joining the shuffled sentences'''
    #         s = ' '
    #         new_sentence = s.join(k)
    #         if new_rev not in augmented:
    #             augmented.append(new_sentence)
    #         else:
    #             reps.append(new_sentence)
    #     return reps

    text_field = data.Field(lower=True, include_lengths=True, batch_first=True, preprocessing=pre, postprocessing=post)


def lossFunc():
    """
    Define a loss function appropriate for the above networks that will
    add a sigmoid to the output and calculate the binary cross-entropy.
    """
    # src: https://pytorch.org/docs/stable/nn.html
    # This loss combines a Sigmoid layer and the BCELoss in one single class
    return torch.nn.BCEWithLogitsLoss()

def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = PreProcessing.text_field
    labelField = data.Field(sequential=False)

    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")
   


    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)

    net = Network().to(device)
    criterion =lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.001)  # Minimise the loss using the Adam algorithm.

    for epoch in range(10):
        running_loss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # PyTorch calculates gradients by accumulating contributions to them (useful for
            # RNNs).  Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            output = net(inputs, length)

            loss = criterion(output, labels)

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            running_loss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                running_loss = 0

    num_correct = 0

    # Save mode
    torch.save(net.state_dict(), "./model.pth")
    print("Saved model")


    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # Get predictions
            outputs = torch.sigmoid(net(inputs, length))
            predicted = torch.round(outputs)

            num_correct += torch.sum(labels == predicted).item()

    accuracy = 100 * num_correct / len(dev)

    print(f"Classification accuracy: {accuracy}")

if __name__ == '__main__':
    main()
