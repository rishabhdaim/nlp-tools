import random

import torch
from torch import nn, optim

from decoder import Decoder
from encoder import Encoder
from lang import process_data, tensor_from_sentence, tensors_from_pair

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, MAX_LENGTH=MAX_LENGTH):
        super().__init__()

        # initialize the encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_forcing_ratio=0.5):

        input_length = source.size(0)  # get the input length (number of words in sentence)
        batch_size = target.shape[1]
        target_length = target.shape[0]
        vocab_size = self.decoder.output_dim

        # initialize a variable to hold the predicted outputs
        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)

        # encode every word in a sentence
        for i in range(input_length):
            encoder_output, encoder_hidden = self.encoder(source[i])

        # use the encoder’s hidden layer as the decoder hidden
        decoder_hidden = encoder_hidden.to(device)

        # add a token before the first predicted word
        decoder_input = torch.tensor([SOS_token], device=device)  # SOS

        # topk is used to get the top K value over a list predict the output word from the current target word. If we
        # enable the teaching force,  then the #next decoder input is the next word, else, use the decoder output
        # highest value.

        for t in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            topv, topi = decoder_output.topk(1)
            input = (target[t] if teacher_force else topi)
            if teacher_force == False and input.item() == EOS_token:
                break

        return outputs


teacher_forcing_ratio = 0.5


def clac_model(model, input_tensor, target_tensor, model_optimizer, criterion):
    model_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    loss = 0
    epoch_loss = 0
    # print(input_tensor.shape)

    output = model(input_tensor, target_tensor)

    num_iter = output.size(0)
    # print(num_iter)

    # calculate the loss from a predicted sentence with the expected result
    for ot in range(num_iter):
        loss += criterion(output[ot], target_tensor[ot])

    loss.backward()
    model_optimizer.step()
    epoch_loss = loss.item() / num_iter

    return epoch_loss


def train_model(model, source, target, pairs, num_iteration):
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    total_loss_iterations = 0

    training_pairs = [tensors_from_pair(source, target, pair) for pair in pairs]

    print('Size of training pair is %d' % (len(training_pairs)))

    for ite in range(1, num_iteration + 1):
        for training_pair in training_pairs:
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            # print('Input %s && output %s', input_tensor, target_tensor)

            loss = clac_model(model, input_tensor, target_tensor, optimizer, criterion)

            total_loss_iterations += loss

        if ite % 2 == 0:
            average_loss = total_loss_iterations / 2
            total_loss_iterations = 0
            print('%d %.4f' % (ite, average_loss))

    return model


def evaluate(model, input_lang, output_lang, sentences, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_lang, sentences[0])
        output_tensor = tensor_from_sentence(output_lang, sentences[1])

        decoded_words = []

        output = model(input_tensor, output_tensor)
        # print(output_tensor)

        for ot in range(output.size(0)):
            topv, topi = output[ot].topk(1)
            # print(topi)

            if topi[0].item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi[0].item()])
    return decoded_words


def evaluate_randomly(model, source, target, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('source {}'.format(pair[0]))
        print('target {}'.format(pair[1]))
        output_words = evaluate(model, source, target, pair)
        output_sentence = ' '.join(output_words)
        print('predicted {}'.format(output_sentence))


if __name__ == '__main__':
    lang1 = 'eng'
    lang2 = 'ind'
    source, target, pairs = process_data('resources', lang1, lang2)

    randomize = random.choice(pairs)
    print('random sentence {}'.format(randomize))

    # # print number of words
    input_size = source.n_words
    output_size = target.n_words
    print('Input : {} Output : {}'.format(input_size, output_size))

    embed_size = 256
    hidden_size = 512
    num_layers = 1
    num_iteration = 200
    #
    # # create encoder-decoder model
    encoder = Encoder(input_size, hidden_size, embed_size, num_layers)
    decoder = Decoder(output_size, hidden_size, embed_size, num_layers)

    model = Seq2Seq(encoder, decoder, device).to(device)
    #
    # # print model
    # print(encoder)
    # print(decoder)

    model = train_model(model, source, target, pairs, num_iteration)
    # model.load_state_dict(torch.load('mytraining.pt', map_location=device))
    evaluate_randomly(model, source, target, pairs)
