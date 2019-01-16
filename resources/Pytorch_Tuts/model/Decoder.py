import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Attn import Attn
######################################################################
# Decoder
# ~~~~~~~
#
# The decoder RNN generates the response sentence in a token-by-token
# fashion. It uses the encoder’s context vectors, and internal hidden
# states to generate the next word in the sequence. It continues
# generating words until it outputs an *EOS_token*, representing the end
# of the sentence. A common problem with a vanilla seq2seq decoder is that
# if we rely soley on the context vector to encode the entire input
# sequence’s meaning, it is likely that we will have information loss.
# This is especially the case when dealing with long input sequences,
# greatly limiting the capability of our decoder.
#
# To combat this, `Bahdanau et al. <https://arxiv.org/abs/1409.0473>`__
# created an “attention mechanism” that allows the decoder to pay
# attention to certain parts of the input sequence, rather than using the
# entire fixed context at every step.
#



######################################################################
# Now that we have defined our attention submodule, we can implement the
# actual decoder model. For the decoder, we will manually feed our batch
# one time step at a time. This means that our embedded word tensor and
# GRU output will both have shape *(1, batch_size, hidden_size)*.
#
# **Computation Graph:**
#
#    1) Get embedding of current input word.
#    2) Forward through unidirectional GRU.
#    3) Calculate attention weights from the current GRU output from (2).
#    4) Multiply attention weights to encoder outputs to get new "weighted sum" context vector.
#    5) Concatenate weighted context vector and GRU output using Luong eq. 5.
#    6) Predict next word using Luong eq. 6 (without softmax).
#    7) Return output and final hidden state.
#
# **Inputs:**
#
# -  ``input_step``: one time step (one word) of input sequence batch;
#    shape=\ *(1, batch_size)*
# -  ``last_hidden``: final hidden layer of GRU; shape=\ *(n_layers x
#    num_directions, batch_size, hidden_size)*
# -  ``encoder_outputs``: encoder model’s output; shape=\ *(max_length,
#    batch_size, hidden_size)*
#
# **Outputs:**
#
# -  ``output``: softmax normalized tensor giving probabilities of each
#    word being the correct next word in the decoded sequence;
#    shape=\ *(batch_size, voc.num_words)*
# -  ``hidden``: final hidden state of GRU; shape=\ *(n_layers x
#    num_directions, batch_size, hidden_size)*
#

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden
