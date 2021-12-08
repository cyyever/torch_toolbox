import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    def __init__(
        self,
        num_embeddings,
        num_classes,
        padding_idx=None,
        n_layers=1,
        bidirectional=True,
        embedding_dim=250,
        hidden_dim=256,
        dropout=0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        # pack sequence
        # lengths need to be on CPU!
        packed_embedded = embedded
        # packed_embedded = nn.utils.rnn.pack_padded_sequence(
        #     embedded, text_lengths.to("cpu")
        # )
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # unpack sequence
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]
        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # hidden = [batch size, hid dim * num directions]
        return self.fc(hidden)
