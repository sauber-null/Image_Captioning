import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Attention(nn.Module):
    def __init__(self, encoder_size, hidden_size, attention_size):
        super().__init__()
        self.encoder_attn = nn.Linear(encoder_size, attention_size)
        self.decoder_attn = nn.Linear(hidden_size, attention_size)
        self.full_attn = nn.Linear(attention_size, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, enc_out, dec_hidden):
        # (batch_size, num_pixels, attention_dim)
        attn1 = self.encoder_attn(enc_out)
        # (batch_size, attention_dim)
        attn2 = self.decoder_attn(dec_hidden)
        # (batch_size, num_pixels)
        attn = self.full_attn(self.relu(attn1 + attn2.unsqueeze(1))).squeeze(2)
        # (batch_size, num_pixels)
        alpha = self.softmax(attn)
        # (batch_size, encoder_dim)
        attn_weighted_encoding = (enc_out * alpha.unsqueeze(2)).sum(dim=1)
        return attn_weighted_encoding, alpha


class Pretrained_embeddings:
    def __init__(self):
        self.embeddings_dict = {}

    def unpack(self):
        def is_float(n):
            try:
                float(n)
                return True
            except ValueError:
                return False

        with open('glove.840B.300d.txt', 'r', encoding='utf-8') as file:
            for line in file:
                word = line.split()[0]
                vector = torch.tensor([float(el) for el in line.split()[1:] if is_float(el)])
                self.embeddings_dict[word] = vector
        return self.embeddings_dict


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torchvision.models.resnet101(pretrained=True)
        self.encoder = nn.Sequential(
            *list(self.encoder.children())[:-2],
            nn.AdaptiveAvgPool2d((14, 14))
        )

    def forward(self, x):
        enc_out = self.encoder(x)
        enc_out = enc_out.permute(0, 2, 3, 1)
        return enc_out


class Model(nn.Module):
    def __init__(self, hidden_size, output_size, attention_dim,
                 embeddings_vocab, inference, encoder_size=2048, dropout=0.7):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.encoder_size = encoder_size
        self.attention_dim = attention_dim
        self.embeddings_vocab = embeddings_vocab
        self.inference = inference
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.huge_embedding = nn.Embedding(
            output_size, 300)
        self.attention = Attention(self.encoder_size, hidden_size,
                                   self.attention_dim)
        self.lstm = nn.LSTMCell(self.encoder_size + hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.init_hidden = nn.Linear(self.encoder_size, hidden_size)
        self.init_cell = nn.Linear(self.encoder_size, hidden_size)
        # linear layer to create a sigmoid-activated gate
        self.f_beta = nn.Linear(hidden_size, self.encoder_size)
        # linear fully-connected layer to find scores over vocabulary
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)
        self.beam_size = 3

    def forward(self, x, captions, cap_lens, vocab):
        batch_size = len(x)
        cap_lens, sort_idx = torch.tensor(cap_lens).sort(dim=0,
                                                         descending=True)
        enc_out = x[sort_idx]
        enc_out = enc_out.view(batch_size, -1, self.encoder_size)
        # hidden_state = self.init_hidden(enc_out.mean(dim=1))
        # cell_state = self.init_cell(enc_out.mean(dim=1))
        num_pixels = enc_out.size(1)

        # create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(
            batch_size, max(cap_lens), self.output_size).to(device)
        alphas = torch.zeros(
            batch_size, max(cap_lens), num_pixels).to(device)
        decoder_lengths = (cap_lens - 1).tolist()

        # do not apply below lines in case of inference
        if not self.inference:
            captions = captions[sort_idx]
            # uncomment line below to use non-pretrained embeddings
            # embedding = self.embedding(captions)
            embedding = self.huge_embedding(captions)
            hidden_state = self.init_hidden(enc_out.mean(dim=1))
            cell_state = self.init_cell(enc_out.mean(dim=1))

            for i in range(max(cap_lens)):
                batch = sum([k > i for k in cap_lens])
                attn_weighted_encoding, alpha = self.attention(
                    enc_out[:batch], hidden_state[:batch])
                # gating scalar, (batch_size_t, encoder_dim)
                gate = self.sigmoid(self.f_beta(hidden_state[:batch]))
                attn_weighted_encoding = gate * attn_weighted_encoding

                hidden_state, cell_state = self.lstm(
                    torch.cat([embedding[:batch, i, :],
                            attn_weighted_encoding], dim=1),
                    (hidden_state[:batch], cell_state[:batch]))
                preds = self.fc(self.dropout(hidden_state))
                predictions[:batch, i, :] = preds
                alphas[:batch, i, :] = alpha
            return predictions, alphas, sort_idx, decoder_lengths

        # other loop for the inference mode
        if self.inference:
            # enc_out = enc_out.expand(self.beam_size, enc_out.size(1), enc_out.size(2))
            hidden_state = self.init_hidden(enc_out.mean(dim=1))
            cell_state = self.init_cell(enc_out.mean(dim=1))
            # batch = sum([k > i for k in cap_lens])
            attn_weighted_encoding, alpha = self.attention(
                enc_out, hidden_state)
            # gating scalar, (batch_size_t, encoder_dim)
            gate = self.sigmoid(self.f_beta(hidden_state))
            attn_weighted_encoding = gate * attn_weighted_encoding
            step = 1
            # k_words = torch.LongTensor(
            #     [[vocab['<start>']]] * self.beam_size).to(device)
            k_words = torch.tensor(0).unsqueeze(0).to(device)
            k_scores = torch.zeros(self.beam_size, 1).to(device)
            sequences = k_words
            complete_sequences = list()
            complete_sequences_scores = list()
            sentence = list()

            while True:
                embedding = self.huge_embedding(k_words)
                # alpha = alpha.view(-1, enc_out.size(1), enc_out.size(1))
                hidden_state, cell_state = self.lstm(
                    torch.cat([embedding, attn_weighted_encoding], dim=1),
                    (hidden_state, cell_state))
                scores = self.fc(self.dropout(hidden_state))
                scores = F.log_softmax(scores, dim=1)
                top_idx = scores[0].topk(1)[1]
                sentence.append(top_idx.item())
                k_words = top_idx
                if len(sentence) > 40 or top_idx == vocab['<end>']:
                    break
            return sentence

            #     scores = k_scores.expand_as(scores) + scores

            #     if step == 1:
            #         k_scores, k_words = scores[0].topk(self.beam_size, 0, True, True)
            #     else:
            #         k_scores, k_words = scores.view(-1).topk(self.beam_size, 0, True, True)
            #     prev_word_idx = k_words / len(vocab)
            #     next_word_idx = k_words % len(vocab)
            #     sequences = torch.cat([sequences[prev_word_idx.long()], next_word_idx.unsqueeze(1)], dim=1)
            #     incomplete_idx = [idx for idx, next_word in enumerate(next_word_idx) if
            #                         next_word != vocab['<end>']]
            #     complete_idx = list(set(range(len(next_word_idx))) - set(incomplete_idx))

            #     # Set aside complete sequences
            #     if len(complete_idx) > 0:
            #         complete_sequences.extend(sequences[complete_idx].tolist())
            #         complete_sequences_scores.extend(k_scores[complete_idx])
            #     self.beam_size -= len(complete_idx)
            #     # proceed with incomplete sequences
            #     if self.beam_size == 0:
            #         break
            #     sequences = sequences[incomplete_idx]
            #     hidden_state = hidden_state[prev_word_idx[torch.tensor(incomplete_idx, dtype=torch.long)].long()]
            #     cell_state = cell_state[prev_word_idx[torch.tensor(incomplete_idx, dtype=torch.long)].long()]
            #     enc_out = enc_out[prev_word_idx[torch.tensor(incomplete_idx, dtype=torch.long)].long()]
            #     k_scores = k_scores[incomplete_idx].unsqueeze(1)
            #     k_words = next_word_idx[incomplete_idx].unsqueeze(1)
            #     # break if we have gone too far
            #     if step > 50:
            #         break
            #     step += 1

            # idx = complete_sequences_scores.index(max(complete_sequences_scores))
            # sequence = complete_sequences[idx]
            # return sequence


class SimpleModel(nn.Module):
    def __init__(self, output_size, hidden_size, encoder_size=2048):
        super().__init__()
        self.encoder = Encoder()
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTMCell(encoder_size + hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, captions, cap_lens):
        batch_size = len(x)
        enc_out = self.encoder(x)
        cap_lens, sort_idx = torch.tensor(cap_lens).sort(dim=0,
                                                         descending=True)
        caps_embed = self.embedding(captions)
        # create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(
            batch_size, max(cap_lens), self.output_size).to(device)
        for i in range(max(cap_lens)):
            batch = sum([k > i for k in cap_lens])
            pre_lstm = torch.cat((caps_embed[:batch, i, :], enc_out), dim=1)
            hidden_state, _ = self.lstm(pre_lstm)
            preds = self.fc(hidden_state)
            predictions[:batch, i, :] = preds
        predictions = torch.argmax(predictions, dim=2)
        return predictions, sort_idx


def get_model(hidden_size, output_size, attention_dim, embeddings_vocab,
              inference=False):
    model = Model(hidden_size, output_size, attention_dim,
                  embeddings_vocab, inference).to(device)
    return model
