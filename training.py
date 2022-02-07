import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from rouge_score import rouge_scorer
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import Encoder, get_model
from utils import *


def training(data_train, data_val, vocab, actual_emb_vocab):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    now = datetime.now()
    epoch_number = 7
    learning_rate = 0.0003
    model_path = './model.pth'
    model_name = '1.01'
    bleu_best = 0
    smoothing = SmoothingFunction()
    hidden_size = attention_dim = 300
    log_dir = './runs/' + model_name + '_' + now.strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(log_dir=log_dir)

    train_loader = DataLoader(data_train, batch_size=4, shuffle=True,
                              num_workers=12, collate_fn=collate_fn,
                              drop_last=True)
    val_loader = DataLoader(data_val, batch_size=4, shuffle=False,
                            num_workers=12, collate_fn=collate_fn,
                            drop_last=True)

    # if os.path.isfile('./vocab.pkl'):
    #     pass
    # else:

    encoder = Encoder().to(device)
    model = get_model(hidden_size, output_size=len(vocab),
                      attention_dim=attention_dim,
                      embeddings_vocab=actual_emb_vocab).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    output_dir = './log'
    output_file_name = now.strftime('%Y%m%d-%H%M%S') + '_' + model_name
    output_file = open(os.path.join(output_dir, output_file_name), 'w')

    print('Training started')
    output_file.write('TRAINING\n')

    for epoch in range(epoch_number):
        model.train()
        output_file.write(f'EPOCH {epoch + 1}\n')
        output_file.write('----------------------------------------------\n')
        now = datetime.now()
        loop = tqdm(train_loader)
        for i, data in enumerate(loop):
            src, label, all_labels_txt, img_names, cap_lens = data
            src, label = src.to(device), label.to(device)
            optimizer.zero_grad()
            enc_out = encoder(src)
            pred, alphas, sort_idx, decoder_lengths = model(
                enc_out, label, cap_lens, vocab)
            # since we sorted imgs in the model, captions should be sorted too
            img_names = [img_names[idx] for idx in list(sort_idx)]
            label = label[sort_idx]
            all_labels_txt = [all_labels_txt[idx] for idx in list(sort_idx)]
            pred_packed = pack_padded_sequence(pred, decoder_lengths,
                                               batch_first=True)
            label_packed = pack_padded_sequence(label, decoder_lengths,
                                                batch_first=True)
            loss = criterion(pred_packed.data, label_packed.data)
            # loss.requires_grad = True
            loss.backward()
            optimizer.step()
            pred = torch.argmax(pred, dim=2)

            # get some human language output
            labels_txt = index_to_word(vocab, label)
            pred_txt = index_to_word(vocab, pred)
            # get bleu score
            bleu_curr = corpus_bleu(
                labels_txt, pred_txt,
                smoothing_function=smoothing.method1)
            # check whether we got the best bleu while training
            bleu_best = max(bleu_best, bleu_curr)
            rougescorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'],
                                                   use_stemmer=True)
            rouge = rougescorer.score(str(labels_txt), str(pred_txt))
            output_to_file(pred_txt, output_file, img_names,
                           all_labels_txt, rouge)

            # write to tensorboard
            writer.add_scalar(
                'Training loss', loss.item(),
                epoch * len(train_loader) + i)
            writer.add_scalar(
                'Training BLEU', bleu_curr,
                epoch * len(train_loader) + i)
            loop.set_description(f'Epoch [{epoch + 1}/{epoch_number}]')
            loop.set_postfix(loss=loss.item())
            writer.flush()

        # validation
        with torch.no_grad():
            model.eval()
            output_file.write('VALIDATION\n')
            output_file.write(
                '----------------------------------------------\n')
            loop = tqdm(val_loader)
            for i, data in enumerate(loop):
                src, label, all_labels_txt, img_names, cap_lens = data
                src, label = src.to(device), label.to(device)
                enc_out = encoder(src)
                pred, alphas, sort_idx, decoder_lengths = model(
                    enc_out, label, cap_lens, vocab)
                # since we sorted imgs in the model,
                # captions should be sorted too
                img_names = [img_names[idx] for idx in list(sort_idx)]
                label = label[sort_idx]
                all_labels_txt = [all_labels_txt[idx] for idx in
                                  list(sort_idx)]
                pred_packed = pack_padded_sequence(pred, decoder_lengths,
                                                   batch_first=True)
                label_packed = pack_padded_sequence(label, decoder_lengths,
                                                    batch_first=True)
                loss = criterion(pred_packed.data, label_packed.data)
                pred = torch.argmax(pred, dim=2)

                # get some human language output
                labels_txt = index_to_word(vocab, label)
                pred_txt = index_to_word(vocab, pred)
                # get bleu score
                bleu_curr = corpus_bleu(
                    labels_txt, pred_txt,
                    smoothing_function=smoothing.method1)
                # check whether we got the best bleu while training
                bleu_best = max(bleu_best, bleu_curr)
                rouge = rougescorer.score(str(labels_txt), str(pred_txt))
                output_to_file(pred_txt, output_file, img_names,
                               all_labels_txt, rouge)

                # write to tensorboard
                writer.add_scalar(
                    'Validation loss', loss.item(),
                    epoch * len(val_loader) + i)
                writer.add_scalar(
                    'Validation BLEU', bleu_curr,
                    epoch * len(val_loader) + i)
                loop.set_description(
                    f'Validation Epoch [{epoch + 1}/{epoch_number}]')
                loop.set_postfix(loss=loss.item())

    output_file.close()
    writer.close()
    print('Training is finished, best BLEU is ', bleu_best)
    torch.save(model.state_dict(), model_path)
