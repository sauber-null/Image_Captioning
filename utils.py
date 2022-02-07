import os
import pickle
import re
from collections import defaultdict

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


def create_vocab(ann_train_file, ann_dir):
    all_anns = defaultdict(list)
    assert ann_train_file in os.listdir(ann_dir), \
        'seems that there is no expected data file'
    with open(os.path.join(ann_dir,
                           ann_train_file), 'r') as file:
        # skip the first line in file with column names
        next(file)
        captions = file.read().splitlines()
        for line in captions:
            img_name = line.split(',', 1)[0]
            ann = line.split(',', 1)[1]
            ann = re.sub('[^A-Za-z]+', ' ', ann).lower()
            ann = '<start> ' + ann + ' <end>'
            all_anns[img_name].append(ann)
    encode_to_pickle(all_anns, 'all_anotations.txt')

    vocab = build_vocab_from_iterator(
        yield_tokens(all_anns.values()),
        specials=['<pad>', '<start>', '<end>', '<unk>'],
        min_freq=1)
    vocab.set_default_index(vocab['<unk>'])
    encode_to_pickle(vocab, 'data_vocab.txt')


def create_embeddings_vocab(vocab, embeddings_dict):
    # now create embeddings vocab
    actual_emb_vocab = defaultdict(torch.tensor)
    for word in vocab.vocab.itos_:
        if word in embeddings_dict.keys():
            actual_emb_vocab[word] = embeddings_dict[word]
    encode_to_pickle(actual_emb_vocab, 'actual_pretrained_embeddings.pkl')


def is_float(n):
    try:
        float(n)
        return True
    except ValueError:
        return False


def create_encoded_embeddings(embeddings_path):
    embeddings_dict = {}
    with open(embeddings_path, 'r', encoding='utf-8') as file:
        for line in file:
            word = line.split()[0]
            vector = torch.tensor([float(el) for el in line.split()[1:]
                                  if is_float(el)])
            embeddings_dict[word] = vector
    output_file = open('big_embeddings_dict.txt', 'wb')
    # serializing our dict
    pickle.dump(embeddings_dict, output_file)
    output_file.close()


def encode_to_pickle(in_data, out_file_name):
    output_file = open(out_file_name, 'wb')
    # serializing
    pickle.dump(in_data, output_file)
    output_file.close()


def decode_from_pickle(in_data):
    with open(in_data, 'rb') as input_file:
        data = input_file.read()
    out_data = pickle.loads(data)
    return out_data


def yield_tokens(data):
    tokenizer = get_tokenizer('basic_english')
    for item in data:
        yield tokenizer(str(item))


def collate_fn(batch):
    img, label, all_labels, img_name, batch_len = zip(*batch)
    label = list(label)
    all_labels = list(all_labels)
    for i in range(len(label)):
        label[i] += [0] * (max(batch_len) - len(label[i]))
    img = torch.stack((img))
    return img, torch.tensor(label), all_labels, img_name, batch_len


def index_to_word(vocab, indices):
    words = list()
    # in case of inference
    if type(indices) == list:
        for i in range(len(indices)):
            idx = indices[i]
            words.append(vocab.vocab.itos_[idx])
        return words
    for i in range(indices.shape[0]):
        elem = list()
        for j in range(indices.shape[1]):
            idx = indices[i, j]
            elem.append(vocab.vocab.itos_[idx])
        words.append(elem)
    return words


def output_to_file(output, file, img_names, all_labels, rouge):
    assert not file.closed
    for i, r in enumerate(output):
        row = str(img_names[i]) + ': '
        for el in r:
            if el in ['<pad>']:
                break
            row += str(el) + ' '
        file.write(row + '\n')
        file.write(str(rouge) + '\n')
    file.write('True labels:\n')
    for i, _ in enumerate(output):
        file.write(str(img_names[i]) + ': ' +
                   str([str(el) for el in all_labels[i]]) + '\n')
    file.write('----------------------------------------------\n')
