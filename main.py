import argparse
import random

import numpy as np
import torch

from flickr_dataset import get_data
from inference import *
from training import training
from utils import *


def main():
    manual_seed = 42
    np.random.seed(manual_seed)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    train_file = 'captions.txt'
    train_dir = './annotations_flickr/'
    raw_embeddings_path = 'glove.840B.300d.txt'
    encoded_raw_embeddings = 'big_embeddings_dict.txt'
    encoded_actual_embeddings = 'actual_pretrained_embeddings.pkl'
    encoded_data_vocab = 'data_vocab.txt'
    encoded_annotations = 'all_anotations.txt'

    assert os.path.isfile(raw_embeddings_path), \
        'for now you should have glove file in the project folder'

    if not os.path.isfile(encoded_actual_embeddings):
        # creates encoded_raw_embeddings file
        create_encoded_embeddings(raw_embeddings_path)
        # creates encoded_data_vocab and encoded_annotations files
        create_vocab(ann_train_file=train_file, ann_dir=train_dir)
        # creates encoded_actual_embeddings file
        create_embeddings_vocab(decode_from_pickle(encoded_data_vocab),
                                decode_from_pickle(encoded_raw_embeddings))
    # decode all needed files
    actual_embeddings_dict = decode_from_pickle(encoded_actual_embeddings)
    vocab = decode_from_pickle(encoded_data_vocab)
    all_anns = decode_from_pickle(encoded_annotations)

    data_train, data_val, data_test = get_data(vocab, all_anns)

    arg_parser = argparse.ArgumentParser(
        description='Run mode: train/val or inference')
    arg_parser.add_argument(
        '--mode', type=str, help='Input \'train\' or \'inference\'')
    args = arg_parser.parse_args()
    
    if args.mode == 'train':
        training(data_train, data_val, vocab, actual_embeddings_dict)
    elif args.mode == 'inference':
        inference(data_test, vocab, actual_embeddings_dict, is_cpu=False)


if __name__ == '__main__':
    main()
