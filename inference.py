import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import get_model
from utils import collate_fn, index_to_word


def load_model_inference(device, vocab, actual_emb_vocab):
    model_path = './best_model/model.pth'
    # choose embedding vector length, leave as 300 if using pretrained ones
    hidden_size = attention_size = 300
    model = get_model(hidden_size, len(vocab),
                      attention_size, actual_emb_vocab, inference=True)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    return model


def inference(data_test, vocab, actual_emb_vocab, is_cpu: bool = False):
    if is_cpu:
        device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model_inference(device, vocab, actual_emb_vocab)
    test_loader = DataLoader(data_test, batch_size=1, shuffle=False,
                             num_workers=12, collate_fn=collate_fn,
                             drop_last=True)
    start_time = datetime.now()
    output_file_name = start_time.strftime('%Y%m%d-%H%M%S') + '_INFERENCE'
    output_dir = './log'
    output_file = open(os.path.join(output_dir, output_file_name), 'w')
    model.eval()

    with torch.no_grad():
        loop = tqdm(test_loader)
        for i, data in enumerate(loop):
            src, label, all_labels_txt, img_name, cap_lens = data
            src, label = src.to(device), label.to(device)
            # img_name = data[3]
            output = model(src, label, cap_lens, vocab)
            output_txt = index_to_word(vocab, output)
            output_file.write(str(img_name) + str(output_txt) + '\n')
            # if i == 1000:
            #     break
        end_time = datetime.now()
        print('Inference for 1000 images took ', end_time-start_time)
    output_file.close()


if __name__ == '__main__':
    inference()
