import os
import random

from PIL import Image
from torch.utils.data import Dataset, random_split
from torchtext.data.utils import get_tokenizer
from torchvision import transforms


class Flickr_dataset(Dataset):
    def __init__(self, img_dir, ann_dir, ann_train_file,
                 transform, vocab, all_anns):
        self.imgs = sorted([item for item in os.listdir(img_dir) if
                            os.path.isfile(os.path.join(img_dir, item))])
        self.anns = {}
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.ann_train_file = ann_train_file
        self.transform = transform
        self.tokenizer = get_tokenizer('basic_english')
        self.captions_per_image = 5
        self.vocab = vocab
        self.all_anns = all_anns

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = Image.open(img_path)
        img = self.transform(img)
        img_name = self.imgs[idx]
        batch_captions, batch_captions_txt = list(), list()
        batch_captions_txt = self.all_anns[img_name]
        for i in range(self.captions_per_image):
            batch_captions.append(self.vocab(self.tokenizer(
                self.all_anns[img_name][i])))
        # choose the random caption to use in the model
        caption = batch_captions[random.randint(0, 4)]
        return img, caption, batch_captions_txt, img_name, len(caption)

    def yield_tokens(self, data):
        for item in data:
            yield self.tokenizer(str(item))


def get_data(vocab, all_anns):
    transform = transforms.Compose(
        [transforms.Resize((640, 640)), transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225])])

    data = Flickr_dataset(img_dir='./flickr',
                          ann_dir='./annotations_flickr/',
                          ann_train_file='captions.txt',
                          transform=transform,
                          vocab=vocab,
                          all_anns=all_anns)

    train_len = int(0.7 * len(data))
    val_len = int(0.15 * len(data))
    test_len = int(len(data) - (train_len + val_len))

    data_train, data_val, data_test = random_split(
        data, [train_len, val_len, test_len])

    return data_train, data_val, data_test
