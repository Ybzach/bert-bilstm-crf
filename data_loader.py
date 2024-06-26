import torch
import numpy as np
import math
from torch.utils.data import Dataset


class NerDataset(Dataset):
    def __init__(self, data, args, tokenizer):
        self.data = data
        self.args = args
        self.tokenizer = tokenizer
        self.label2id = args.label2id
        self.max_seq_len = args.max_seq_len

    def tokenize_and_align_labels(self, item):
        tokenized_input = self.tokenizer(
            item['text'],
            truncation=False,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        word_ids = tokenized_input.word_ids() 
        previous_word_idx = None
        label_ids = []
        for i, token_id in enumerate(tokenized_input['input_ids']):
            if word_ids[i] is None:
                label_ids.append(0)
            elif word_ids[i] != previous_word_idx:
                label_ids.append(self.label2id[item['labels'][word_ids[i]]])
            else: 
                label_ids.append(0)
        previous_word_idx = word_ids[i] 

        tokenized_input['labels'] = label_ids
        return tokenized_input

    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        tokenized_input = self.tokenize_and_align_labels(self.data[item])
        input_ids = tokenized_input['input_ids']
        attention_mask = tokenized_input['attention_mask']
        labels = tokenized_input['labels']

        number_of_tokens = len(tokenized_input['input_ids'])
        if number_of_tokens > (self.max_seq_len - 2):
            # crop = int(math.ceil((number_of_tokens - (self.max_seq_len - 2)) / 2))
            # input_ids = [101] + input_ids[crop:number_of_tokens - crop] + [102]
            # attention_mask = [1] + attention_mask[crop:number_of_tokens - crop] + [1]
            # labels = [0] + labels[crop:number_of_tokens - crop] + [0]
            input_ids = [101] + input_ids[-1*(self.max_seq_len - 1):]
            attention_mask = [1] + attention_mask[-1*(self.max_seq_len - 1):]
            labels = [0] + labels[-1*(self.max_seq_len - 1):]


        input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
        attention_mask = attention_mask + [0] * (self.max_seq_len - len(attention_mask))
        labels = labels  + [0] * (self.max_seq_len - len(labels))

        input_ids = torch.tensor(np.array(input_ids))
        attention_mask = torch.tensor(np.array(attention_mask))
        labels = torch.tensor(np.array(labels))

        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return data