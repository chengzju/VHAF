import json
import random
import numpy as np
import torch

class MyJSONFileDataLoader():
    def __init__(self, args, dataset=None, tokenizer=None, max_len=None, shuffle=True):
        self.args = args
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.train_path = 'dataset/' + dataset + '/train.json'
        self.val_path = 'dataset/' + dataset + '/val.json'
        self.test_path = 'dataset/' + dataset +'/test.json'
        self.label_separate_path = 'dataset/separate.json'
        with open(self.train_path, 'r') as f:
            self.train_JSON_dict = json.load(f)
        with open(self.val_path, 'r') as f:
            self.val_JSON_dict = json.load(f)
        with open(self.test_path, 'r') as f:
            self.test_JSON_dict = json.load(f)
        sta_max_token_len = max(self.find_max_token_len(file_path=self.train_path),
                                  self.find_max_token_len(file_path=self.val_path),
                                  self.find_max_token_len(file_path=self.test_path))
        print('sta_max_len', str(sta_max_token_len))
        if max_len is None:
            self.max_token_len = sta_max_token_len
        else:
            self.max_token_len = max_len

    def find_max_token_len(self, file_path=None):
        with open(file_path, 'r') as f:
            JSON_dict = json.load(f)
        JSON_Value_list = list(JSON_dict.values())
        max_len = 0
        for i in range(len(JSON_Value_list)):
            for j in range(len(JSON_Value_list[i])):
                words = JSON_Value_list[i][j][0]
                text = ' '.join(words)
                tokens = self.tokenizer.tokenize(text)
                max_len = max(max_len, len(tokens))
        return max_len

    def tokenize_text_and_mask(self, words, max_len=None):
        if max_len is None:
            length = self.max_token_len
        else:
            length = max_len
        text = ' '.join(words)
        input_toks = self.tokenizer.tokenize(text)
        input_toks = input_toks[:length]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_toks)
        input_type_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * (length - len(input_ids))
        input_type_ids += [self.tokenizer.pad_token_type_id] * (length - len(input_type_ids))
        input_mask += [0] * (length - len(input_mask))
        return input_ids, input_type_ids, input_mask

    def tokenize_aspect_and_mask(self, aspects):
        length = self.args.max_class_words
        aspect_ids = []
        aspect_type_ids = []
        aspect_mask = []
        for a in aspects:
            words = a.split('_')
            text = ' '.join(words)
            toks = self.tokenizer.tokenize(text)[:length]
            one_ids = self.tokenizer.convert_tokens_to_ids(toks)
            one_type_ids = [0] * len(one_ids)
            one_mask = [1] * len(one_ids)
            one_ids += [self.tokenizer.pad_token_id] * (length - len(one_ids))
            one_type_ids += [self.tokenizer.pad_token_type_id] * (length - len(one_type_ids))
            one_mask += [0] * (length - len(one_mask))
            aspect_ids.append(one_ids)
            aspect_type_ids.append(one_type_ids)
            aspect_mask.append(one_mask)
        return aspect_ids, aspect_type_ids, aspect_mask

    def make_label(self, classes, aspects):
        label = [0] * len(classes)
        for i in range(len(classes)):
            if classes[i] in aspects:
                label[i] = 1
        return label

    def next_one(self, N, K, Q, JSON_dict):
        support_set = {'text_ids': [], 'text_type_ids': [], 'text_masks': [], 'meta_label': []}
        query_set = {'text_ids': [], 'text_type_ids': [], 'text_masks': [], 'meta_label': []}
        target_classes = random.sample(JSON_dict.keys(), N)
        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(list(range(len(JSON_dict[class_name]))), size=K + Q, replace=False)
            text_ids_pool, text_type_ids_pool, text_mask_pool = [], [], []
            label_meta_pool = []
            for j in range(len(indices)):
                index = indices[j]
                words = JSON_dict[class_name][index][0]
                aspects = JSON_dict[class_name][index][1]
                text_ids, text_type_ids, text_mask = self.tokenize_text_and_mask(words)
                label_meta = self.make_label(classes=target_classes, aspects=aspects)
                text_ids_pool.append(text_ids)
                text_type_ids_pool.append(text_type_ids)
                text_mask_pool.append(text_mask)
                label_meta_pool.append(label_meta)
            support_ids, query_ids, _ = np.split(text_ids_pool, [K, K + Q])
            support_type_ids, query_type_ids, _ = np.split(text_type_ids_pool, [K, K + Q])
            support_masks, query_masks, _ = np.split(text_mask_pool, [K, K + Q])
            support_meta_labels, query_meta_labels, _ = np.split(label_meta_pool, [K, K + Q])
            support_set['text_ids'].append(support_ids)
            support_set['text_type_ids'].append(support_type_ids)
            support_set['text_masks'].append(support_masks)
            support_set['meta_label'].append(support_meta_labels)
            query_set['text_ids'].append(query_ids)
            query_set['text_type_ids'].append(query_type_ids)
            query_set['text_masks'].append(query_masks)
            query_set['meta_label'].append(query_meta_labels)
        aspect_ids, aspect_type_ids, aspect_mask = self.tokenize_aspect_and_mask(target_classes)
        support_set['text_ids'] = np.stack(support_set['text_ids'], axis=0)  # N, K, max_len
        support_set['text_type_ids'] = np.stack(support_set['text_type_ids'], axis=0)  # N, K, max_len
        support_set['text_masks'] = np.stack(support_set['text_masks'], axis=0)  # N, K, max_len
        support_set['meta_label'] = np.stack(support_set['meta_label'], axis=0)  # N, K, N
        support_set['aspect_ids'] = aspect_ids
        support_set['aspect_type_ids'] = aspect_type_ids
        support_set['aspect_mask'] = aspect_mask
        query_set['text_ids'] = np.concatenate(query_set['text_ids'], axis=0)  # N*Q, max_len
        query_set['text_type_ids'] = np.concatenate(query_set['text_type_ids'], axis=0)  # N*Q, max_len
        query_set['text_masks'] = np.concatenate(query_set['text_masks'], axis=0)  # N*Q, max_len
        query_set['meta_label'] = np.concatenate(query_set['meta_label'], axis=0)  # N*Q, N
        if self.shuffle:
            perm = np.random.permutation(N * Q)
            query_set['text_ids'] = query_set['text_ids'][perm]
            query_set['text_type_ids'] = query_set['text_type_ids'][perm]
            query_set['text_masks'] = query_set['text_masks'][perm]
            query_set['meta_label'] = query_set['meta_label'][perm]
        query_set['text_ids'] = np.reshape(query_set['text_ids'], (N, Q, -1))
        query_set['text_type_ids'] = np.reshape(query_set['text_type_ids'], (N, Q, -1))
        query_set['text_masks'] = np.reshape(query_set['text_masks'], (N, Q, -1))
        query_set['meta_label'] = np.reshape(query_set['meta_label'], (N, Q, -1))
        return support_set, query_set

    def next_batch(self, B, N, K, Q, phrase='train'):
        JSON_dict = {}
        if phrase == 'train':
            JSON_dict = self.train_JSON_dict
        if phrase == 'eval':
            JSON_dict = self.val_JSON_dict
        if phrase == 'test':
            JSON_dict = self.test_JSON_dict
        batch_support_set = {'text_ids': [], 'text_type_ids': [], 'text_masks': [],
                             'aspect_ids': [], 'aspect_type_ids': [], 'aspect_mask': [], 'meta_label': []}
        batch_query_set = {'text_ids': [], 'text_type_ids': [], 'text_masks': [], 'meta_label': []}
        for one_batch in range(B):
            current_support_set, current_query_set = self.next_one(N=N, K=K, Q=Q, JSON_dict=JSON_dict)
            batch_support_set['text_ids'].append(current_support_set['text_ids'])
            batch_support_set['text_type_ids'].append(current_support_set['text_type_ids'])
            batch_support_set['text_masks'].append(current_support_set['text_masks'])
            batch_support_set['meta_label'].append(current_support_set['meta_label'])
            batch_support_set['aspect_ids'].append(current_support_set['aspect_ids'])
            batch_support_set['aspect_type_ids'].append(current_support_set['aspect_type_ids'])
            batch_support_set['aspect_mask'].append(current_support_set['aspect_mask'])
            batch_query_set['text_ids'].append(current_query_set['text_ids'])
            batch_query_set['text_type_ids'].append(current_query_set['text_type_ids'])
            batch_query_set['text_masks'].append(current_query_set['text_masks'])
            batch_query_set['meta_label'].append((current_query_set['meta_label']))
        batch_support_set['text_ids'] = np.stack(batch_support_set['text_ids'], axis=0)  # B, N, K, max len
        batch_support_set['text_type_ids'] = np.stack(batch_support_set['text_type_ids'], axis=0)  # B, N, K, max len
        batch_support_set['text_masks'] = np.stack(batch_support_set['text_masks'], axis=0)  # B, N, K, max len
        batch_support_set['meta_label'] = np.stack(batch_support_set['meta_label'], axis=0) # B, N, K, N
        batch_support_set['aspect_ids'] = np.stack(batch_support_set['aspect_ids'], axis=0)
        batch_support_set['aspect_type_ids'] = np.stack(batch_support_set['aspect_type_ids'], axis=0)
        batch_support_set['aspect_mask'] = np.stack(batch_support_set['aspect_mask'], axis=0)
        batch_query_set['text_ids'] = np.stack(batch_query_set['text_ids'], axis=0)  # B, N, K, max len
        batch_query_set['text_type_ids'] = np.stack(batch_query_set['text_type_ids'], axis=0)  # B, N, K, max len
        batch_query_set['text_masks'] = np.stack(batch_query_set['text_masks'],axis=0)  # B, N, K, max len
        batch_query_set['meta_label'] = np.stack(batch_query_set['meta_label'], axis=0)  # B, N, Q, N
        batch_support_set['text_ids'] = torch.tensor(batch_support_set['text_ids'])
        batch_support_set['text_type_ids'] = torch.tensor(batch_support_set['text_type_ids'])
        batch_support_set['text_masks'] = torch.tensor(batch_support_set['text_masks'])
        batch_support_set['meta_label'] = torch.tensor(batch_support_set['meta_label']).to(torch.float)
        batch_support_set['aspect_ids'] = torch.tensor(batch_support_set['aspect_ids'])
        batch_support_set['aspect_type_ids'] = torch.tensor(batch_support_set['aspect_type_ids'])
        batch_support_set['aspect_mask'] = torch.tensor(batch_support_set['aspect_mask'])
        batch_query_set['text_ids'] = torch.tensor(batch_query_set['text_ids'])
        batch_query_set['text_type_ids'] = torch.tensor(batch_query_set['text_type_ids'])
        batch_query_set['text_masks'] = torch.tensor(batch_query_set['text_masks'])
        batch_query_set['meta_label'] = torch.tensor(batch_query_set['meta_label']).to(torch.float)
        return batch_support_set, batch_query_set

