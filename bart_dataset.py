# coding: utf-8
# author: noctli
import json
import pickle
import logging
import copy
from typing_extensions import TypeAlias
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset
from itertools import chain

# from train import SPECIAL_TOKENS, MODEL_INPUTS, PADDED_INPUTS
BART_SPECIAL_TOKENS = ["<s>", "</s>", "<speaker1>", "<speaker2>","<cap>", "<video>", "<pad>"]
BART_SPECIAL_TOKENS_DICT = {'bos_token': "<s>", 'eos_token': "</s>", 'additional_special_tokens': ["<speaker1>", "<speaker2>", "<video>", "<cap>"], 'pad_token': "<pad>"}
MODEL_INPUTS = ["input_ids", "token_type_ids","lm_labels"]
PADDED_INPUTS = ["input_ids", "token_type_ids","lm_labels"]


def tokenize(obj,tokenizer):
    if isinstance(obj, str): # 对 string 格式的文本 tokenize
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict): # 对字典格式的文本 tokenize -> key:tokenized value
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj) # 其他情况

def get_dataset(tokenizer, data_file, feature_path=None, undisclosed_only=False, n_history=3):
    """
    input data format: read datafile: {"image_id": "", "summary": "", "dialogs""[{"answer":"","question":""}], "caption":""}
    output data format: dialog_list: num of dialogs * num of turns [{'vid':'','history':max 3 turns [[q],[a],...[cur q]],'answer':[a],'caption':[[caption list], [summary list]]}]
                        all_feature: {'vggish':{'vid':(filepath,filepath)},'i3d_flow':{},'i3d_rgb':{}}
    """
    dialog_data = json.load(open(data_file, 'r'))
    dialog_list = []
    vid_set = set()
    for dialog in dialog_data['dialogs']: # dict {}
        caption = [tokenize(dialog['caption'],tokenizer)] + [tokenize(dialog['summary'],tokenizer)] # capition 和 summary 合并 [[caption id list],[summary id list]]
        questions = [tokenize(d['question'],tokenizer) for d in dialog['dialog']] # [[question id list],[],...]
        answers = [tokenize(d['answer'],tokenizer) for d in dialog['dialog']] # [[answer id list], [], ...]
        vid = dialog["image_id"] # vid
        vid_set.add(vid) # vid set
        if undisclosed_only: # train data always false
            it = range(len(questions) - 1, len(questions))
        else: # train
            it = range(len(questions))
        qalist=[]
        history = [] # history: list
        if undisclosed_only: # test
            for n in range(len(questions)-1):
                qalist.append(questions[n])
                qalist.append(answers[n])
            history=qalist[max(-len(qalist),-n_history*2):]
        for n in it: # train range(len(questions))
            if undisclosed_only: # test
                assert dialog['dialog'][n]['answer'] == '__UNDISCLOSED__'
            question = questions[n]
            answer = answers[n]
            history.append(question)
            if n_history == 0:
                item = {'vid': vid, 'history': [question], 'answer': answer, 'caption': caption}
            else: # default 3
                item = {'vid': vid, 'history': history, 'answer': answer, 'caption': caption}
            dialog_list.append(item)
            qalist.append(question)
            qalist.append(answer)
            history=qalist[max(-len(qalist),-n_history*2):]

    all_features = {}
    if feature_path is not None:
        fea_types = ['vggish', 'i3d_flow', 'i3d_rgb']
        dataname = '<FeaType>/<ImageID>.npy'
        for ftype in fea_types:
            if undisclosed_only:
                basename = dataname.replace('<FeaType>', ftype+'_testset')
            else:
                basename = dataname.replace('<FeaType>', ftype)
            features = {}
            for vid in vid_set:
                filename = basename.replace('<ImageID>', vid)
                filepath = feature_path + filename
                features[vid] = (filepath, filepath)
            all_features[ftype] = features
        return dialog_list, all_features
        """
        dialog_list: [{'vid':'','history':'','answer':'','caption':''}]
        all_feature: {'vggish':{'vid':(filepath,filepath)},'i3d_flow':{},'i3d_rgb':{}}
        """
    return dialog_list


class AVSDDataSet(Dataset):
    def __init__(self, dialogs, tokenizer, features=None, drop_rate=0.5, train=True, model='bart', eos=2):
        self.dialogs = dialogs # dialog_list
        self.features = features # all_feature
        self.tokenizer = tokenizer
        self.drop_rate = drop_rate
        self.train = train
        self.model = model
        self.eos = [eos]

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, index):
        dialog = self.dialogs[index]
        vid = dialog['vid']
        his = self.dialogs[index]['history'] # [[q],[a],...[q]] 
        cap = self.dialogs[index]['caption'] # [[caption ], [summary]]
        ans = self.dialogs[index]['answer'] # [[a]]
        
        if np.random.rand() < self.drop_rate: 
            encoder_instance, decoder_instance, _, _ = build_input_from_segments(cap, his, ans, self.tokenizer, video=False, drop_caption=True, train=self.train, model=self.model)
        else: # train/validate: drop_rate = 0
            encoder_instance, decoder_instance, _, _ = build_input_from_segments(cap, his, ans, self.tokenizer, video=False, drop_caption=False, train=self.train, model=self.model)
        encoder_input_ids = torch.Tensor(encoder_instance["input_ids"]).long() 
        decoder_input_ids = torch.Tensor(decoder_instance["input_ids"]).long()
        encoder_token_type_ids = torch.Tensor(encoder_instance["token_type_ids"]).long() #
        decoder_token_type_ids = torch.Tensor(decoder_instance["token_type_ids"]).long() #
        lm_labels = torch.Tensor(decoder_instance["lm_labels"]).long()
        type_labels = torch.Tensor(decoder_instance["type_labels"]).long()

        # if self.features is not None:
        #     # add eos before lan
        #     eos = torch.Tensor(self.eos).long()
        #     # encoder_input_ids = torch.cat([eos, encoder_input_ids], dim=1)
        #     decoder_input_ids = torch.cat([eos, decoder_input_ids])
        #     # encoder_token_type_ids = torch.cat([eos, encoder_token_type_ids], dim=1)
        #     decoder_token_type_ids = torch.cat([eos, decoder_token_type_ids])
        #     lm_labels = torch.cat([eos, lm_labels])
        #     type_labels = torch.cat([eos, type_labels])

        if self.features is not None:
            try:
                vgg = np.load(self.features[0]["vggish"][vid][0]) 
                i3d_flow = np.load(self.features[0]["i3d_flow"][vid][0]) 
                i3d_rgb = np.load(self.features[0]["i3d_rgb"][vid][0]) 
            except KeyError: # validate_data
                vgg = np.load(self.features[1]["vggish"][vid][0])
                i3d_flow = np.load(self.features[1]["i3d_flow"][vid][0])
                i3d_rgb = np.load(self.features[1]["i3d_rgb"][vid][0])
            
            # sample_step = i3d_flow.shape[0] // vgg.shape[0]
            # if sample_step == 0:
            #     sample_step = 1
            sample_step = 1

            sample_i3d_flow = i3d_flow[range(1, i3d_flow.shape[0], sample_step)]
            sample_i3d_rgb = i3d_rgb[range(1, i3d_rgb.shape[0], sample_step)]

            vgg = torch.from_numpy(vgg).float()
            i3d_flow = torch.from_numpy(sample_i3d_flow).float()
            i3d_rgb = torch.from_numpy(sample_i3d_rgb).float()
            min_length = min([i3d_flow.size(0), i3d_rgb.size(0), vgg.size(0)])
            i3d = torch.cat([i3d_flow[:min_length], i3d_rgb[:min_length], vgg[:min_length]], dim=1)

            return encoder_input_ids, encoder_token_type_ids, decoder_input_ids, decoder_token_type_ids, lm_labels, i3d, type_labels
        else:
            return encoder_input_ids, encoder_token_type_ids, decoder_input_ids, decoder_token_type_ids, lm_labels, type_labels


def collate_fn(batch, pad_token, features=None):
    def padding(seq, pad_token):
        max_len = max([i.size(0) for i in seq])
        if len(seq[0].size()) == 1:
            result = torch.ones((len(seq), max_len)).long() * pad_token
        else:
            result = torch.ones((len(seq), max_len, seq[0].size(-1))).float()
        for i in range(len(seq)):
            result[i, :seq[i].size(0)] = seq[i]
        return result

    encoder_input_ids_list, encoder_token_type_ids_list, decoder_input_ids_list, decoder_token_type_ids_list,lm_labels_list, i3d_list, type_labels_list = [], [], [], [], [], [], []
    for i in batch:
        encoder_input_ids_list.append(i[0])
        encoder_token_type_ids_list.append(i[1])
        decoder_input_ids_list.append(i[2])
        decoder_token_type_ids_list.append(i[3])
        lm_labels_list.append(i[4])
        if features is not None:
            i3d_list.append(i[5])
            type_labels_list.append(i[6])
        else:
            type_labels_list.append(i[5])
        

    encoder_input_ids = padding(encoder_input_ids_list, pad_token)
    encoder_token_type_ids = padding(encoder_token_type_ids_list, pad_token)
    decoder_input_ids = padding(decoder_input_ids_list, pad_token)
    decoder_token_type_ids = padding(decoder_token_type_ids_list, pad_token)
    lm_labels = padding(lm_labels_list, -1)
    type_labels = padding(type_labels_list, -1)
    encoder_input_mask = encoder_input_ids != pad_token
    decoder_input_mask = decoder_input_ids != pad_token
    if features is not None:
        i3d = padding(i3d_list, pad_token)
        i3d_mask = torch.sum(i3d != 1, dim=2) != 0
        encoder_input_mask = torch.cat([i3d_mask, encoder_input_mask], dim=1) 
        return encoder_input_ids, encoder_token_type_ids, encoder_input_mask, decoder_input_ids, decoder_token_type_ids, decoder_input_mask, lm_labels, i3d, type_labels
    else:
        return encoder_input_ids, encoder_token_type_ids, encoder_input_mask, decoder_input_ids, decoder_token_type_ids, decoder_input_mask, lm_labels, type_labels


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padd only batches but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "labels" else -1] * (max_l - len(x)) for x in dataset[name]]
    return dataset

def build_input_from_segments(caption, history, reply, tokenizer, with_eos=True, video=False, drop_caption=False, train=True, model='bart'):
    """
    caption: [[caption], [summary]] history: [[q], [a], ..., [q]], reply: [a]  other: default if train dataset
    """
    """ Build a sequence of input from 3 segments: caption(caption+summary) history and last reply """
    SPECIAL_TOKENS = BART_SPECIAL_TOKENS
    bos, eos, speaker1, speaker2, cap = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-2])
    if not drop_caption: # train/validate/test
        instance = {}
        encoder_instance = {}
        decoder_instance = {}
        sequence = [[bos] + list(chain(*caption))] + history + [reply + ([eos] if with_eos else [])] 
        # [[bos, caption]] + [[q], [a], ...] + [[a, eos]] -> [[bos, caption], [q], [a], ... [a, eos]] # train with_eos
        encoder_sequence = [[bos] + list(chain(*caption))] + history
        # [[bos, caption]] + [[q], [a], ...] ] -> [[bos, caption], [q], [a], ... [q]]
        decoder_sequence = reply + ([eos] if with_eos else [])
        # [a, eos]
        sequence = [[cap] + sequence[0] + [eos]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])] 
        # [[cap, bos, caption, eos]] + [[speaker1, q], [speaker2, a], ..., [speaker1, a, eos]]
        # -> [[cap, bos, caption, eos], [speaker1, q], [speaker2, a], ..., [speaker2, a, eos] ]
        encoder_sequence = [[cap] + encoder_sequence[0] + [eos]] + [[speaker2 if (len(encoder_sequence)-i) % 2 else speaker1] + s for i, s in enumerate(encoder_sequence[1:])]
        # [[cap, bos, caption, eos]] + [[speaker1, q], [speaker2, a], ..., [speaker1, q]]
        # -> [[cap, bos, caption, eos], [speaker1, q], [speaker2, a], ..., [speaker2, q]]
        decoder_sequence = [[speaker2] + decoder_sequence]
        # [[speaker2, a, eos]]
        instance["input_ids"] = list(chain(*sequence))
        # [cap, bos, caption eos, speaker1, q, ..., speaker2, a, eos]
        encoder_instance["input_ids"] = list(chain(*encoder_sequence))
        # [cap, bos, caption eos, speaker1, q, ..., speaker1, q]

        # way 1 不带历史
        decoder_instance["input_ids"] = list(chain(*decoder_sequence))
        # [speaker2, a, eos]
        # # way 2 带历史
        # decoder_instance["input_ids"] = instance["input_ids"]
        # [cap, bos, caption eos, speaker1, q, ..., speaker2, a, eos]

        instance["cap_type_ids"] = [cap] * len(sequence[0])
        # [cap, ...]
        instance["history_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(encoder_sequence[1:]) for _ in s]
        # [speaker1, ..., speaker2, ...]
        instance["token_type_ids"] = [cap] * len(sequence[0]) + [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:]) for _ in s]
        # [cap, ...] + [speaker1, ..., speaker2, ...] -> [cat, ..., speaker1, ..., speaker2, ...]
        encoder_instance["token_type_ids"] = [cap] * len(encoder_sequence[0]) + [speaker2 if i % 2 else speaker1 for i, s in enumerate(encoder_sequence[1:]) for _ in s]
        # [cap, ...] + [speaker1, ..., speaker2, ...] -> [cat, ..., speaker1, ..., speaker2, ..., speaker1]
        decoder_instance["token_type_ids"] = [speaker2 for s in decoder_sequence for _ in s]
        if video and train: # if use caption loss
            #instance["lm_labels"] = sequence[0] + ([-1]*sum(len(s) for s in sequence[1:-1])) + sequence[-1]
            instance["lm_labels"] = sequence[0] + ([-1]*sum(len(s) for s in sequence[1:-1])) + sequence[-1]
            # [cap, bos, caption, eos] + [-1, ... ] + [speaker2, a, eos] -> [cap, bos, caption, -1, ..., speaker2, a, eos]
            instance["type_labels"] = instance["cap_type_ids"] + ([-1]*sum(len(s) for s in sequence[1:-1])) + decoder_instance["token_type_ids"]
            
            # way 1 不带历史
            decoder_instance["lm_labels"] = encoder_sequence[0] + decoder_sequence[0]
            decoder_instance["type_labels"] = instance["cap_type_ids"] + decoder_instance["token_type_ids"]

            # # way 2 带历史
            # decoder_instance["lm_labels"] = instance["lm_labels"]
            # decoder_instance["type_labels"] = instance["type_labels"]
            
        else: # wo caption
            instance["lm_labels"] = ([-1]*sum(len(s) for s in sequence[:-1])) + sequence[-1]
            # [-1,..., speaker2, a, eos]
            instance['type_labels'] = ([-1]*sum(len(s) for s in sequence[:-1])) + decoder_instance["token_type_ids"]

            # way 1 不带历史
            decoder_instance["lm_labels"] = sequence[-1]
            # [speaker2, a, eos]
            decoder_instance["type_labels"] = decoder_instance["token_type_ids"]

            # # way 2 带历史
            # decoder_instance["lm_labels"] = instance["lm_labels"]
            # decoder_instance["type_labels"] = instance["type_labels"]
            
    else:
        instance = {}
        encoder_instance = {}
        decoder_instance = {}
        sequence = history + [reply + ([eos] if with_eos else [])]
        # [[q], [a], ..., [q]] + [[a, eos]] -> [[q], [a], ..., [q], [a, eos]]
        encoder_sequence = history
        # [[q], [a], ..., [q]]
        decoder_sequence = reply + ([eos] if with_eos else [])
        # [a, eos]
        sequence = [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence)]
        # [[speaker1, q], ..., [speaker2, a, eos]]
        encoder_sequence = [[speaker2 if (len(encoder_sequence)-i) % 2 else speaker1] + s for i, s in enumerate(encoder_sequence)]
        # [[speaker1, q], ..., ]
        decoder_sequence = [[speaker2] + decoder_sequence]
        # [speaker2, a, eos]
        instance["input_ids"] = list(chain(*sequence))
        # [speaker1, q, ..., speaker2, a, eos]
        encoder_instance["input_ids"] = list(chain(*encoder_sequence))
        # [speaker1, q, ..., speaker1, q]

        # way1 不带历史
        decoder_instance["input_ids"] = list(chain(*decoder_sequence))
        # way2 带历史
        # decoder_instance["input_ids"] = instance["input_ids"]

        instance["history_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(encoder_sequence) for _ in s]
        instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
        # [speaker1, ..., speaker2, ...]
        decoder_instance["token_type_ids"] = [speaker2 for s in decoder_sequence for _ in s]
        if video:
            instance["lm_labels"] = ([-1]*sum(len(s) for s in sequence[:-1])) + sequence[-1]
            # [-1, ..., speaker2, a, eos]
            instance['type_labels'] = ([-1]*sum(len(s) for s in sequence[:-1])) + decoder_instance["token_type_ids"]

            # way1 不带历史
            decoder_instance["lm_labels"] = sequence[-1]
            decoder_instance["type_labels"] = decoder_instance["token_type_ids"]

            # # way2 带历史
            # decoder_instance["lm_labels"] = instance["lm_labels"]
            # decoder_instance["type_labels"] = instance["type_labels"]

        else:
            instance["lm_labels"] = ([-1]*sum(len(s) for s in sequence[:-1])) + sequence[-1]
            # [-1, ..., speaker2, a, eos]
            instance['type_labels'] = ([-1]*sum(len(s) for s in sequence[:-1])) + decoder_instance["token_type_ids"]

            # way1 不带历史
            decoder_instance["lm_labels"] = sequence[-1]
            decoder_instance["type_labels"] = decoder_instance["token_type_ids"]

            # # way2 带历史
            # decoder_instance["lm_labels"] = instance["lm_labels"]
            # decoder_instance["type_labels"] = instance["type_labels"]

    return encoder_instance, decoder_instance, instance, sequence


