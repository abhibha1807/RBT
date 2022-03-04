import sys
import os
import json
from tokenizers import BertWordPieceTokenizer
from tokenizers.processors import BertProcessing
import argparse


def train_tokenizer(filename):
    """
    Train a BertWordPieceTokenizer with the specified params and save it
    """
    # Get tokenization params
    save_location = '/content/'
    max_length = 10
    min_freq = 2
    vocabsize = 10000

    tokenizer = BertWordPieceTokenizer()
    tokenizer.do_lower_case = False
    special_tokens = ["[S]","[PAD]","[/S]","[UNK]","[MASK]", "[SEP]","[CLS]"]
    tokenizer.train_from_iterator(filename, vocab_size=vocabsize, min_frequency=min_freq, special_tokens = special_tokens)

    tokenizer._tokenizer.post_processor = BertProcessing(("[SEP]", tokenizer.token_to_id("[SEP]")), ("[CLS]", tokenizer.token_to_id("[CLS]")),)
    tokenizer.enable_truncation(max_length=max_length)

    print("Saving tokenizer ...")
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    tokenizer.save_model(save_location)
    return tokenizer


'''
Load training data and train BertWordPieceTokenizer
'''
print('training tokenizers')    



tokenizer = train_tokenizer([i[0] for i in pairs])
tokenizer = train_tokenizer([i[1] for i in pairs])


