import re, os
import glob
import collections
import numpy as np
import pandas as pd
import random
import torch
import codecs
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch.utils.data import Dataset
from SmilesPE.tokenizer import SPE_Tokenizer
from transformers import PreTrainedTokenizer
from typing import List, Optional

## Based on github.com/codertimo/BERT-pytorch

class Vocab(object):
	def __init__(self):
		self.pad_index = 0
		self.mask_index = 1
		self.unk_index = 2
		self.start_index = 3
		self.end_index = 4

		# check 'Na' later
		self.voca_list = ['<pad>', '<mask>', '<unk>', '<start>', '<end>'] + ['C', '[', '@', 'H', ']', '1', 'O', \
							'(', 'n', '2', 'c', 'F', ')', '=', 'N', '3', 'S', '/', 's', '-', '+', 'o', 'P', \
							 'R', '\\', 'L', '#', 'X', '6', 'B', '7', '4', 'I', '5', 'i', 'p', '8', '9', '%', '0', '.', ':', 'A']

		self.dict = {s: i for i, s in enumerate(self.voca_list)}

	def __len__(self):
		return len(self.voca_list)

class SmilesDataset(Dataset):
	def __init__(self, smiles_path, vocab, seq_len, mat_position):
		self.vocab = vocab
		self.atom_vocab = ['C', 'O', 'n', 'c', 'F', 'N', 'S', 's', 'o', 'P', 'R', 'L', 'X', 'B', 'I', 'i', 'p', 'A']
		self.smiles_dataset = []
		self.adj_dataset = []
		self.seq_len = seq_len
		self.mat_pos = mat_position

		folder_list = os.listdir(smiles_path)

		for folder in folder_list:
			smiles_data = glob.glob(smiles_path + "/" + folder + "/*.smi")
			#print(smiles_data)
			for small_data in smiles_data:
				text = pd.read_csv(small_data, sep=" ")
				smiles_list = np.asarray(text['smiles'])
				for i in smiles_list:
					#adj_mat = GetAdjacencyMatrix(Chem.MolFromSmiles(i))
					#self.adj_dataset.append(self.zero_padding(adj_mat, (seq_len, seq_len)))
					self.adj_dataset.append(i)

					self.smiles_dataset.append(self.replace_halogen(i))

	def __len__(self):
		return len(self.smiles_dataset)

	def __getitem__(self, idx):
		item = self.smiles_dataset[idx]
		input_random, input_label, input_adj_mask = self.random_masking(item)

		input_data = [self.vocab.start_index] + input_random + [self.vocab.end_index]
		input_label = [self.vocab.pad_index] + input_label + [self.vocab.pad_index]
		input_adj_mask = [0] + input_adj_mask + [0]
		# give info to start token
		if self.mat_pos == 'start':
			input_adj_mask = [1] + [0 for _ in range(len(input_adj_mask)-1)]

		smiles_bert_input = input_data[:self.seq_len]
		smiles_bert_label = input_label[:self.seq_len]
		smiles_bert_adj_mask = input_adj_mask[:self.seq_len]

		padding = [0 for _ in range(self.seq_len - len(smiles_bert_input))]
		smiles_bert_input.extend(padding)
		smiles_bert_label.extend(padding)
		smiles_bert_adj_mask.extend(padding)
		mol = Chem.MolFromSmiles(self.adj_dataset[idx])
		smiles_bert_value = QED.qed(mol)

		adj_mat = GetAdjacencyMatrix(mol)
		smiles_bert_adjmat = self.zero_padding(adj_mat, (self.seq_len, self.seq_len))

		output = {"smiles_bert_input": smiles_bert_input, "smiles_bert_label": smiles_bert_label,  \
					"smiles_bert_adj_mask": smiles_bert_adj_mask, "smiles_bert_adjmat": smiles_bert_adjmat, "smiles_bert_value": smiles_bert_value}

		return {key:torch.tensor(value) for key, value in output.items()}

	def random_masking(self,smiles):
		tokens = [i for i in smiles]
		output_label = []
		adj_masking = []

		for i, token in enumerate(tokens):
			if token in self.atom_vocab:
				adj_masking.append(1)
			else:
				adj_masking.append(0)

			prob = random.random()
			if prob < 0.15:
				prob /= 0.15

				if prob < 0.8:
					tokens[i] = self.vocab.mask_index

				# replace the token except special token
				elif prob < 0.9:
					tokens[i] = random.randrange(5,len(self.vocab))

				else:
					tokens[i] = self.vocab.dict.get(token, self.vocab.unk_index)

				output_label.append(self.vocab.dict.get(token, self.vocab.unk_index))

			else:
				tokens[i] = self.vocab.dict.get(token, self.vocab.unk_index)
				output_label.append(self.vocab.pad_index) #modify to num

		return tokens, output_label, adj_masking

	def replace_halogen(self,string):
	    """Regex to replace Br,Cl,Sn,Na with single letters"""
	    br = re.compile('Br')
	    cl = re.compile('Cl')
	    sn = re.compile('Sn')
	    na = re.compile('Na')
	    string = br.sub('R', string)
	    string = cl.sub('L', string)
	    string = sn.sub('X', string)
	    string = na.sub('A', string)
	    return string

	def zero_padding(self, array, shape):
		padded = np.zeros(shape, dtype=np.float32)
		padded[:array.shape[0], :array.shape[1]] = array
		return padded

	def construct_vocab(self,smiles, vocab):
		smiles = replace_halogen(smiles)
		for i in smiles:
			if i not in vocab:
				vocab.append(i)
		return vocab

	def smiles_tokenizer(self,smiles):
		smiles = replace_halogen(smiles)
		char_list = [i for i in smiles]
		return char_list


def build_vocab(path):
	vocab = []
	folder_list = os.listdir(path)
	for folder in folder_list:
		smiles_data = glob.glob(path + "/"+folder+"/*.smi")
		for i in smiles_data:
			text = pd.read_csv(i, sep=" ")
			smiles_list = np.asarray(text['smiles'])
			for j in smiles_list:
				vocab = construct_vocab(j, vocab)
	print(vocab)

def load_vocab(vocab_file):
    """
    读取静态存储的spe切分字典，该字典的token的id为了与PubMedBERT的字典合并，整体进行了偏移。并且Special Token的ID采用和PubMedBERT相同
    :param vocab_file: 字典的路径
    :return:
      vocab (:obj:'dict', )({token:id, ...}):
    """
    vocab = collections.OrderedDict()  # OrderedDict是为了兼容py3.6之前的python，这时的python字典是无序的。
    with open(vocab_file, "r", encoding="utf-8") as reader:
        lines = reader.readlines()
    for line in lines:
        token, index = line.split(' ')
        vocab[token] = int(index.rstrip("\n"))
    return vocab


# 定义SMILES的Tokenizer
class SMILES_SPE_Tokenizer(PreTrainedTokenizer):
    r"""
    Constructs a SMILES tokenizer. Based on SMILES Pair Encoding (https://github.com/XinhaoLi74/SmilesPE).
    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.
    Args:
        vocab_file (:obj:`string`):
            File containing the vocabulary.
        spe_file (:obj:`string`):
            File containing the trained SMILES Pair Encoding vocabulary.
        unk_token (:obj:`string`, `optional`, defaults to "[UNK]"):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`string`, `optional`, defaults to "[SEP]"):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences
            for sequence classification or for a text and a question for question answering.
            It is also used as the last token of a sequence built with special tokens.
        pad_token (:obj:`string`, `optional`, defaults to "[PAD]"):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`string`, `optional`, defaults to "[CLS]"):
            The classifier token which is used when doing sequence classification (classification of the whole
            sequence instead of per-token classification). It is the first token of the sequence when built with
            special tokens.
        mask_token (:obj:`string`, `optional`, defaults to "[MASK]"):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
    """

    def __init__(
            self,
            vocab_file,
            spe_file,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'.".format(vocab_file)
            )
        if not os.path.isfile(spe_file):
            raise ValueError(
                "Can't find a SPE vocabulary file at path '{}'.".format(spe_file)
            )
        self.vocab = load_vocab(vocab_file)
        self.spe_vocab = codecs.open(spe_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.spe_tokenizer = SPE_Tokenizer(self.spe_vocab)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **self.added_tokens_encoder)

    def _tokenize(self, text, **kwargs):
        return self.spe_tokenizer.tokenize(text).split(' ')

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A BERT sequence has the following format:
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True if the token list is already formatted with special tokens for the model
        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A BERT sequence pair mask has the following format:
        ::
            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |
        if token_ids_1 is None, only returns the first portion of the mask (0's).
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, vocab_path, filename_prefix: Optional[str] = None):
        """
        Save the sentencepiece vocabulary (copy original file) and special tokens file to a directory.
        Args:
            vocab_path (:obj:`str`):
                The directory in which to save the vocabulary.
            filename_prefix:
                父类的默认参数
        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, 'vocab.txt')
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        "Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file)
                    )
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)

