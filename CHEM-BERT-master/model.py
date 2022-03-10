import torch
import torch.nn as nn
from Embedding import Smiles_embedding

class BERT_double_tasks(nn.Module):
	def __init__(self, model, out1, out2):
		super().__init__()
		self.bert = model
		self.linear_value = out1
		self.linear_mask = out2
	def forward(self, x, pos_num, adj_mask=None, adj_mat=None):
		x = self.bert(x, pos_num, adj_mask, adj_mat)
		return self.linear_value(x[:,0]), self.linear_mask(x)

class BERT_add_feature(nn.Module):
	def __init__(self, model, output_layer):
		super().__init__()
		self.bert = model
		self.linear = output_layer
	def forward(self, x, feature, pos_num, adj_mask=None, adj_mat=None):
		x = self.bert(x, pos_num, adj_mask, adj_mat)
		x = self.linear(torch.cat((x[:,0], feature), dim=1))
		return x

class BERT_base(nn.Module):
	def __init__(self, model, output_layer):
		super().__init__()
		self.bert = model
		self.linear = output_layer
	def forward(self, x, pos_num, adj_mask=None, adj_mat=None):
		x = self.bert(x, pos_num, adj_mask, adj_mat)
		x = self.linear(x)
		return x

class BERT_base_dropout(nn.Module):
	def __init__(self, model, output_layer):
		super().__init__()
		self.bert = model
		self.linear = output_layer
		self.drop = nn.Dropout(0.2)
	def forward(self, x, pos_num, adj_mask=None, adj_mat=None):
		x = self.bert(x, pos_num, adj_mask, adj_mat)
		x = self.linear(x)
		x = self.drop(x)
		return x

class Smiles_BERT_BC(nn.Module):
	def __init__(self, model, output_layer):
		super().__init__()
		self.smiles_bert = model
		self.linear = output_layer
	def forward(self, x, pos_num, adj_mask=None, adj_mat=None):
		x = self.smiles_bert(x, pos_num, adj_mask, adj_mat)
		x = self.linear(torch.mean(x, 1))
		return x

class classification_layer(nn.Module):
	def __init__(self, hidden):
		super().__init__()
		self.linear = nn.Linear(hidden,1)
	def forward(self, x):
		return self.linear(x)

class Masked_prediction(nn.Module):
	def __init__(self, hidden, vocab_size):
		super().__init__()
		self.linear = nn.Linear(hidden, vocab_size)
	def forward(self, x):
		return self.linear(x)

class Smiles_BERT(nn.Module):
	def __init__(self, vocab_size, max_len=256, feature_dim=1024, nhead=4, feedforward_dim=1024, nlayers=6, adj=False, dropout_rate=0):
		super(Smiles_BERT, self).__init__()
		self.embedding = Smiles_embedding(vocab_size, feature_dim, max_len, adj=adj)
		trans_layer = nn.TransformerEncoderLayer(feature_dim, nhead, feedforward_dim, activation='gelu', dropout=dropout_rate)
		self.transformer_encoder = nn.TransformerEncoder(trans_layer, nlayers)
		
		#self.linear = Masked_prediction(feedforward_dim, vocab_size)

	def forward(self, src, pos_num, adj_mask=None, adj_mat=None):
		# True -> masking on zero-padding. False -> do nothing
		#mask = (src == 0).unsqueeze(1).repeat(1, src.size(1), 1).unsqueeze(1)
		mask = (src == 0)
		mask = mask.type(torch.bool)
		#print(mask.shape)

		x = self.embedding(src, pos_num, adj_mask, adj_mat)
		x = self.transformer_encoder(x.transpose(1,0), src_key_padding_mask=mask)
		x = x.transpose(1,0)
		#x = self.linear(x)
		return x


import os
import sys

import torch
import torch.nn as nn
from transformers import BertModel, AutoModel
from loguru import logger


base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_dir)


class BaseModel(nn.Module):
	def __init__(self):
		super(BaseModel, self).__init__()
		self.mode = 'train'
		self.bert = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

		self.bert_config = self.bert.config


	@staticmethod
	def _init_weights(blocks, **kwargs):
		for block in blocks:
			for module in block.modules():
				if isinstance(module, nn.Linear):
					nn.init.zeros_(module.bias)
				elif isinstance(module, nn.Embedding):
					nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
				elif isinstance(module, nn.LayerNorm):
					nn.init.zeros_(module.bias)
					nn.init.ones_(module.weight)


class DT1Model(BaseModel):
    def __init__(self,
                 task_num):

        super(DT1Model, self).__init__()

        # 扩充词表故需要重定义
        self.bert.pooler = None
        # self.bert.resize_token_embeddings(_config.len_of_tokenizer)
        out_dims = self.bert_config.hidden_size


        self.classifier = nn.Linear(out_dims, task_num)

        # 模型初始化
        init_blocks = [self.classifier]
        self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                labels):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        out = out.last_hidden_state[:, 0, :]  # 取cls对应的embedding
        out = self.classifier(out)

        return out