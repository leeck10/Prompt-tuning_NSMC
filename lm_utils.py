import pandas as pd
import os
import pickle
from pathlib import Path
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from transformers import GPTJForCausalLM, GPTJForSequenceClassification
from torch.nn import CrossEntropyLoss

class TSVDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train', block_size=512, get_annotations=False):
        self.print_count = 5
        self.eos_token_id = tokenizer.eos_token_id

        cached_features_file, data = self.load_data(file_path, block_size, args)
        self.data = data

        if get_annotations: cached_features_file = cached_features_file + '_annotated'

        if os.path.exists(cached_features_file):
            print ('Loading features from', cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
            return

        print ('Saving features from ', file_path, ' into ', cached_features_file) 

        examples = []
        lable_to_str = {0: '부정', 1: '긍정'}
        for line in data:
            sent, label = line
            if args.in_len:
                text1 = sent
            else:
                text1 = '{}='.format(sent)
            tokenized_text1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text1))
            prompt_length = len(tokenized_text1)
            tokenized_text, total_length = tokenized_text1, len(tokenized_text1)
            if get_annotations:
                text2 = lable_to_str[label]
                tokenized_text2 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text2))
                tokenized_text = tokenized_text1 + tokenized_text2
                tokenized_text = tokenized_text + [self.eos_token_id]
                total_length = len(tokenized_text)
                if len(tokenized_text) > block_size:
                    breakpoint()
                    tokenized_text = tokenized_text[:block_size]
                if len(tokenized_text) < block_size:
                    tokenized_text = tokenized_text + [self.eos_token_id] * (block_size-len(tokenized_text))
            if self.print_count > 0:
                print ('example: ', text1 + text2 if get_annotations else text1)
                self.print_count = self.print_count - 1
            examples.append((tokenized_text, prompt_length, total_length, label))
        self.examples = examples

        print ('Saving ', len(self.examples), ' examples')
        with open(cached_features_file, 'wb') as handle:
            pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item][0]), self.examples[item][1], self.examples[item][2], self.examples[item][3]

    def get_example_text(self, index):
        return self.data['prompt'][index]

    def add_explanation(self, index, explanation):
        explanation_name = 'Generated_Explanation'
        self.data.at[self.data.index[index], explanation_name] = explanation

    def load_data(self, file_path, block_size, args):
        assert os.path.isfile(file_path)
        data = []
        for line in open(file_path, 'r'):
            _, sent, label = line.split('\t')
            data.append((sent, int(label)))
        directory, filename = os.path.split(file_path)
        if args.in_len:
            cached_features_file = os.path.join(directory, 'cached_lm_in_{}_{}'.format(block_size, filename))
        else:
            cached_features_file = os.path.join(directory, 'cached_lm_{}_{}'.format(block_size, filename))
        return cached_features_file, data

    def save(self, filename):
        self.data.to_csv(filename, sep='\t')


class GPTPromptTuningMixin:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        soft_prompt_path: str = None,
        pre_n_tokens: int = 0,
        in_n_tokens: int = 0,
        initialize_from_vocab: bool = True,
        random_range: float = 0.5,
        lm_head_tuning: bool = False,
        **kwargs,
    ):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Make sure to freeze Tranformers model
        if lm_head_tuning:
            for name, param in model.named_parameters():
                if name not in ['lm_head.weight', 'lm_head.bias']:
                    param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = False

        if soft_prompt_path is not None:
            model.set_soft_prompt_embeds(soft_prompt_path, pre_n_tokens, in_n_tokens, lm_head_tuning)
        else:
            print("Initializing soft prompt...")
            model.initialize_soft_prompt(
                pre_n_tokens=pre_n_tokens,
                in_n_tokens=in_n_tokens,
                initialize_from_vocab=initialize_from_vocab,
                random_range=random_range,
                lm_head_tuning=lm_head_tuning,
            )

        return model

    def set_soft_prompt_embeds(
        self,
        soft_prompt_path: str,
        pre_n_tokens: int,
        in_n_tokens: int,
        lm_head_tuning: bool,
    ) -> None:
        """
        Args:
            soft_prompt_path: torch soft prompt file path
        """
        self.soft_prompt = torch.load(
            os.path.join(soft_prompt_path, "soft_prompt.model"), map_location=torch.device("cpu")
        )
        self.pre_n_tokens = pre_n_tokens
        self.in_n_tokens = in_n_tokens

        self.lm_head_tuning = lm_head_tuning
        if lm_head_tuning:
            self.lm_head = torch.load(
                os.path.join(soft_prompt_path, "lm_head.model"), map_location=torch.device("cpu")
            )
        print(f"Set soft prompt! (n_tokens: {self.n_tokens})")
        print(f"Set answer soft prompt! (n_tokens: {self.answer_n_tokens})")

    def initialize_soft_prompt(
        self,
        pre_n_tokens: int = 0,
        in_n_tokens: int = 0,
        initialize_from_vocab: bool = True,
        random_range: float = 0.5,
        lm_head_tuning: bool = False,
    ) -> None:
        self.pre_n_tokens = pre_n_tokens
        self.in_n_tokens = in_n_tokens
        self.lm_head_tuning = lm_head_tuning

        if initialize_from_vocab:
            init_prompt_value = self.transformer.wte.weight[:pre_n_tokens + in_n_tokens].clone().detach()
        else:
            init_prompt_value = torch.FloatTensor(pre_n_tokens + in_n_tokens, self.transformer.wte.weight.size(1)).uniform_(
                -random_range, random_range
            )

        self.soft_prompt = nn.Embedding(pre_n_tokens + in_n_tokens, self.config.n_embd)

        # Initialize weight
        self.soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)

    def _cat_learned_embedding_to_input(self, input_ids, lengths, labels=None) -> torch.Tensor:
        # [batch_size, n_tokens, n_embd]
        if self.in_n_tokens:
            if labels is not None:
                input_ids = torch.cat((input_ids, torch.ones((input_ids.size(0), self.in_n_tokens)).type_as(input_ids)), 1)
                input_ids[torch.arange(input_ids.size(0)), lengths + self.in_n_tokens] = input_ids[torch.arange(input_ids.size(0)), lengths]
                inputs_embeds = self.transformer.wte(input_ids)
                if len(list(inputs_embeds.shape)) == 2:
                    inputs_embeds = inputs_embeds.unsqueeze(0)
                in_index = torch.arange(self.in_n_tokens).repeat(inputs_embeds.size(0), 1).to(self.device) + lengths.unsqueeze(-1)
                inputs_embeds[torch.arange(inputs_embeds.size(0)).unsqueeze(-1), in_index] = self.soft_prompt.weight[self.pre_n_tokens:].repeat(inputs_embeds.size(0), 1, 1)
            else:
                inputs_embeds = self.transformer.wte(input_ids)
                if len(list(inputs_embeds.shape)) == 2:
                    inputs_embeds = inputs_embeds.unsqueeze(0)
                inputs_embeds = torch.cat((inputs_embeds, self.soft_prompt.weight[self.pre_n_tokens:].repeat(inputs_embeds.size(0), 1, 1)), dim=1)
        else:
            inputs_embeds = self.transformer.wte(input_ids)
            if len(list(inputs_embeds.shape)) == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)
        if self.pre_n_tokens:
            learned_embeds = self.soft_prompt.weight[:self.pre_n_tokens].repeat(inputs_embeds.size(0), 1, 1) 
            inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

        return inputs_embeds

    def _extend_labels(self, labels, ignore_index=-100) -> torch.Tensor:
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)
        n_batches = labels.shape[0]

        return torch.cat(
            [
                torch.full((n_batches, self.pre_n_tokens + self.in_n_tokens), ignore_index).to(self.device),
                labels,
            ],
            dim=1,
        )

    def _extend_attention_mask(self, attention_mask):

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [torch.full((n_batches, self.n_tokens), 1).to(self.device), attention_mask],
            dim=1,
        )

    def save_soft_prompt(self, path: str, filename: str = "soft_prompt.model"):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.soft_prompt, os.path.join(path, filename))
        if self.lm_head_tuning:
            torch.save(self.lm_head, os.path.join(path, "lm_head.model"))

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        lengths=None
    ):

        if input_ids is not None:
            inputs_embeds = self._cat_learned_embedding_to_input(input_ids, lengths, labels=labels).to(self.device)

        if labels is not None:
            labels = self._extend_labels(labels).to(self.device)

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask).to(self.device)

        # Drop most of the args for now
        return super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            return_dict=return_dict,
        )

class KoGPTPromptTuningLM(GPTPromptTuningMixin, GPTJForCausalLM):
    def __init__(self, config):
        super().__init__(config)