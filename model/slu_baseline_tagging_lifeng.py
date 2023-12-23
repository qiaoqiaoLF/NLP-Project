#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import transformers

class SLUTagging(nn.Module):

    def __init__(self, config,Example_word_vocab):
        super(SLUTagging, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        self.Example_word_vocab = Example_word_vocab
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config.pretrained_model)
        self.model = transformers.AutoModel.from_pretrained(config.pretrained_model)
        for param in self.model.parameters():
            param.requires_grad = False
        # self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoderLifeng(config.hidden_size, config.num_tags, config.tag_pad_idx)

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths
        # embed = self.word_embed(input_ids)
        new_input_ids = input_ids.clone()
        for i in range(new_input_ids.shape[0]):
            for j in range(new_input_ids.shape[1]):
                word = self.Example_word_vocab.id2word[input_ids[i,j].item()]
                if word == "<pad>" or word == " ":
                    new_input_ids[i,j] = self.tokenizer.pad_token_id
                elif word == "<unk>":
                    new_input_ids[i,j] = self.tokenizer.unk_token_id
                elif word == "<s>":
                    new_input_ids[i,j] = self.tokenizer.cls_token_id
                elif word == "</s>" :
                    new_input_ids[i,j] = self.tokenizer.sep_token_id
                else:
                    token = self.tokenizer.encode(word,add_special_tokens=False)
                    assert len(token) == 1 
                    new_input_ids[i,j] = token[0]
        with torch.no_grad():
            attention_mask = new_input_ids != self.tokenizer.pad_token_id
            output = self.model(input_ids = new_input_ids,attention_mask = attention_mask )
            embed = output['last_hidden_state']
        hiddens = self.dropout_layer(embed)
        # packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True)
        # packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
        # rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)
        # hiddens = self.dropout_layer(rnn_out)
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)

        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch)
        prob = output[0]
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        if len(output) == 1:
            return predictions
        else:
            loss = output[1]
            return predictions, labels, loss.cpu().item()


class TaggingFNNDecoderLifeng(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoderLifeng, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Sequential(
            nn.Linear(input_size,input_size),
            nn.ReLU(),
            nn.LayerNorm(input_size),
            nn.Linear(input_size,num_tags)
        )
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return (prob, )

class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return (prob, )
