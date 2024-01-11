import json

from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator

class Example():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path):
        dataset = json.load(open(data_path, 'r'))
        examples = []
        for di, data in enumerate(dataset):
            curr_word = None
            curr_word2 = None
            
            for ui, utt in enumerate(data):
                max_len = max(len(utt['asr_1best']), 20)
                # breakpoint()
                if curr_word == None:
                    utt['curr_len'] = len(utt['asr_1best'])
                    curr_word = utt['asr_1best']
                    curr_word2 = utt['manual_transcript']
                    
                else:
                    utt['curr_len'] = len(utt['asr_1best'])
                    utt['asr_1best'] = curr_word + utt['asr_1best']
                    if len(utt['asr_1best'])>max_len:
                        utt['asr_1best'] = utt['asr_1best'][-max_len:]
                        

                    utt['manual_transcript'] = curr_word2 + utt['manual_transcript']
                    if len(utt['manual_transcript'])>max_len:
                        utt['manual_transcript'] = utt['manual_transcript'][-max_len:]

                ex = cls(utt, f'{di}-{ui}')
                examples.append(ex)
        return examples

    def __init__(self, ex: dict, did):
        super(Example, self).__init__()
        # breakpoint()
        self.ex = ex
        self.did = did

        self.utt = ex['asr_1best']
        self.slot = {}
        for label in ex['semantic']:
            act_slot = f'{label[0]}-{label[1]}'
            if len(label) == 3:
                self.slot[act_slot] = label[2]
        self.tags = ['O'] * len(self.utt)
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1:
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        self.input_idx = [Example.word_vocab[c] for c in self.utt]
        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
