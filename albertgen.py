from random import randint, shuffle
from random import random as rand
import multiprocessing as mp
import queue
import setproctitle

import torch

import tokenization
from utils import set_seeds, get_device, truncate_tokens_pair, _sample_mask


from datagen import DataGen, DataFeeder, DataWorker


# Input file format :
# 1. One sentence per line. These should ideally be actual sentences,
#    not entire paragraphs or arbitrary spans of text. (Because we use
#    the sentence boundaries for the "next sentence prediction" task).
# 2. Blank lines between documents. Document boundaries are needed
#    so that the "next sentence prediction" task doesn't span between documents.

def seek_random_offset(f, back_margin=2000):
    """ seek random offset of file pointer """
    f.seek(0, 2)
    # we remain some amount of text to read
    max_offset = f.tell() - back_margin
    f.seek(randint(0, max_offset), 0)
    f.readline() # throw away an incomplete sentence

class SentPairDataMaker():
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, tokenize, max_len, short_sampling_prob=0.1, pipeline=[]):
        super().__init__()
        self.tokenize = tokenize # tokenize function
        self.max_len = max_len # maximum length of tokens
        self.short_sampling_prob = short_sampling_prob
        self.pipeline = pipeline
    
    def read_tokens_2x(self, par, length):
        """ Read tokens from file pointer with limited length """
        t1, t2 = [], []
        lines = iter(par.split("\n"))
        
        try:
            while len(t1) < length:
                t1.extend(self.tokenize(next(lines).strip()))
                
            while len(t2) < length:
                t2.extend(self.tokenize(next(lines).strip()))
        except StopIteration:
            pass
            #print("Input too short", repr(par))

        return t1, t2

    def create_example(self, par):
        # sampling length of each tokens_a and tokens_b
        # sometimes sample a short sentence to match between train and test sequences
        # ALBERT is same  randomly generate input
        # sequences shorter than 512 with a probability of 10%.
        len_tokens = randint(1, int(self.max_len / 2)) \
            if rand() < self.short_sampling_prob \
            else int(self.max_len / 2)

        is_next = rand() < 0.5 # whether token_b is next to token_a or not

        tokens_a, tokens_b = self.read_tokens_2x(par, len_tokens)

        # SOP, sentence-order prediction
        instance = (is_next, tokens_a, tokens_b) if is_next \
            else (is_next, tokens_b, tokens_a)

        for proc in self.pipeline:
            instance = proc(instance)
            
        return instance


class Pipeline():
    """ Pre-process Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError
    
class Preprocess4Pretrain(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len,
                 mask_alpha, mask_beta, max_gram):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred # max tokens of prediction
        self.mask_prob = mask_prob # masking probability
        self.vocab_words = vocab_words # vocabulary (sub)words

        self.indexer = indexer # function from token to token index
        self.max_len = max_len
        self.mask_alpha = mask_alpha
        self.mask_beta = mask_beta
        self.max_gram = max_gram

    def __call__(self, instance):
        is_next, tokens_a, tokens_b = instance

        # -3  for special tokens [CLS], [SEP], [SEP]
        truncate_tokens_pair(tokens_a, tokens_b, self.max_len - 3)

        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        segment_ids = [0]*(len(tokens_a)+2) + [1]*(len(tokens_b)+1)
        input_mask = [1]*len(tokens)

        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = min(self.max_pred, max(1, int(round(len(tokens) * self.mask_prob))))

        # For masked Language Models
        masked_tokens, masked_pos, tokens = _sample_mask(tokens, self.mask_alpha,
                                            self.mask_beta, self.max_gram,
                                            goal_num_predict=n_pred)

        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        input_ids = self.indexer(tokens)
        masked_ids = self.indexer(masked_tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        input_mask.extend([0]*n_pad)

        # Zero Padding for masked target
        if self.max_pred > len(masked_ids):
            masked_ids.extend([0] * (self.max_pred - len(masked_ids)))
        if self.max_pred > len(masked_pos):
            masked_pos.extend([0] * (self.max_pred - len(masked_pos)))
        if self.max_pred > len(masked_weights):
            masked_weights.extend([0] * (self.max_pred - len(masked_weights)))

        return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next)

    
class AlbertFeder(DataFeeder):
    def feed(self):
        while True:
            with open(self.config.args.data_file) as f:
                for par in self._split_pars(f):
                    if "=" in par[:5]:
                        continue
                    self.queue.put(par)

    @staticmethod
    def _split_pars(f):
        par = []
        for line in f:
            line = line.strip()
            if not line:
                if par:
                    yield "\n".join(par)
                    par = []
            par.append(line)
        
            
class AlbertWorker(DataWorker):
    def run(self):
        try:
            setproctitle.setproctitle(self.name)
            self.context = self._init_context()

            pars = []
            while self.should_run:
                try:
                    par = self.inqueue.get(block=True, timeout=1)
                except queue.Empty:
                    continue
                
                pars.append(par)
                if len(pars) == self.batch_size:
                    example = self._create_example(pars)
                    self.outqueue.put(example)
                    pars = []

        except KeyboardInterrupt:
            print(f"Worker {self.idx} interrupted!\n", flush=True, end="")

        self._destroy_context(self.context)
    
    def _create_example(self, pars):
        batch = []
        for par in pars:
            batch.append(self.data_maker.create_example(par))
        
        # To Tensor
        batch_tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*batch)]
        return batch_tensors
    
    
    def _init_context(self):
        cfg = self.config
        self.batch_size = cfg.pre.batch_size
        
        tokenizer = tokenization.FullTokenizer(vocab_file=cfg.args.vocab, do_lower_case=True)
        tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))
        
        pipeline = [
            Preprocess4Pretrain(
                cfg.args.max_pred,
                cfg.args.mask_prob,
                list(tokenizer.vocab.keys()),
                tokenizer.convert_tokens_to_ids,
                cfg.model.max_len,
                cfg.args.mask_alpha,
                cfg.args.mask_beta,
                cfg.args.max_gram
            )
        ]
        
        self.data_maker = SentPairDataMaker(
            tokenize,
            cfg.model.max_len,
            pipeline=pipeline
        )

def get_ALBERT_datagen(cfg):
    return DataGen(cfg, cfg.args.tok_workers, AlbertWorker, AlbertFeder)
    
#class Config:
    #model: ModelCFG
    #pre: PretrainCFG
    #args: None


#class ModelCFG:
    #embedding: int  = 128
    #hidden: int     = 768
    #hidden_ff: int  = 3072
    #n_layers:int    = 12
    #n_heads: int    = 12
    #max_len: int    = 512
    #n_segments: int = 2
    #vocab_size: int = 30522


#class PretrainCFG:
    #seed: int           = 3431,
    #batch_size: int     = 48,
    #lr: float           = 1e-4,
    #n_epochs: int       = 25,
    #warmup: float       = 0.1,
    #save_steps: int     = 10000,
    #total_steps: int    = 1000000,
    #mixed: bool         = True,
    #prof: bool          = False,
    #prof_start: int     = 2,
    #prof_stop: int      = 22

    
#def main(args):
    #cfg = train.Config.from_json(args.train_cfg)
    #model_cfg = models.Config.from_json(args.model_cfg)

    #set_seeds(cfg.seed)




    #data_iter = SentPairDataLoader(args.data_file,
                                   #cfg.batch_size,
                                   #tokenize,
                                   #model_cfg.max_len,
                                   #pipeline=pipeline)
