from transformers import EsmTokenizer, EsmModel, AutoTokenizer, EsmForMaskedLM
from tqdm import tqdm
import numpy as np
import torch 


class PLMEmbedder():

    def __init__(self, model=None, tokenizer=None, checkpoint:str=None):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.from_pretrained(checkpoint)
        self.model.to(self.device) # Move model to GPU.
        self.model.eval() # Set model to evaluation model.
        self.tokenizer = tokenizer.from_pretrained(checkpoint, do_lower_case=False, legacy=True, clean_up_tokenization_spaces=True)


    def embed_batch(self, seqs:list) -> torch.FloatTensor:

        # Should contain input_ids and attention_mask. Make sure everything's on the GPU. 
        # The tokenizer defaults mean that add_special_tokens=True and padding=True is equivalent to padding='longest'
        inputs = {k:torch.tensor(v).to(self.device) for k, v in self.tokenizer(seqs, padding=True).items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs
   
    def __call__(self, seqs:list, max_aa_per_batch:int=1000):

        seqs = self._preprocess(seqs)

        embs = list()
        aa_count = 0
        batch_seqs = list()
        for seq in tqdm(seqs, desc='PLMEmbedder.__call__'):

            batch_seqs.append(seq)
            aa_count += len(seq)

            if aa_count > max_aa_per_batch:
                outputs = self.embed_batch(batch_seqs)
                embs += self._postprocess(outputs, seqs=batch_seqs)
                batch_seqs = list()
                aa_count = 0

        # Handles the case in which the minimum batch size is not reached.
        if aa_count > 0:
            outputs = self.embed_batch(batch_seqs)
            embs += self._postprocess(outputs, seqs=batch_seqs)

        embs = torch.cat([torch.unsqueeze(emb, 0) for emb in embs]).float()
        return embs.numpy()



class ESMEmbedder(PLMEmbedder):
    checkpoints ={'3b':'facebook/esm2_t36_3B_UR50D', '650m':'facebook/esm2_t33_650M_UR50D'}

    @staticmethod
    def _pooler_gap(emb:torch.FloatTensor, seq:str) -> torch.FloatTensor:
        emb = emb[1:len(seq) + 1] # First remove the CLS token from the mean-pool, as well as any padding... 
        emb = emb.mean(dim=0)
        return emb 

    @staticmethod
    def _pooler_cls(emb:torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
        return emb[0] # Extract the CLS token, which is the first element of the sequence. 

    def __init__(self, model_size:str='650m', pooler:str='gap'):

        models = {'gap':EsmModel, 'log':EsmForMaskedLM, 'cls':EsmModel}
        poolers = {'gap':ESMEmbedder._pooler_gap, 'cls':ESMEmbedder._pooler_cls} 
        checkpoint = ESMEmbedder.checkpoints.get(model_size)

        super(ESMEmbedder, self).__init__(model=models[pooler], tokenizer=AutoTokenizer, checkpoint=checkpoint)
        self.pooler = poolers.get(pooler, None)

    def _preprocess(self, seqs:list):
        # Based on the example Jupyter notebook, it seems as though sequences require no real pre-processing for the ESM model.
        return seqs 

    def _postprocess(self, outputs:torch.FloatTensor, seqs:list=None):
        ''''''
        # Transferring the outputs to CPU and reassigning should free up space on the GPU. 
        # https://discuss.pytorch.org/t/is-the-cuda-operation-performed-in-place/84961/6 
        if self.pooler is not None:
            outputs = outputs.last_hidden_state.cpu() # if (self.model_name == 'pt5') else outputs.pooler_output
            outputs = [self.pooler(emb, seq) for emb, seq in zip(outputs, seqs)]
        else: 
            raise Exception('TODO')
        return outputs       



class ProtT5Embedder(PLMEmbedder):

    checkpoint = 'Rostlab/prot_t5_xl_half_uniref50-enc'

    def __init__(self):

        super(ProtT5Embedder, self).__init__(model=T5EncoderModel, tokenizer=T5Tokenizer, checkpoint=ProtT5Embedder.checkpoint)

    def _preprocess(self, seqs:list) -> list:
        ''''''
        seqs = [seq.replace('*', '') for seq in seqs]
        seqs = [seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X').replace('B', '') for seq in seqs] # Replace rare amino acids with X token.
        seqs = [' '.join(list(seq)) for seq in seqs] # Characters in the sequence need to be space-separated, apparently. 
        return seqs  

    def _postprocess(self, outputs:torch.FloatTensor, seqs:list=None) -> List[torch.FloatTensor]:
        ''''''
        seqs = [''.join(seq.split()) for seq in seqs] # Remove the added whitespace so length is correct. 

        outputs = outputs.last_hidden_state.cpu()
        outputs = [emb[:len(seq)] for emb, seq in zip(outputs, seqs)]
        outputs = [emb.mean(dim=0) for emb in outputs] # Take the average over the sequence length. 
        return outputs  