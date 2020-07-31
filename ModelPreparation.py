from transformers import *
import torch.nn as nn

MODELS = {'Bert' : {'Model' : BertModel, 'Encoder' : BertTokenizer, 'Param' : 'bert-base-uncased'},
          'GPT' : {'Model' : OpenAIGPTModel, 'Encoder' : OpenAIGPTTokenizer, 'Param' : 'openai-gpt'},
          'GPT2' : {'Model' : GPT2Model, 'Encoder' : GPT2Tokenizer, 'Param' : 'gpt2'},
          'CTRL' : {'Model' : CTRLModel, 'Encoder' : CTRLTokenizer, 'Param' : 'ctrl'},
          'TransformerXL' : {'Model' : TransfoXLModel, 'Encoder' : TransfoXLTokenizer, 'Param' : 'transfo-xl-wt103'},
          'XLNet' : {'Model' : XLNetModel, 'Encoder' : XLNetTokenizer, 'Param' : 'xlnet-base-cased'},
          'DistillBert' : {'Model' : DistilBertModel, 'Encoder' : DistilBertTokenizer, 'Param' : 'distilbert-base-cased'},
          'Roberta' : {'Model' : RobertaModel, 'Encoder' : RobertaTokenizer, 'Param' : 'roberta-base'}
         }

class NLPModel(nn.Module):
    
    def __init__(self, model):
        super(NLPModel, self).__init__()
        raw_tr = MODELS[model]['Model']
        pretrain = MODELS[model]['Param']
        self.tr = raw_tr.from_pretrained(pretrain)
        self.hidden_size = self.tr.config.hidden_size
        self.cls = nn.Linear(self.hidden_size, 1)
        self.dropout = nn.Dropout(0.4)
    
    def forward(self, x, mask):
        cls_emb = self.tr(x, mask)[1]
        prediction = self.cls(self.dropout(cls_emb))
        return prediction

def build_encoder(model):
    '''
    Building the encoder
    
    Args:
        model(str): the predefined model name

    Returns:
        The corresponding encoder
    '''
    raw_coder = MODELS[model]['Encoder']
    pretrain = MODELS[model]['Param']
    encoder = raw_coder.from_pretrained(pretrain, do_lower_case = True)
    
    return encoder

def build_model(model):
    '''
    Building the model

    Args:
        model(str): the predefined model name

    Returns:
        The corresponding model
    '''
    model = NLPModel(model)
    
    return model