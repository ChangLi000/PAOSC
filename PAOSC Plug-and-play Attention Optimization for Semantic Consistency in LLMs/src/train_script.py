import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from train import trainer
from transformers.utils import logging
from transformers.models.opt.configuration_opt import OPTConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import AutoTokenizer, OPTModel, LlamaTokenizer, LlamaModel, PreTrainedTokenizerFast, Qwen2Tokenizer, Qwen3Model, Qwen2Model, AutoModel, GPT2Tokenizer, GPT2Model
from data import HTTP_RLDataLoader, SpamDataLoader, AGnewsDataLoader, HttpPramaDataLoader
from model import TransformerDiscriminator, TransformerGenerator, ClassificationHead


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.set_verbosity_error()

def main():
    parser = argparse.ArgumentParser(description="arguments")
    # 定义参数
    parser.add_argument("--modelname", type=str, required=True, help="the name of model")
    parser.add_argument("--modelpath", type=str, required=True, help="model file path")
    parser.add_argument("--datapath", type=str, required=True, help="datsets path")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--topk", type=int, default=3, help="hyperparameter")
    parser.add_argument("--max_steps", type=int, default=100, help="max training steps")
    parser.add_argument("--epoch", type=int, default=3, help="total training epoch")
    parser.add_argument("--lambda_mask", type=float, default=0.3, help="prob of geedy select")
    parser.add_argument("--savepath", type=str, default=".", help="for saving model weights file")

    # parser parameter
    args = parser.parse_args()

    model_path=args.modelpath
    if "OPT" in args.modelname:
        config = OPTConfig()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = OPTModel.from_pretrained(model_path,device_map="auto",weights_only=True)
    elif "Llama2-7B" in args.modelname:
        config = LlamaConfig()
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = LlamaModel.from_pretrained(model_path,device_map="auto")
    elif "Llama3-8B" in args.modelname:
        config = LlamaConfig()
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        model = LlamaModel.from_pretrained(model_path,device_map="auto")
    elif "Qwen3" in args.modelname:
        config = Qwen3Config()
        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        model = Qwen3Model.from_pretrained(model_path,device_map="auto")
    elif "Qwen2" in args.modelname:
        config = Qwen2Config()
        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        model = Qwen2Model.from_pretrained(model_path,device_map="auto")
    elif "GLM" in args.modelname:
        config = {"hidden_size":4096}
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
    elif "GPT2" in args.modelname:
        config = GPT2Config()
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2Model.from_pretrained(model_path)

    if not tokenizer.pad_token:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer),mean_resizing=False)
    
    # Initialize G D C model ... ...
    D = TransformerDiscriminator(hidden_dim=config.hidden_size) 
    G = TransformerGenerator(embed_dim=config.hidden_size)
    C = ClassificationHead(embed_dim=config.hidden_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G.to(device)
    D.to(device)
    C.to(device)
    
    lr = args.lr
    lr_d = args.lr
    criterion_D = nn.CrossEntropyLoss(reduction='none')    
    optimizer_D = optim.Adam(D.parameters(), lr=lr_d)
    optimizer_GC = torch.optim.Adam(
        list(G.parameters()) + list(C.parameters()), lr=lr
    )

    # Loading datasets ... ...

    if "HTTP_RL" in args.datapath:
        train_loader, test_loader = HTTP_RLDataLoader(tokenizer, data_path = args.datapath)
    elif "news" in args.datapath:
        train_loader, test_loader = AGnewsDataLoader(tokenizer, data_path = args.datapath)
    elif "spam" in args.datapath:
        train_loader, test_loader = SpamDataLoader(tokenizer, data_path = args.datapath)
    elif "payload" in args.datapath:
        train_loader, test_loader = HttpPramaDataLoader(tokenizer, data_path = args.datapath)
    else:
        raise ValueError("Error: Cannot find dataset, please check if the file name is correct!")

    # Training ... ...
    trainer(train_loader, G, D, C, model, tokenizer, optimizer_GC, optimizer_D, criterion_D, device, save_path=".",topk=args.topk, max_steps=args.max_steps, lambda_mask=args.lambda_mask,epoch=args.epoch)
    
    

if __name__ == "__main__":
    main()
