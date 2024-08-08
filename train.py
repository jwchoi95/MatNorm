import numpy as np
import torch
import argparse
import logging
import time
import os
import random
from utils import (
    evaluate
)
import pandas as pd
from tqdm import tqdm
from src.matnorm import (
    QueryDataset, 
    CandidateDataset, 
    DictionaryDataset,
    TextPreprocess, 
    RerankNet, 
    MatNorm
)


LOGGER = logging.getLogger()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='MatNorm start')


    parser.add_argument('--model_name_or_path', 
                        default ='m3rg-iitd/matscibert',
                        help='Directory for pretrained model')
    parser.add_argument('--train_dictionary_path', type=str, 
                        default = './datasets/train_dictionary.txt',
                    help='train dictionary path')
    parser.add_argument('--train_dir', type=str, 
                        default = './datasets/processed_traindev',
                    help='training set directory')
    parser.add_argument('--output_dir', type=str, 
                        default = './tmp/matsyn-matbert-mat',
                        help='Directory for output')
    

    parser.add_argument('--max_length', default=25, type=int)


    parser.add_argument('--seed',  type=int, 
                        default=0)
    parser.add_argument('--use_cuda', default =True, action="store_true")
    parser.add_argument('--draft',  action="store_true")
    parser.add_argument('--topk',  type=int, 
                        default=20)
    parser.add_argument('--learning_rate',
                        help='learning rate',
                        default=0.0001, type=float)
    parser.add_argument('--weight_decay',
                        help='weight decay',
                        default=0.01, type=float)
    parser.add_argument('--train_batch_size',
                        help='train batch size',
                        default=16, type=int)
    parser.add_argument('--epoch',
                        help='epoch to train',
                        default=10, type=int)
    parser.add_argument('--initial_sparse_weight',
                        default=0, type=float)
    parser.add_argument('--dense_ratio', type=float,
                        default=0.5)
    parser.add_argument('--save_checkpoint_all', action="store_true")

    args = parser.parse_args()
    return args

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000

    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def load_dictionary(dictionary_path):

    dictionary = DictionaryDataset(
            dictionary_path = dictionary_path
    )
    
    return dictionary.data
    
def load_queries(data_dir, filter_composite, filter_duplicate, filter_cuiless):

    dataset = QueryDataset(
        data_dir=data_dir,
        filter_composite=filter_composite,
        filter_duplicate=filter_duplicate,
        filter_cuiless=filter_cuiless
    )
    
    return dataset.data

def train(args, data_loader, model):
    LOGGER.info("train!")
    
    train_loss = 0
    train_steps = 0
    model.train()
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        model.optimizer.zero_grad()
        batch_x, batch_y = data
        batch_pred = model(batch_x)  
        loss = model.get_loss(batch_pred, batch_y)
        loss.backward()
        model.optimizer.step()
        train_loss += loss.item()
        train_steps += 1

    train_loss /= (train_steps + 1e-9)
    return train_loss
    
def main(args):
    init_logging()
    init_seed(args.seed)
    print(args)
    

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        

    train_dictionary = load_dictionary(dictionary_path=args.train_dictionary_path)
    train_queries = load_dictionary(dictionary_path=args.train_dir)

    if args.draft:
        train_dictionary = train_dictionary[:100]

        args.output_dir = args.output_dir + "_draft"
        

    names_in_train_dictionary = train_dictionary[:,0]
    names_in_train_queries = train_queries[:,0]
    matnorm = MatNorm(
        max_length=args.max_length,
        use_cuda=args.use_cuda,
        initial_sparse_weight=args.initial_sparse_weight
    )
    matnorm.init_sparse_encoder(corpus=names_in_train_dictionary)
    matnorm.load_dense_encoder(
        model_name_or_path=args.model_name_or_path
    )

    model = RerankNet(
        learning_rate=args.learning_rate, 
        weight_decay=args.weight_decay,
        encoder = matnorm.get_dense_encoder(),
        sparse_weight=matnorm.get_sparse_weight(),
        use_cuda=args.use_cuda
    )


    LOGGER.info("Sparse embedding")
    train_query_sparse_embeds = matnorm.embed_sparse(names=names_in_train_queries)
    train_dict_sparse_embeds = matnorm.embed_sparse(names=names_in_train_dictionary)
    train_sparse_score_matrix = matnorm.get_score_matrix(
        query_embeds=train_query_sparse_embeds, 
        dict_embeds=train_dict_sparse_embeds
    )
    train_sparse_candidate_idxs = matnorm.retrieve_candidate(
        score_matrix=train_sparse_score_matrix, 
        topk=args.topk
    )

    # prepare for data loader of train and dev
    train_set = CandidateDataset(
        queries = train_queries, 
        dicts = train_dictionary, 
        tokenizer = matnorm.get_dense_tokenizer(), 
        s_score_matrix=train_sparse_score_matrix,
        s_candidate_idxs=train_sparse_candidate_idxs,
        topk = args.topk, 
        d_ratio=args.dense_ratio,
        max_length=args.max_length
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        shuffle=True,
    )

    start = time.time()
    for epoch in range(1,args.epoch+1):

        LOGGER.info("Epoch {}/{}".format(epoch,args.epoch))
        LOGGER.info("train_set dense embedding for iterative candidate retrieval")
        train_query_dense_embeds = matnorm.embed_dense(names=names_in_train_queries, show_progress=True)
        train_dict_dense_embeds = matnorm.embed_dense(names=names_in_train_dictionary, show_progress=True)
        train_dense_score_matrix = matnorm.get_score_matrix(
            query_embeds=train_query_dense_embeds, 
            dict_embeds=train_dict_dense_embeds
        )
        train_dense_candidate_idxs = matnorm.retrieve_candidate(
            score_matrix=train_dense_score_matrix, 
            topk=args.topk
        )

        train_set.set_dense_candidate_idxs(d_candidate_idxs=train_dense_candidate_idxs)

        # train
        train_loss = train(args, data_loader=train_loader, model=model)
        LOGGER.info('loss/train_per_epoch={}/{}'.format(train_loss,epoch))
        

        if args.save_checkpoint_all:
            checkpoint_dir = os.path.join(args.output_dir, "checkpoint_{}".format(epoch))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            matnorm.save_model(checkpoint_dir)
        

        if epoch == args.epoch:
            matnorm.save_model(args.output_dir)
            
    end = time.time()
    training_time = end-start
    training_hour = int(training_time/60/60)
    training_minute = int(training_time/60 % 60)
    training_second = int(training_time % 60)
    LOGGER.info("Training Time!{} hours {} minutes {} seconds".format(training_hour, training_minute, training_second))
    

        
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
