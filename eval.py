import argparse
import logging
import os
import json
from tqdm import tqdm
from utils import (
    evaluate
)
from src.matnorm import (
    DictionaryDataset,
    QueryDataset,
    MatNorm
)
LOGGER = logging.getLogger()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='MatSyn evaluation')
    parser.add_argument('--model_name_or_path', help='Directory for model', default = './tmp/matsyn-matscibert-mat_0')
    parser.add_argument('--dictionary_path', type=str, help='dictionary path', default = './datasets/train_dictionary_avg.txt')
    parser.add_argument('--data_dir', type=str, help='data set to evaluate', default = './datasets/train_dictionary.txt')
    parser.add_argument('--use_cuda',  action="store_true")
    parser.add_argument('--topk',  type=int, default=20)
    parser.add_argument('--score_mode',  type=str, default='hybrid', help='hybrid/dense/sparse')
    parser.add_argument('--output_dir', type=str, default='./output/', help='Directory for output')
    parser.add_argument('--filter_composite', action="store_true", help="filter out composite mention queries")
    parser.add_argument('--filter_duplicate', action="store_true", help="filter out duplicate queries")
    parser.add_argument('--save_predictions', action="store_true", default = 'TRUE', help="whether to save predictions")
    parser.add_argument('--max_length', default=25, type=int)
    
    args = parser.parse_args()
    return args
    
def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

def load_dictionary(dictionary_path): 
    dictionary = DictionaryDataset(
        dictionary_path = dictionary_path
    )
    return dictionary.data

def load_queries(data_dir, filter_composite, filter_duplicate):
    dataset = QueryDataset(
        data_dir=data_dir,
        filter_composite=filter_composite,
        filter_duplicate=filter_duplicate
    )
    return dataset.data
                
def main(args):
    init_logging()
    print(args)


    eval_dictionary = load_dictionary(dictionary_path=args.dictionary_path)
    eval_queries = load_dictionary(dictionary_path=args.data_dir)
 
    matnorm = MatNorm(
        max_length=args.max_length,
        use_cuda=args.use_cuda
    )
    matnorm.load_model(
        model_name_or_path=args.model_name_or_path,
    )
    
    result_evalset = evaluate(
        biosyn=matnorm,
        eval_dictionary=eval_dictionary,
        eval_queries=eval_queries,
        topk=args.topk,
    )
    
    LOGGER.info("acc@1={}".format(result_evalset['acc1']))
    LOGGER.info("acc@5={}".format(result_evalset['acc5']))
    LOGGER.info("acc@10={}".format(result_evalset['acc10']))
    LOGGER.info("acc@20={}".format(result_evalset['acc20']))
    
    if args.save_predictions:
        output_file = os.path.join(args.output_dir,"predictions_eval.json")
        with open(output_file, 'w') as f:
            json.dump(result_evalset, f, indent=2)

if __name__ == '__main__':
    args = parse_args()
    main(args)
