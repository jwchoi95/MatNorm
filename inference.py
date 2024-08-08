import argparse
import os
import pdb
import pickle
import pandas as pd
from src.matnorm import (
    DictionaryDataset,
    MatNorm,
    TextPreprocess
)
import json
bert= 'bert'
def parse_args():
    parser = argparse.ArgumentParser(description='MatNorm Inference')
    parser.add_argument('--mention', type=str, help='mention to normalize', default = './datasets/pub-mat/mention_raw.xlsx')
    parser.add_argument('--model_name_or_path', help='Directory for model', default = f'./tmp/pubchem/matsyn-scibert-mat_epoch5')
    parser.add_argument('--show_embeddings',  action="store_true")
    parser.add_argument('--show_predictions',  action="store_true", default = 'TRUE')
    parser.add_argument('--dictionary_path', type=str, help='dictionary path', default = './datasets/pub-mat/train_dictionary_avg.txt')
    parser.add_argument('--use_cuda',  action="store_true", default = 'TRUE')
    
    args = parser.parse_args()
    return args
    
def cache_or_load_dictionary(matnorm, model_name_or_path, dictionary_path):
    dictionary_name = os.path.splitext(os.path.basename(args.dictionary_path))[0]
    
    cached_dictionary_path = os.path.join(
        './tmp',
        f"cached_{model_name_or_path.split('/')[-1]}_{dictionary_name}.pk"
    )

    # If exist, load the cached dictionary
    if os.path.exists(cached_dictionary_path):
        with open(cached_dictionary_path, 'rb') as fin:
            cached_dictionary = pickle.load(fin)
        print("Loaded dictionary from cached file {}".format(cached_dictionary_path))

        dictionary, dict_sparse_embeds, dict_dense_embeds = (
            cached_dictionary['dictionary'],
            cached_dictionary['dict_sparse_embeds'],
            cached_dictionary['dict_dense_embeds'],
        )

    else:
        dictionary = DictionaryDataset(dictionary_path = dictionary_path).data
        dictionary_names = dictionary[:,0]
        dict_sparse_embeds = matnorm.embed_sparse(names=dictionary_names, show_progress=True)
        dict_dense_embeds = matnorm.embed_dense(names=dictionary_names, show_progress=True)
        cached_dictionary = {
            'dictionary': dictionary,
            'dict_sparse_embeds' : dict_sparse_embeds,
            'dict_dense_embeds' : dict_dense_embeds
        }

        if not os.path.exists('./tmp'):
            os.mkdir('./tmp')
        with open(cached_dictionary_path, 'wb') as fin:
            pickle.dump(cached_dictionary, fin)
        print("Saving dictionary into cached file {}".format(cached_dictionary_path))

    return dictionary, dict_sparse_embeds, dict_dense_embeds

def main(args):
    matnorm = MatNorm(
        max_length=25,
        use_cuda=args.use_cuda
    )
    
    matnorm.load_model(model_name_or_path=args.model_name_or_path)
    temp = pd.read_excel(args.mention)
    result = []
    temp_df = []
    for i in range(len(temp)):
        mention = temp.iloc[i][0]
        mention = TextPreprocess().run(mention)
        mention_sparse_embeds = matnorm.embed_sparse(names=[mention])
        mention_dense_embeds = matnorm.embed_dense(names=[mention])
        output = {}
        output['idx']= i
        output['mention']= mention
        output['type']= temp.iloc[i][1]


        if args.show_embeddings: #do not use
            output = {
                'mention': mention,
                'mention_sparse_embeds': mention_sparse_embeds.squeeze(0),
                'mention_dense_embeds': mention_dense_embeds.squeeze(0)
            }

        if args.show_predictions:
            if args.dictionary_path == None:
                print('insert the dictionary path')
                return
            dictionary, dict_sparse_embeds, dict_dense_embeds = cache_or_load_dictionary(matnorm, args.model_name_or_path, args.dictionary_path)
            sparse_score_matrix = matnorm.get_score_matrix(
                query_embeds=mention_sparse_embeds,
                dict_embeds=dict_sparse_embeds
            )
            dense_score_matrix = matnorm.get_score_matrix(
                query_embeds=mention_dense_embeds,
                dict_embeds=dict_dense_embeds
            )
            sparse_weight = matnorm.get_sparse_weight().item()
            hybrid_score_matrix = sparse_weight * sparse_score_matrix + dense_score_matrix
            hybrid_score_idxs, hybrid_candidate_idxs = matnorm.retrieve_candidate_with_score(
                score_matrix = hybrid_score_matrix, 
                topk = 10
            )
            predictions = dictionary[hybrid_candidate_idxs].squeeze(0)
            pred = []

            for j, prediction in enumerate(predictions):
                predicted_name = prediction[0]
                predicted_id = prediction[1]
                score = float(hybrid_score_idxs[0][j])  
                pred.append({
                    'name': predicted_name,
                    'id': predicted_id,
                    'score': score  
                })
                temp_df.append([output['idx'], output['mention'], output['type'],j, predicted_name,predicted_id,score ])
            output['predictions'] = pred
        result.append(output)
    file_path = 'output/inference.json'
    temp_df = pd.DataFrame(temp_df)
    temp_df.columns = ['idx','mention','type','rank','predicted_name','predicted_id','score']
    temp_df.to_excel(f'output/inference.xlsx', index= False)
    with open(file_path, 'w') as json_file:
        json.dump(result, json_file, indent=4)
if __name__ == '__main__':
    args = parse_args()
    main(args)