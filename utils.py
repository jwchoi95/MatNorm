import numpy as np
from tqdm import tqdm

def check_k(queries):
    return len(queries[0]['mentions'][0]['candidates'])

def evaluate_topk_acc(data):
    """
    evaluate acc@1~acc@k
    """
    queries = data['queries']
    k = check_k(queries)

    for i in range(0, k):
        hit = 0
        for query in queries:
            mentions = query['mentions']
            mention_hit = 0
            for mention in mentions:
                candidates = mention['candidates'][:i+1] # to get acc@(i+1)
                mention_hit += np.any([candidate['label'] for candidate in candidates])

            if mention_hit == len(mentions):
                hit +=1
        
        data['acc{}'.format(i+1)] = hit/len(queries)

    return data

def check_label(predicted_cui, golden_cui):
    """
    Some composite annotation didn't consider orders
    So, set label '1' if any cui is matched within composite cui (or single cui)
    Otherwise, set label '0'
    """
    return int(len(set(predicted_cui.split("|")).intersection(set(golden_cui.split("|"))))>0)

def predict_topk(matnorm, eval_dictionary, eval_queries, topk, score_mode='hybrid'):

    encoder = matnorm.get_dense_encoder()
    tokenizer = matnorm.get_dense_tokenizer()
    sparse_encoder = matnorm.get_sparse_encoder()
    sparse_weight = matnorm.get_sparse_weight().item() # must be scalar value

    dict_sparse_embeds = matnorm.embed_sparse(names=eval_dictionary[:,0], show_progress=True)
    dict_dense_embeds = matnorm.embed_dense(names=eval_dictionary[:,0], show_progress=True)
    
    queries = []
    for eval_query in tqdm(eval_queries, total=len(eval_queries)):
        mentions = eval_query[0].replace("+","|").split("|")
        golden_cui = eval_query[1].replace("+","|")
        
        dict_mentions = []
        for mention in mentions:
            mention_sparse_embeds = matnorm.embed_sparse(names=np.array([mention]))
            mention_dense_embeds = matnorm.embed_dense(names=np.array([mention]))
            sparse_score_matrix = matnorm.get_score_matrix(
                query_embeds=mention_sparse_embeds, 
                dict_embeds=dict_sparse_embeds
            )
            dense_score_matrix = matnorm.get_score_matrix(
                query_embeds=mention_dense_embeds, 
                dict_embeds=dict_dense_embeds
            )
            if score_mode == 'hybrid':
                score_matrix = sparse_weight * sparse_score_matrix + dense_score_matrix
            elif score_mode == 'dense':
                score_matrix = dense_score_matrix
            elif score_mode == 'sparse':
                score_matrix = sparse_score_matrix
            else:
                raise NotImplementedError()
            candidate_idxs = matnorm.retrieve_candidate(
                score_matrix = score_matrix, 
                topk = topk
            )
            np_candidates = eval_dictionary[candidate_idxs].squeeze()
            dict_candidates = []
            for np_candidate in np_candidates:
                dict_candidates.append({
                    'name':np_candidate[0],
                    'cui':np_candidate[1],
                    'label':check_label(np_candidate[1],golden_cui)
                })
            dict_mentions.append({
                'mention':mention,
                'golden_cui':golden_cui, 
                'candidates':dict_candidates
            })
        queries.append({
            'mentions':dict_mentions
        })
    
    result = {
        'queries':queries
    }

    return result

def evaluate(matnorm, eval_dictionary, eval_queries, topk, score_mode='hybrid'):

    result = predict_topk(matnorm,eval_dictionary,eval_queries, topk, score_mode)
    result = evaluate_topk_acc(result)
    
    return result