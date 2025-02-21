import os
import sys
import json
import pickle as pkl
import logging
import time
import copy
import random
from tqdm import tqdm
import re
import pdb
import string
from collections import Counter
from omegaconf import ListConfig
import multiprocessing

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
try:
    from pyserini.search.lucene import LuceneSearcher
except:
    logging.warning("Failed to import pyserini! Please install it from https://github.com/castorini/pyserini/tree/master.")


import contriever.src.index
import contriever.src.contriever
import contriever.src.utils
import contriever.src.slurm
from contriever.src.evaluation import calculate_matches
import contriever.src.normalize_text

from src.data import load_eval_data
from src.index import Indexer, get_index_dir_and_passage_paths, get_index_passages_and_id_map, get_bm25_index_dir
from src.decontamination import check_below_lexical_overlap_threshold
try:
    from utils.deduplication import remove_duplicates_with_minhash, multiprocess_deduplication
except:
    print("Cannot import from utils")

os.environ["TOKENIZERS_PARALLELISM"] = "true"


device = 'cuda' if torch.cuda.is_available()  else 'cpu'


def embed_queries(args, queries, model, tokenizer, model_name_or_path):
    if "sentence-transformers" in model_name_or_path:
        all_question = []
        for k, q in enumerate(queries):
            if args.lowercase:
                q = q.lower()
            if args.normalize_text:
                q = contriever.src.normalize_text.normalize(q)
            all_question.append(q)
        
        embeddings = model.encode(all_question, batch_size=min(128, args.per_gpu_batch_size))  # sentence-transformer has extra memory overhead and can only support a smaller batch size
    
    else:
        model.eval()
        embeddings, batch_question = [], []
        with torch.no_grad():

            for k, q in tqdm(enumerate(queries)):
                if args.lowercase:
                    q = q.lower()
                if args.normalize_text:
                    q = contriever.src.normalize_text.normalize(q)
                batch_question.append(q)

                if len(batch_question) == args.per_gpu_batch_size or k == len(queries) - 1:

                    encoded_batch = tokenizer.batch_encode_plus(
                        batch_question,
                        return_tensors="pt",
                        max_length=args.question_maxlength,
                        padding=True,
                        truncation=True,
                    )

                    encoded_batch = {k: v.to(device) for k, v in encoded_batch.items()}
                    output = model(**encoded_batch)
                    if "contriever" not in model_name_or_path:
                        output = output.last_hidden_state[:, 0, :]
                    embeddings.append(output.cpu())

                    batch_question = []

        embeddings = torch.cat(embeddings, dim=0).numpy()
    
    print(f"Questions embeddings shape: {embeddings.shape}")

    if args.get('cache_query_embedding', False):
        with open(args.query_embedding_save_path, 'wb') as fout:
            pkl.dump(embeddings, fout)

    return embeddings



def validate(data, workers_num):
    match_stats = calculate_matches(data, workers_num)
    top_k_hits = match_stats.top_k_hits

    print("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(data) for v in top_k_hits]
    message = ""
    for k in [5, 10, 20, 100]:
        if k <= len(top_k_hits):
            message += f"R@{k}: {top_k_hits[k-1]} "
    print(message)
    return match_stats.questions_doc_hits


def add_passages(data, passages, top_passages_and_scores, valid_query_idx, domain=None):
    # add passages to original data
    assert len(valid_query_idx) == len(top_passages_and_scores)
    idx = 0
    for i, d in enumerate(data):
        if i in valid_query_idx:
            results_and_scores = top_passages_and_scores[idx]
            docs = [passages[doc_id] for doc_id in results_and_scores[0]]
            next_docs = [passages[str(int(doc_id)+1)] if int(doc_id)+1 < len(passages) else passages[doc_id] for doc_id in results_and_scores[0]]
            scores = [str(score) for score in results_and_scores[1]]
            ctxs_num = len(docs)
            d["ctxs"] = [
                {
                    "id": results_and_scores[0][c],
                    "source": domain,
                    "origin_id": docs[c]["origin_id"] if 'origin_id' in docs[c] else None,
                    # "retrieval title": docs[c]["title"],
                    "retrieval text": docs[c]["text"],
                    "retrieval next text": next_docs[c]["text"],
                    "retrieval score": scores[c],
                }
                for c in range(ctxs_num)
            ]
            idx += 1
        else:
            d["ctxs"] = [None]


def add_hasanswer(data, hasanswer):
    # add hasanswer to data
    for i, ex in enumerate(data):
        for k, d in enumerate(ex["ctxs"]):
            d["hasanswer"] = hasanswer[i][k]


def get_search_output_path(cfg, index_shard_ids):
    eval_args = cfg.evaluation
    shards_postfix = '_'.join([str(shard_id) for shard_id in index_shard_ids])
    output_dir = os.path.join(eval_args.eval_output_dir, shards_postfix)
    output_path = os.path.join(output_dir, os.path.basename(eval_args.data.eval_data).replace('.jsonl', '_retrieved_results.jsonl'))
    return output_path


def get_merged_search_output_path(cfg):
    index_args = cfg.datastore.index
    eval_args = cfg.evaluation

    if isinstance(index_args.index_shard_ids[0], ListConfig):
        print(f"Multi-index mode: building {len(index_args.index_shard_ids)} index for {index_args.index_shard_ids} sequentially...")
        index_shard_ids_list = index_args.index_shard_ids
    else:
        print(f"Single-index mode: building a single index over {index_args.index_shard_ids} shards...")
        index_shard_ids_list = [index_args.index_shard_ids]
    
    merged_postfix = ''
    for index_shard_ids in sorted(index_shard_ids_list, key=lambda x: int(x[0])):
        shards_postfix = '_'.join([str(shard_id) for shard_id in index_shard_ids])
        merged_postfix += '-' + shards_postfix
    merged_postfix = merged_postfix.strip('-')

    output_dir = os.path.join(eval_args.eval_output_dir, merged_postfix)
    output_path = os.path.join(output_dir, os.path.basename(eval_args.data.eval_data).replace('.jsonl', '_retrieved_results.jsonl'))
    return output_path


def get_merged_subsampled_search_output_path(cfg):
    index_args = cfg.datastore.index
    eval_args = cfg.evaluation

    if isinstance(index_args.index_shard_ids[0], ListConfig):
        print(f"Multi-index mode: building {len(index_args.index_shard_ids)} index for {index_args.index_shard_ids} sequentially...")
        index_shard_ids_list = index_args.index_shard_ids
    else:
        print(f"Single-index mode: building a single index over {index_args.index_shard_ids} shards...")
        index_shard_ids_list = [index_args.index_shard_ids]
    
    merged_postfix = ''
    for index_shard_ids in sorted(index_shard_ids_list, key=lambda x: int(x[0])):
        shards_postfix = '_'.join([str(shard_id) for shard_id in index_shard_ids])
        merged_postfix += '-' + shards_postfix
    merged_postfix = merged_postfix.strip('-')

    if cfg.evaluation.search.get('topk_subsample_p', None):
        seed = cfg.evaluation.search.get('subsample_seed', 1000)
        output_dir = os.path.join(eval_args.eval_output_dir, os.path.join(f'subsampled_{cfg.evaluation.search.topk_subsample_p}_seed_{seed}', merged_postfix))
    else:
        output_dir = os.path.join(eval_args.eval_output_dir, merged_postfix)

    output_path = os.path.join(output_dir, os.path.basename(eval_args.data.eval_data).replace('.jsonl', '_retrieved_results.jsonl'))
    return output_path


def search_dense_topk(cfg):
    index_args = cfg.datastore.index
    eval_args = cfg.evaluation
    ds_domain = cfg.datastore.domain

    if isinstance(index_args.index_shard_ids[0], ListConfig):
        print(f"Multi-index mode: building {len(index_args.index_shard_ids)} index for {index_args.index_shard_ids} sequentially...")
        index_shard_ids_list = index_args.index_shard_ids
    else:
        print(f"Single-index mode: building a single index over {index_args.index_shard_ids} shards...")
        index_shard_ids_list = [index_args.index_shard_ids]

    all_exist = True
    for index_shard_ids in index_shard_ids_list:
        # check if all search results exist
        output_path = get_search_output_path(cfg, index_shard_ids)
        all_exist = all_exist and os.path.exists(output_path)
    
    if all_exist and not eval_args.search.overwrite:
        logging.info(f'All search results for {index_args.index_shard_ids} exist, skipping searching.')
    
    else:
        # load model and evaluation data
        logging.info(f"Loading model from: {cfg.model.datastore_encoder}")
        model_name_or_path = cfg.model.query_encoder
        tokenizer_name_or_path = cfg.model.query_tokenizer
        if "contriever" in model_name_or_path:
            query_encoder, query_tokenizer, _ = contriever.src.contriever.load_retriever(model_name_or_path)
        elif "dragon" in model_name_or_path:
            query_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
            query_encoder = AutoModel.from_pretrained(model_name_or_path)
        elif "sentence-transformers" in model_name_or_path:
            query_tokenizer = None
            query_encoder = SentenceTransformer(model_name_or_path)
        else:
            print(f"{model_name_or_path} is not supported!")
            raise AttributeError

        query_encoder.eval()
        query_encoder = query_encoder.to(device)
        if not index_args.no_fp16:
            query_encoder = query_encoder.half()
        
        # load eval data
        data = load_eval_data(cfg)
        
        # if eval_args.data.num_eval_samples is not None:
        #     random.seed(eval_args.data.seed)
        #     data = random.sample(data, int(eval_args.data.num_eval_samples))

        queries = []
        valid_query_idx = []
        for idx, ex in enumerate(data):
            raw_query = ex["raw_query"]
            if raw_query:
                queries.append(ex["raw_query"])
                valid_query_idx.append(idx)
        
        logging.info(f"Searching for {len(queries)} queries from {len(data)} total evaluation samples...")
        if eval_args.search.get('cache_query_embedding', False) and os.path.exists(eval_args.search.get('query_embedding_save_path', "")):
            logging.info(f"Loading query embeddings from {eval_args.search.query_embedding_save_path}")
            with open(eval_args.search.query_embedding_save_path, 'rb') as fin:
                questions_embedding = pkl.load(fin)
        else:
            questions_embedding = embed_queries(eval_args.search, queries, query_encoder, query_tokenizer, model_name_or_path)
        if eval_args.search.get('cache_query_embedding_only', False):
            return

        # load index
        for index_shard_ids in index_shard_ids_list:
            output_path = get_search_output_path(cfg, index_shard_ids)
            
            if os.path.exists(output_path) and not eval_args.search.overwrite:
                logging.info(f'{output_path} exists, skipping searching.')

            else:
                copied_data = copy.deepcopy(data)

                index_dir, _ = get_index_dir_and_passage_paths(cfg, index_shard_ids)
                index = Indexer(index_args.projection_size, index_args.n_subquantizers, index_args.n_bits)
                index.deserialize_from(index_dir)

                # load passages and id mapping corresponding to the index
                passages, passage_id_map = get_index_passages_and_id_map(cfg, index_shard_ids)
                assert len(passages) == index.index.ntotal, f"number of documents {len(passages)} and number of embeddings {index.index.ntotal} mismatch"

                # get top k results
                start_time_retrieval = time.time()

                top_ids_and_scores = index.search_knn(questions_embedding, eval_args.search.n_docs)
                logging.info(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

                # todo: double check valid_query_idx
                logging.info(f"Adding documents to eval data...")
                add_passages(copied_data, passage_id_map, top_ids_and_scores, valid_query_idx, domain=ds_domain)
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                safe_write_jsonl(copied_data, output_path)
    
    if cfg.evaluation.search.get('merge_multi_source_results', False) and cfg.evaluation.search.get("topk_subsample_p", None):
        post_hoc_merge_topk_multi_domain(cfg)

    elif cfg.evaluation.search.get('merge_multi_index_results', True):
        post_hoc_merge_topk(cfg)
    

def post_hoc_merge_topk(cfg):
    """
    Post hoc merge the searched results obtained by multiple indices.
    """
    index_args = cfg.datastore.index
    output_path = get_merged_search_output_path(cfg)
    if os.path.exists(output_path) and not cfg.evaluation.search.overwrite:
        print(f"The merged path exists, skipping...\n{output_path}")
        return

    if isinstance(index_args.index_shard_ids[0], ListConfig) and len(index_args.index_shard_ids) > 1:
        print(f"Multi-index mode: building {len(index_args.index_shard_ids)} index for {index_args.index_shard_ids} sequentially...")
        index_shard_ids_list = index_args.index_shard_ids
    else:
        print(f"Single-index mode: no need to merge")
        return
    
    merged_data = []
    for i, index_shard_ids in enumerate(index_shard_ids_list):
        path_to_merge = get_search_output_path(cfg, index_shard_ids)
        print(f"Adding {path_to_merge}")
        
        data_to_merge = []
        with open(path_to_merge, 'r') as file:
            idx = 0
            for line in file:
                try:
                    _ex = json.loads(line)
                except:
                    print(f"Line read error when reading {path_to_merge}")
                    continue
                
                if not _ex['ctxs'] or _ex['ctxs'][0] is None:
                    assert idx == 0  # the first example in ppl eval does not have query
                    ctxs = []
                else:
                    ctxs = _ex['ctxs']

                _ex['ctxs'] = ctxs
                data_to_merge.append(_ex)
        
        if i == 0:
            merged_data = data_to_merge
        
        else:
            for id_, (_, _ex) in enumerate(zip(merged_data, data_to_merge)):
                
                assert merged_data[id_]['raw_query'] == _ex['raw_query']
                merged_data[id_]['ctxs'].extend(_ex['ctxs'])

                # Rerank based on score and only keep the top n_docs to avoid memory explosion
                if merged_data[id_]['ctxs'] and merged_data[id_]['ctxs'][0] is not None:
                    merged_data[id_]['ctxs'] = sorted(merged_data[id_]['ctxs'], key=lambda x: float(x['retrieval score']), reverse=True)
                    merged_data[id_]['ctxs'] = merged_data[id_]['ctxs'][:cfg.evaluation.search.n_docs]
                    # make sure we still have n_docs documents
                    assert len(merged_data[id_]['ctxs']) == cfg.evaluation.search.n_docs
                else:
                    assert id_ == 0 or id_ == 983  # the 983rd example in RPJ has an empty query 

    # Write merged and reranked data to a new JSONL file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    safe_write_jsonl(merged_data, output_path)



def subsample_by_coin_flip(items, probability):
    subsampled_list = []
    for item in items:
        # Perform a coin flip with probability p of being True (keep the item)
        if random.random() < probability:
            subsampled_list.append(item)
    return subsampled_list


def post_hoc_merge_topk_multi_domain(cfg):
    """
    Post hoc merge the searched results obtained by multiple domains/sources. Each source may have multiple indices.
    Required inputs:
    1. A list of searched results to be merged defined by `cfg.evaluation.search.paths_to_merge`
    2. A path to save merged results defined by `cfg.evaluation.search.merged_path`
    """
    txt_file_with_paths_to_merge = cfg.evaluation.search.paths_to_merge
    base_merged_path = cfg.evaluation.search.merged_path
    merged_path = os.path.join(os.path.dirname(base_merged_path), os.path.basename(base_merged_path).strip('dedup_'))

    if not os.path.exists(base_merged_path) or not cfg.evaluation.search.use_saved_dedup_data:

        if cfg.evaluation.search.get('topk_subsample_p', 1) < 1:
            # Set a random seed for subsampling
            seed = cfg.evaluation.search.get('subsample_seed', 1000)
            random.seed(seed)

        if not os.path.exists(merged_path):
            # Read .txt file containing all files of searched results to merge
            paths_to_merge = []
            with open(txt_file_with_paths_to_merge, 'r') as file:
                for line in file:
                    path = line.strip()
                    paths_to_merge.append(path)
                    assert os.path.exists(path), f'{path}'
            print(f"Merging files:\n{paths_to_merge}")

            datastore_domain_pattern = re.compile(r'/([^/]+)_datastore')

            merged_data = []
            for domain_idx, path_to_merge in tqdm(enumerate(paths_to_merge)):
                print(f"Adding {path_to_merge}")

                data_to_merge = []
                with open(path_to_merge, 'r') as file:
                    # annotate datastore domain for analysis
                    matches = datastore_domain_pattern.findall(path_to_merge)
                    ds_domain = matches[0] if matches else None

                    idx = 0
                    for line in file:
                        try:
                            _ex = json.loads(line)
                        except:
                            print(f"Line read error when reading {path_to_merge}")
                            raise AttributeError
                        
                        if not _ex['ctxs'] or _ex['ctxs'][0] is None:
                            assert idx == 0  # the first example in ppl eval does not have query
                            ctxs = []
                        else:
                            if not "source" in _ex['ctxs'][0].keys() or not _ex['ctxs'][0]["source"]:
                                for ctx_idx in range(len(_ex['ctxs'])):
                                    _ex['ctxs'][ctx_idx]["source"] = ds_domain
                            ctxs = _ex['ctxs']

                        _ex['ctxs'] = ctxs
                        data_to_merge.append(_ex)

                if domain_idx == 0:
                    merged_data = data_to_merge
                
                else:
                    for id_, (_, _ex) in enumerate(zip(merged_data, data_to_merge)):
                        assert merged_data[id_]['raw_query'] == _ex['raw_query']
                        merged_data[id_]['ctxs'].extend(_ex['ctxs'])

                        # Rerank based on score and only keep the top n_docs to avoid memory explosion
                        if merged_data[id_]['ctxs'] and merged_data[id_]['ctxs'][0] is not None:
                            merged_data[id_]['ctxs'] = sorted(merged_data[id_]['ctxs'], key=lambda x: x['retrieval score'], reverse=True)
                            merged_data[id_]['ctxs'] = merged_data[id_]['ctxs'][:cfg.evaluation.search.n_docs]
                            # make sure we still have n_docs documents
                            assert len(merged_data[id_]['ctxs']) == cfg.evaluation.search.n_docs
                        else:
                            assert id_ == 0 or id_ == 983  # the 983rd example in RPJ has an empty query 
            
            safe_write_jsonl(merged_data, merged_path)
        else:
            merged_data = []
            with open(merged_path, 'r') as fin:
                for line in fin:
                    ex = json.loads(line)
                    merged_data.append(ex)

        # Post-process to remove duplication using multithreading
        use_multi_process = True
        if use_multi_process:
            merged_data = multiprocess_deduplication(merged_data)
        else:
            for id_, ex in enumerate(merged_data):
                merged_data[id_]['ctxs'] =  remove_duplicates_with_minhash(merged_data[id_]['ctxs'], string_for_decontamination=merged_data[id_]['raw_query'])
                # merged_data[id_]['ctxs'] =  remove_duplicates_with_minhash(merged_data[id_]['ctxs'], string_for_decontamination=None)
                # pass

    if os.path.exists(base_merged_path) and cfg.evaluation.search.use_saved_dedup_data:
        merged_data = []
        with open(base_merged_path, 'r') as fin: 
            for line in fin:
                ex = json.loads(line)
                merged_data.append(ex)
    else:
        # Write merged and reranked data to a new JSONL file
        os.makedirs(os.path.dirname(base_merged_path), exist_ok=True)
        safe_write_jsonl(merged_data, base_merged_path)
    
    # Subsample document from B(n_docs, p)
    seed = cfg.evaluation.search.get('subsample_seed', 1000)
    if cfg.evaluation.search.topk_subsample_p < 1:
        # Set a random seed for subsampling
        random.seed(seed)
        
        for id_, _ in enumerate(merged_data):
            subsampled_ctxs = subsample_by_coin_flip(merged_data[id_]['ctxs'], cfg.evaluation.search.topk_subsample_p)
            merged_data[id_]['ctxs'] = subsampled_ctxs
    
    # Post-process to rerank
    if cfg.evaluation.search.get('rerank_method', None):
        rerank_n_docs = cfg.evaluation.search.get('rerank_n_docs', None)
        no_enough_rerank_data_cout = 0
        for id_, ex in enumerate(merged_data):
            merged_data[id_]['ctxs'], no_enough_rerank_data = extract_rerank_docs(merged_data[id_]['ctxs'], rerank_n_docs)
            no_enough_rerank_data_cout += no_enough_rerank_data
        if no_enough_rerank_data_cout:
            print(f"WARNING: there are {no_enough_rerank_data_cout} example having no enough data for reranking!")
        
        print(f"Reranking with method: {cfg.evaluation.search.rerank_method}")
        if cfg.evaluation.search.rerank_method in ['lexical', 'unigram_f1', 'inclusion']:
            all_answers = get_answers(cfg)
            for id_, ex in tqdm(enumerate(merged_data)):
                query = ex['raw_query']
                merged_data[id_]['ctxs'] = post_rerank_ctxs(ex['ctxs'], all_answers[query], cfg)
    
    # Additional decontamination for ablation study
    ablation_study = False
    if ablation_study:
        for id_, _ in enumerate(merged_data):
            merged_data[id_]['ctxs'] = additional_decon(merged_data[id_])
    
    # Additional short chunk removal
    for id_, _ in enumerate(merged_data):
        merged_data[id_]['ctxs'] = additional_remove_short_chunk(merged_data[id_]['ctxs'])

    # Check the number of remaining documents
    no_enough_data_count = 0
    for id_, _ in enumerate(merged_data):
        if len(merged_data[id_]['ctxs']) < 3:
            no_enough_data_count += 1
            print(f"WARNING: the subsampled documents only have {len(merged_data[id_]['ctxs'])} left!")
    
    # Write merged and reranked data to a new JSONL file
    output_path = f"full_subsampled_{str(cfg.evaluation.search.topk_subsample_p)}_{seed}_{os.path.basename(base_merged_path)}"
    output_path = os.path.join(os.path.dirname(base_merged_path), output_path)
    if cfg.evaluation.search.get('rerank_method', None):
        output_path = output_path.replace('.jsonl', f'_rerank_{cfg.evaluation.search.rerank_method}.jsonl')
    elif ablation_study:
        output_path = output_path.replace('.jsonl', f'_standard_decon.jsonl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    safe_write_jsonl(merged_data, output_path)
    
    print(f"Saved merged results to {output_path} with {no_enough_data_count} documents having less than 5 documents.")


def additional_decon(example):
    answer = example['raw_query']
    num_doc_before = len(example['ctxs'])
    clean_ctxs = []
    for ctx in example['ctxs']:
        # if check_below_lexical_overlap_threshold(ctx['retrieval text'], answer, 8, 'longest'):
        if check_below_lexical_overlap_threshold(ctx['retrieval text'], answer, 0.8, 'jaccard'):
            clean_ctxs.append(ctx)
    num_doc_after = len(clean_ctxs)
    print(f"Additional decon: {num_doc_before - num_doc_after} documents are removed")
    return clean_ctxs

def additional_remove_short_chunk(ctxs):
    new_ctxs = []
    for ctx in ctxs:
        if len(ctx['retrieval text'].split(' ')) > 12:
            new_ctxs.append(ctx)
    return new_ctxs

def extract_rerank_docs(ctxs, rerank_n_docs):
    filtered_ctxs = [ctx for ctx in ctxs if ctx['quality score']]
    if rerank_n_docs is None or len(filtered_ctxs) >= rerank_n_docs:
        return filtered_ctxs[:rerank_n_docs], 0
    else:
        return filtered_ctxs, 1


def post_process_ctxs(ctxs):
    # + remove ctx that is shorter than 5 words
    # + deduplicate ctx with >80% 13-gram overlap
    if ctxs[0] is None:
        return ctxs

    def remove_short_ctx(ctxs):
        new_ctxs = []
        for ctx in ctxs:
            # remove chunks that have less than 5 words
            if len(ctx['retrieval text'].split(' ')) > 5:
                new_ctxs.append(ctx)
        if len(new_ctxs) < 5:
            new_ctxs = ctxs[:5]
        return new_ctxs
                
    def remove_duplication(ctxs, first_k=5):
        new_ctxs = []
        num_passed = 0
        ctx_idx = 0
        while ctx_idx < len(ctxs) and num_passed < first_k:
            ctx = ctxs[ctx_idx]
            ctx_idx += 1
            can_add = True
            for added_ctx in new_ctxs:
                can_add = check_below_lexical_overlap_threshold(ctx['retrieval text'], added_ctx['retrieval text'], threshold=0.8, mode='jaccard')
                if not can_add:
                    # with open('count_intra_and_inter.txt', 'a') as fout:
                    #     if ctx['source'] == added_ctx['source']:
                    #         fout.write('intra\n')
                    #     else:
                    #         fout.write('inter\n')
                    break
            if can_add:
                new_ctxs.append(ctx)
                num_passed += 1
            
        new_ctxs =  new_ctxs + ctxs[ctx_idx:]
        # if len(new_ctxs) < 5:
        #     pdb.set_trace()
        return new_ctxs
    
    return remove_duplication(remove_short_ctx(ctxs))


def post_rerank_ctxs(ctxs, answers, cfg):
    rerank_method = cfg.evaluation.search.rerank_method

    good_ctxs = [ctx for ctx in ctxs if ctx['quality score']]
    bad_ctxs = [ctx for ctx in ctxs if not ctx['quality score']]
    assert len(good_ctxs) + len(bad_ctxs) == len(ctxs)
    
    if rerank_method == 'lexical':
        good_ctxs = lexical_rerank(good_ctxs, answers)
    elif rerank_method == 'inclusion':
        good_ctxs = inclusion_rerank(good_ctxs, answers)
    elif rerank_method == 'unigram_f1':
        good_ctxs = unigram_f1_rerank(good_ctxs, answers)
    
    return good_ctxs + bad_ctxs

def get_answers(cfg):
    
    if cfg.tasks.eval.task_name == 'perplexity':
        eval_data = load_eval_data(cfg)

        all_answers = []
        for ex in eval_data:
            answer = extract_ppl_answer(ex['raw_inputs'], ex['raw_query'])
            all_answers.append([answer])
    
    elif cfg.tasks.eval.task_name == 'lm-eval':
        answer_path = cfg.evaluation.search.answer_path

        all_answers = {}
        with open(answer_path, 'r') as fin:
            for line in fin:
                ex = json.loads(line)
                if 'triviaqa' in answer_path:
                    answer = {ex['query']: ex['answer']['normalized_aliases']}
                elif 'nq_open' in answer_path:
                    answer = {ex['query']: ex['answer']}
                else:
                    answer = {ex['query']: ex['answer']}
                all_answers.update(answer)

    return all_answers

def extract_ppl_answer(raw_input, raw_query):
    inputs = raw_input.replace('<|endoftext|>', '')
    query = raw_query.replace('<|endoftext|>', '')
    try:
        answer = inputs.replace(query, '')
    except:
        try:
            answer = inputs.replace(query[:-1], '')
        except:
            answer = inputs[-len(inputs)//2:]
    return answer

def inclusion_rerank(ctxs, answers):
    inclusion_scores = [inclusion_metric(ctx['retrieval text'], answers) for ctx in ctxs]
    ctxs = sort_ctxs_with_1_scores(ctxs, inclusion_scores)
    return ctxs

def unigram_f1_rerank(ctxs, answers):
    unigram_f1_scores = [unigram_f1_metric(ctx['retrieval text'], answers) for ctx in ctxs]
    ctxs = sort_ctxs_with_1_scores(ctxs, unigram_f1_scores)
    return ctxs

def lexical_rerank(ctxs, answers):
    if not ctxs or ctxs[0] is None:
        return ctxs
    
    inclusion_scores = [inclusion_metric(ctx['retrieval text'], answers) for ctx in ctxs]
    unigram_f1_scores = [unigram_f1_metric(ctx['retrieval text'], answers) for ctx in ctxs]
    retrieval_scores = [ctx['retrieval score'] for ctx in ctxs]

    ctxs = sort_ctxs_with_3_scores(ctxs, inclusion_scores, unigram_f1_scores, retrieval_scores)
    return ctxs

def inclusion_metric(ctx, answers):
    if not ctx or not answers:
        return 0
    
    score_list = []
    for answer in answers:
        score = 1 if normalize_text(answer) in normalize_text(ctx) else 0
        score_list.append(score)
    return max(score_list)

def unigram_f1_metric(ctx, answers):
    if not ctx or not answers:
        return 0
    
    norm_answers = [normalize_text(ans) for ans in answers]
    norm_ctx = normalize_text(ctx)

    common_tokens = [
        Counter(norm_ctx.split()) & Counter(norm_ans.split()) for norm_ans in norm_answers
    ]
    num_same = [sum(common.values()) for common in common_tokens]

    score_list = []
    for i, num in enumerate(num_same):
        if num == 0:
            score_list.append(0.0)
        else:
            p = 1.0 * num / len(norm_ctx.split())
            r = 1.0 * num / len(norm_answers[i].split())
            f1 = 2 * p * r / (p + r)
            score_list.append(f1)

    return max(score_list)

def sort_ctxs_with_1_scores(ctxs, scores_1):
    combined_list = list(zip(scores_1, ctxs))

    combined_list.sort(key=lambda x: x[0], reverse=True)

    sorted_ctxs = [ctx for _, ctx in combined_list]
    return sorted_ctxs

def sort_ctxs_with_3_scores(ctxs, scores_1, scores_2, scores_3):
    combined_list = list(zip(scores_1, scores_2, scores_3, ctxs))

    combined_list.sort(key=lambda x: x[2], reverse=True)
    combined_list.sort(key=lambda x: x[1], reverse=True)
    combined_list.sort(key=lambda x: x[0], reverse=True)

    sorted_ctxs = [ctx for _, _, _, ctx in combined_list]
    return sorted_ctxs


def normalize_text(text):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(lower(text)))


def search_sparse_topk(cfg):
    index_args = cfg.datastore.index
    eval_args = cfg.evaluation

    if isinstance(index_args.index_shard_ids[0], ListConfig):
        print(f"Multi-index mode: building a BM25 index over {len(index_args.index_shard_ids)} shards...")
        index_shard_ids_list = [i for index_shards in index_args.index_shard_ids for i in index_shards]
    else:
        print(f"Single-index mode: building a BM25 index over {index_args.index_shard_ids} shards...")
        index_shard_ids_list = index_args.index_shard_ids

    # check if all search results exist
    output_path = get_search_output_path(cfg, index_shard_ids_list)
    all_exist = os.path.exists(output_path)

    if all_exist and not eval_args.search.overwrite:
        logging.info(f'All search results for {index_args.index_shard_ids} exist, skipping searching.')
    
    else:
        # load eval data
        data = load_eval_data(cfg)
        logging.info(f"Searching for {len(data)} total evaluation samples...")

        # load index
        bm25_index_path = os.path.join(get_bm25_index_dir(cfg, index_shard_ids_list), 'index')
        assert os.path.exists(bm25_index_path), f"The index path does not exist, please build the index first\nMissing: {bm25_index_path}"
        logging.info(f"Loading BM25 index from {bm25_index_path}")
        searcher = LuceneSearcher(bm25_index_path)

        for ex in tqdm(data):
            query = ex["raw_query"]
            if query:
                hits = searcher.search(query, cfg.evaluation.search.n_docs)  
                # ctxs = []
                # for i in range(len(hits)):
                #     raw = searcher.doc(hits[i].docid).raw()
                #     ex = json.loads(raw)
                #     ctxs.append(
                #         {
                #             "id": int(ex["id"]),
                #             "retrieval text": ex["contents"],
                #             "retrieval score": hits[i].score,
                #         } for i in range(len(hits))
                #     )
                # if len(hits) < cfg.evaluation.search.n_docs:  # will there be any case where len(hits) < n_docs?
                #     dummy_ctx = {"id": None, "retrieval text": '', "retrieval score": float('-inf')}
                #     ctxs += [dummy_ctx] * (cfg.evaluation.search.n_docs - len(hits))
                #     print(f"The number of retrieved documents is less than n_docs: {len(hits)} < {cfg.evaluation.search.n_docs}")
                ex["ctxs"] = [
                    {
                        # "id": int(ex["id"]),
                        "retrieval text": json.loads(searcher.doc(hits[i].docid).raw())["contents"],
                        "retrieval score": hits[i].score,
                        } for i in range(len(hits))
                ]
            else:
                ex["ctxs"] = [None]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        safe_write_jsonl(data, output_path)


def safe_write_jsonl(data, output_file):
    success = False
    try:
        with open(output_file, 'w') as fout:
            for ex in data:
                fout.write(json.dumps(ex) + "\n")
            success = True
        logging.info(f"Saved results to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
    # If an error was raised, and success is still False, delete the file
        if not success and os.path.exists(output_file):
            os.remove(output_file)
            print(f"File '{output_file}' has been deleted due to an error.")


def search_topk(cfg):
    if cfg.model.get("sparse_retriever", None):
        search_sparse_topk(cfg)
    else:
        search_dense_topk(cfg)