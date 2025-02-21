import os
import json
import csv
import pickle
import gzip
import logging
import pdb

import numpy as np
import transformers


############################## Training ##############################
def fast_load_jsonl_shard(args, shard_index):
    """
    This function is designed to handle large datasets by only loading the specific portion of data (shard) that 
    corresponds to the given shard index.

    Shards are determined by dividing the total size of all files in the directory evenly by `num_shards`. 
    This function reads only the data portion of the `shard_index` shard, chunks the text from each line 
    based on `chunk_sz`, and appends each chunk to a list with an incremental ID.
    """
    raw_data_path = args.raw_data_path
    num_shards = args.num_shards
    chunk_sz = args.chunk_size
    min_chunk_sz = args.get('min_chunk_sz', 0)
    keep_last = args.get('keep_last_chunk', True)

    passage_shard_save_path = os.path.join(args.passages_dir, f'raw_passages-{shard_index}-of-{num_shards}.pkl')
    
    if os.path.exists(passage_shard_save_path):
        logging.info(f'Loading from {passage_shard_save_path}...')
        with open(passage_shard_save_path, 'rb') as file:
            passages = pickle.load(file)
        return passages

    if not os.path.exists(raw_data_path):
        logging.info(f"{raw_data_path} does not exist")
        return

    if os.path.isdir(raw_data_path):
        all_file_paths = [os.path.join(raw_data_path, file) for file in os.listdir(raw_data_path)]
    else:
        all_file_paths = [raw_data_path]
    
    file_sizes = []
    for file in all_file_paths:
        if os.path.isdir(raw_data_path):
            file_path = os.path.join(raw_data_path, file)
        else:
            file_path =  file
        file_sizes.append(os.path.getsize(file_path))
    total_size = sum(file_sizes)

    shard_size = total_size / num_shards
    shard_start = shard_size * shard_index
    shard_end = shard_start + shard_size if shard_index < shard_size - 1 else total_size
    
    current_pos = 0
    shard_files = []
    for file_path, file_size in zip(all_file_paths, file_sizes):
        next_pos = current_pos + file_size
        if next_pos > shard_start and current_pos < shard_end:
            # This file is part of the i-th shard
            start_in_file = max(shard_start - current_pos, 0)
            end_in_file = min(shard_end - current_pos, file_size)
            shard_files.append((file_path, start_in_file, end_in_file))
        current_pos = next_pos

    passages = []
    idx = 0
    for file_path, start_in_file, end_in_file in shard_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            file.seek(int(start_in_file))
            # Skip the rest of the partial line after seeking, if not at the start of the file
            if start_in_file != 0:
                file.readline()
            
            while file.tell() < end_in_file:
                line = file.readline().strip()
                if line:
                    ex = json.loads(line)
                    chunks = split_data_into_chunks(ex['text'].strip(), chunk_sz, min_chunk_sz, keep_last)
                    for chunk in chunks:
                        passages.append({
                            "text": chunk,
                            "origin_id": ex['id'] if 'id' in ex else None,
                            "id": idx,
                            "shard_id": shard_index,
                            "num_shards": num_shards,
                        })
                        idx += 1
                else:
                    break
    
    if args.get('passages_dir', None):
        os.makedirs(args.passages_dir, exist_ok=True)
        with open(passage_shard_save_path, 'wb') as file:
            pickle.dump(passages, file)

    return passages

# Used for passage retrieval (old, inefficient bc it needs load the whole data)
def load_passages(path, chunk_sz=None, min_chunk_sz=0, keep_last=True, num_load_files=None):
    if not os.path.exists(path):
        logging.info(f"{path} does not exist")
        return
    passages = []
    idx = 0
    # load all passages together if is passed a directory
    if not os.path.isdir(path):
        paths = [path]
    else:
        paths = [os.path.join(path, file) for file in os.listdir(path)]
        # todo: support subsampling
        try:
            paths = sorted(paths, key=lambda x: int(x.split('-')[-1].split('.jsonl')[0]))
        except:
            print('No sorting on the raw data paths.')
        if num_load_files:
            paths = paths[0:num_load_files]
        print(paths)
    for path in paths:
        logging.info(f"Loading passages from: {path}")
        with open(path) as fin:
            if path.endswith(".jsonl"):
                for k, line in enumerate(fin):
                    ex = json.loads(line)
                    chunks = split_data_into_chunks(ex["text"].strip(), chunk_sz, keep_last)
                    for chunk in chunks:
                        passages.append({
                            "text": chunk,
                            "id": idx,
                            })
                        idx += 1   
            elif path.endswith(".csv"):
                # the dpr wiki is pre-chunked to 100 words
                reader = csv.reader(fin, delimiter="\t")
                for k, row in enumerate(reader):
                    if not row[0] == "id":
                        ex = {"id": row[0], "title": row[2], "text": row[1]}
                        passages.append(ex)
            elif path.endswith(".parquet"):
                import pandas as pd
                df = pd.read_parquet(path, engine='fastparquet')

                if 'wikitext' in path:
                    idx = 0
                    for ex_text in df.text:
                        if ex_text:  # skip empty string (1467/4358 is empty in the test set)
                            chunks = split_data_into_chunks(ex_text, chunk_sz, keep_last)
                            for chunk in chunks:
                                passages.append({
                                    "text": chunk,
                                    "id": idx,
                                    })
                                idx += 1
            elif path.endswith(".json.gz"):
                with gzip.open(path, 'rb') as gz_file:
                    file_content = gz_file.read()
                    decoded_content = file_content.decode('utf-8')
                    json_strings = decoded_content.split('\n')
                    json_strings = [js for js in json_strings if js]
                    data = [json.loads(js) for js in json_strings]  # dict_keys(['added', 'attributes', 'created', 'id', 'metadata', 'source', 'text', 'version'])
                    for ex in data:
                        chunks = split_data_into_chunks(data['text'], chunk_sz, min_chunk_sz, keep_last)
                        for chunk in chunks:
                            passages.append({
                                "text": chunk,
                                "id": idx,
                            })
                            idx += 1
                        
    return passages


def split_data_into_chunks(text, chunk_sz, min_chunk_sz, keep_last):
    # returns chunks of size <= chunk_sz + min_chunk_sz
    if chunk_sz is None:
        return [text]
    
    text = text.split()
    N = len(text) if keep_last else len(text)-len(text)%chunk_sz
    chunks = [' '.join(text[i:i+chunk_sz]) for i in range(0,N,chunk_sz)]

    if len(chunks) > 1 and len(chunks[-1].split(' ')) < min_chunk_sz:
        # merge the last min_chunk_sz words to the previous chunk
        last_chunk = chunks.pop()
        chunks[-1] += ' ' + last_chunk

    return chunks


############################## Evaluation ##############################
def load_eval_data(cfg):
    path = cfg.evaluation.data.eval_data
    task_name = cfg.tasks.eval.task_name
    
    # use lm_tokenizer to make sure the number of tokens consitent with the ones for PPL computation
    tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model.lm_model)

    if path.endswith('.jsonl'):
        data = load_jsonl(path)    # 'text', 'meta'
    elif path.endswith('.parquet'):
        data = load_parquet(path)
    
    if task_name == 'perplexity':
        eval_data_args = cfg.evaluation.data

        data = prepare_ppl_eval_data(
            data, 
            tokenizer, 
            eval_data_args.max_eval_data_seq_length, 
            eval_data_args.eval_stride, 
            eval_data_args.merge, 
            eval_data_args.num_eval_samples,
            eval_data_args.seed,
            )
    
    elif task_name == 'lm-eval':
        # prepare data for lm-evaluate-harness
        data = prepare_lm_eval_data(data)
    
    elif task_name == 'mmlu':
        # (test case) prepare mmlu for instruct-eval
        data = prepare_mmlu_eval_data(data)
    
    else:
        raise AttributeError

    return data


def prepare_lm_eval_data(data):
    """
    Use the question as the query. (0-shot)
    """
    new_data = []
    for ex in data:
        ex.update({'raw_query': ex['query']})
        new_data.append(ex)
    return new_data


def prepare_mmlu_eval_data(data):
    """
    Use the question as the query. (0-shot)
    """
    new_data = []
    for ex in data:
        ex.update({'raw_query': ex['prompt_end']})
        new_data.append(ex)
    return new_data


def prepare_ppl_eval_data(data, tokenizer, max_seq_length, stride, merge, num_eval_samples=None, seed=310):
    if tokenizer is None:
        logging.info(f"Constructing evaluation samples from {len(data)} raw documents for close-book evaluation...")
        return [{'raw_inputs': ex['text']} for ex in data]

    input_ids = [tokenizer(ex['text'])['input_ids'] for ex in data]

    pad_token_id = tokenizer.pad_token_id if tokenizer.eos_token_id is None else tokenizer.eos_token_id
    if merge:
        # todo: cluster similar context together before merging
        flatten_input_ids = np.array([_id for ids in input_ids for _id in ids])
        all_input_ids, all_targets = batch_merged(flatten_input_ids, max_seq_length=max_seq_length, stride=stride, pad_token_id=pad_token_id)    # if pad to -100 that will be ignored in HF CE but cannot decode
    else:
        # todo: no tokens used for query when length < stride
        all_input_ids, all_targets = batch(input_ids, max_seq_length=max_seq_length, stride=stride, pad_token_id=pad_token_id)
    
    if num_eval_samples:
        np.random.seed(seed)
        indices = np.random.permutation(len(all_input_ids))[:num_eval_samples]
        all_input_ids, all_targets = all_input_ids[indices], all_targets[indices]

    new_data = []
    logging.info(f"Constructing evaluation samples from {len(all_input_ids)} raw documents...")
    for input_ids, targets in zip(all_input_ids, all_targets):
        input_ids, targets = input_ids.tolist(), targets.tolist()
        query_ids = [int(_id) for _id, t in zip(input_ids, targets) if t==pad_token_id]
        new_data.append({
            # 'input_ids': input_ids,
            # 'targets': targets,  # not used for HF models
            'raw_inputs': tokenizer.decode(input_ids, skip_special_tokens=True), #  <- removing [CLS] will cause inputs to be shorter than targets
            'raw_query': tokenizer.decode(query_ids, skip_special_tokens=True),
        })
    logging.info(f"Finished construction with {len(new_data)} evaluation samples.")

    return new_data


def load_jsonl(data_path):
    assert os.path.exists(data_path)
    data = []
    with open(data_path, "r") as file:
        for line in file:
            ex = json.loads(line)
            data.append(ex)
    return data


def load_parquet(data_path):
    import pandas as pd
    df = pd.read_parquet(data_path, engine='fastparquet')
    data = []
    for ex_text in df.text:
        if ex_text:
            data.append({'text': ex_text})
    return data


def batch_merged(flatten_input_ids, max_seq_length, stride, pad_token_id, flatten_masks=None):
    all_input_ids = []
    all_targets = []
    prev_end_loc = 0

    for begin_loc in range(0, len(flatten_input_ids)-1, stride):
        end_loc = min(begin_loc + max_seq_length, len(flatten_input_ids)-1)
        trg_len = end_loc - prev_end_loc

        # we feed begin_loc ~ prev_end_log ~ end_log
        # but calculcate loss only for prev_end_log ~ end_log
        input_ids = flatten_input_ids[begin_loc:end_loc].copy()
        target_ids = flatten_input_ids[begin_loc+1:end_loc+1].copy()

        if flatten_masks is not None:
            for i, m in enumerate(flatten_masks[begin_loc+1:end_loc+1]):
                if not m:
                    target_ids[i] = pad_token_id

        target_ids[:-trg_len] = pad_token_id
        assert input_ids.shape==target_ids.shape

        if end_loc == len(flatten_input_ids)-1 and len(input_ids)==len(target_ids)<max_seq_length:
            pads = np.array([pad_token_id for _ in range(max_seq_length-len(input_ids))])
            input_ids = np.concatenate([input_ids, pads])
            target_ids = np.concatenate([target_ids, pads])

        assert len(input_ids)==len(target_ids)==max_seq_length, (begin_loc, end_loc, len(flatten_input_ids))

        all_input_ids.append(input_ids)
        all_targets.append(target_ids)

        prev_end_loc = end_loc

        if end_loc == len(flatten_input_ids)-1:
            break

    assert np.all([len(input_ids)==max_seq_length for input_ids in all_input_ids])
    assert np.all([len(input_ids)==max_seq_length for input_ids in all_targets])
    return np.stack(all_input_ids), np.stack(all_targets)

def batch(input_ids, max_seq_length, stride, pad_token_id):
    all_input_ids, all_targets = [], []
    for _input_ids in input_ids:
        _all_input_ids, _all_targets = batch_merged(np.array(_input_ids), max_seq_length, stride, pad_token_id)
        all_input_ids.append(_all_input_ids)
        all_targets.append(_all_targets)
    return np.concatenate(all_input_ids, 0), np.concatenate(all_targets, 0)
