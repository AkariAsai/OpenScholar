# Modified based on https://github.com/facebookresearch/contriever/blob/main/passage_retrieval.py
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import logging
import pickle
import time
import glob
from tqdm import tqdm
import pdb
from typing import List, Tuple, Any
from abc import ABC, abstractmethod
from omegaconf import ListConfig
import subprocess

import faiss
import numpy as np
import torch
from transformers import GPTNeoXTokenizerFast
try:
    from pyserini.search.lucene import LuceneSearcher
except:
    logging.warning("Failed to import pyserini! Please install it from https://github.com/castorini/pyserini/tree/master.")

import contriever.src.contriever
import contriever.src.utils
import contriever.src.slurm
from contriever.src.evaluation import calculate_matches
import contriever.src.normalize_text

from src.data import fast_load_jsonl_shard, load_jsonl

os.environ["TOKENIZERS_PARALLELISM"] = "true"


device = 'cuda' if torch.cuda.is_available()  else 'cpu'


class BaseIndexer(ABC):
    # base class for all types of index
    def __init__(self):
        pass
    
    @abstractmethod
    def index_data(self):
        raise NotImplementedError()
    
    @abstractmethod
    def search_topk(self):
        raise NotImplementedError()
    
    @abstractmethod
    def save_index(self):
        raise NotImplementedError()
    
    @abstractmethod
    def load_index(self):
        raise NotImplementedError()


class Indexer(object):

    def __init__(self, vector_sz, n_subquantizers=0, n_bits=8):
        if n_subquantizers > 0:
            # do not use buggy PQ, it returns the same results regardless of query in current codes
            self.index = faiss.IndexPQ(vector_sz, n_subquantizers, n_bits, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(vector_sz)
        #self.index_id_to_db_id = np.empty((0), dtype=np.int64)
        self.index_id_to_db_id = []

    def index_data(self, ids, embeddings):
        self._update_id_mapping(ids)
        embeddings = embeddings.astype('float32')
        if not self.index.is_trained:
            self.index.train(embeddings)
        self.index.add(embeddings)

        print(f'Total data indexed {len(self.index_id_to_db_id)}')

    def search_knn(self, query_vectors: np.array, top_docs: int, index_batch_size: int = 2048) -> List[Tuple[List[object], List[float]]]:
        query_vectors = query_vectors.astype('float32')
        result = []
        nbatch = (len(query_vectors)-1) // index_batch_size + 1
        for k in tqdm(range(nbatch)):
            start_idx = k*index_batch_size
            end_idx = min((k+1)*index_batch_size, len(query_vectors))
            q = query_vectors[start_idx: end_idx]
            scores, indexes = self.index.search(q, top_docs)
            # convert to external ids
            db_ids = [[str(self.index_id_to_db_id[i]) for i in query_top_idxs] for query_top_idxs in indexes]
            result.extend([(db_ids[i], scores[i]) for i in range(len(db_ids))])
        return result

    def serialize(self, dir_path):
        index_file = os.path.join(dir_path, 'index.faiss')
        meta_file = os.path.join(dir_path, 'index_meta.faiss')
        print(f'Serializing index to {index_file}, meta data to {meta_file}')

        faiss.write_index(self.index, index_file)
        with open(meta_file, mode='wb') as f:
            pickle.dump(self.index_id_to_db_id, f)

    def deserialize_from(self, dir_path):
        index_file = os.path.join(dir_path, 'index.faiss')
        meta_file = os.path.join(dir_path, 'index_meta.faiss')
        print(f'Loading index from {index_file}, meta data from {meta_file}')

        self.index = faiss.read_index(index_file)
        print('Loaded index of type %s and size %d', type(self.index), self.index.ntotal)

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert len(
            self.index_id_to_db_id) == self.index.ntotal, 'Deserialized index_id_to_db_id should match faiss index size'

    def _update_id_mapping(self, db_ids: List):
        #new_ids = np.array(db_ids, dtype=np.int64)
        #self.index_id_to_db_id = np.concatenate((self.index_id_to_db_id, new_ids), axis=0)
        self.index_id_to_db_id.extend(db_ids)



def load_embeds(embed_path, dstore_size, dimension, dtype):
    assert os.path.exists(embed_path), embed_path
    return np.memmap(embed_path,
                     dtype=dtype,
                     mode="r",
                     shape=(dstore_size, dimension))


class IndexPQIVF(object):
    def __init__(self,
                 embed_path,
                 index_path,
                 trained_index_path,
                 prev_index_path,
                 dstore_size,
                 embeds=None,
                 dimension=2048,
                 dtype=np.float16,
                 ncentroids=4096,
                 code_size=64,
                 probe=8,
                 num_keys_to_add_at_a_time=1000000,
                 DSTORE_SIZE_BATCH=51200000,
                 index_type="ivfpq",
                 ):

        self.embed_path = embed_path
        self.index_path = index_path
        self.prev_index_path = prev_index_path
        self.trained_index_path = trained_index_path
        self.cuda = True

        self.dstore_size = dstore_size
        self.dimension = dimension
        self.ncentroids = ncentroids
        self.code_size = code_size
        self.probe = probe
        self.num_keys_to_add_at_a_time = num_keys_to_add_at_a_time
        self.index_type = index_type

        if embeds is not None:
            assert embeds.shape == (dstore_size, dimension)
            self.embs = embeds

        elif embed_path is not None and os.path.exists(embed_path):
            print ("Loading embeds (%d, %d) from %s" % (dstore_size, dimension, embed_path))
            self.embs = load_embeds(embed_path, dstore_size, dimension, dtype)

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self.index.nprobe = self.probe
        else:
            start_time = time.time()
            if self.prev_index_path is not None:
                assert os.path.exists(self.trained_index_path), self.trained_index_path
                assert os.path.exists(self.prev_index_path), self.prev_index_path

            if not os.path.exists(self.trained_index_path):
                print ("Sampling...")
                sample_size = 1000000
                random_samples = np.random.choice(np.arange(dstore_size),
                                                  size=[min(sample_size, dstore_size)],
                                                  replace=False)
                t0 = time.time()
                sampled_embs = self.get_embs(random_samples)
                #print(sampled_embs.shape, "<-- sampled embs shape")
                print (time.time()-t0)
                print ("Training index...")
                self._train_index(sampled_embs, self.trained_index_path)
                print ("Finish training (%ds)" % (time.time()-start_time))

            print ("Building index...")
            self.index = self._add_keys(self.index_path, self.prev_index_path if self.prev_index_path is not None else self.trained_index_path)

    def get_embs(self, indices):
        if type(self.embs)==list:
            # indices: [batch_size, K]
            embs = np.zeros((indices.shape[0], indices.shape[1], self.dimension), dtype=self.embs[0].dtype)
            for i, ref_embs in enumerate(self.embs):
                start = self.dstore_size*i
                end = self.dstore_size*(i+1)
                ref_indices = np.minimum(np.maximum(indices, start), end-1)
                embs += (indices >= start) * (indices < self.dstore_size*(i+1)) * ref_embs[ref_indices]
        else:
            embs = self.embs[indices]

        return embs.astype(np.float32)

    def search(self, query_embs, k=4096):
        all_scores, all_indices = self.index.search(query_embs.astype(np.float32), k)
        return all_scores, all_indices

    def get_knn_scores(self, query_emb, indices):
        embs = self.get_embs(indices) # [batch_size, k, dimension]
        scores = - np.sqrt(np.sum((np.expand_dims(query_emb, 1)-embs)**2, -1)) # [batch_size, k]
        return scores

    def _train_index(self, sampled_embs, trained_index_path):
        if self.index_type == "ivfpq":
            print("Building index with IVFPQ")
            quantizer = faiss.IndexFlatL2(self.dimension)
            start_index = faiss.IndexIVFPQ(quantizer,
                                           self.dimension,
                                           self.ncentroids,
                                           self.code_size,
                                           8)
        elif self.index_type == "pq":
            print("Building index with PQ")
            start_index = faiss.IndexPQ(self.dimension,
                                        self.code_size,
                                        8)
        else:
            raise AttributeError
        start_index.nprobe = self.probe
        np.random.seed(1)

        if self.cuda:
            # Convert to GPU index
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            gpu_index = faiss.index_cpu_to_gpu(res, 0, start_index, co)
            gpu_index.verbose = False

            # Train on GPU and back to CPU
            gpu_index.train(sampled_embs)
            start_index = faiss.index_gpu_to_cpu(gpu_index)
        else:
            # Faiss does not handle adding keys in fp16 as of writing this.
            start_index.train(sampled_embs)
        faiss.write_index(start_index, trained_index_path)

    def _add_keys(self, index_path, trained_index_path):
        index = faiss.read_index(trained_index_path)
        start_time = time.time()
        start = 0
        while start < self.dstore_size:
            end = min(self.dstore_size, start + self.num_keys_to_add_at_a_time)
            to_add = self.get_embs(range(start, end)).copy()
            index.add(to_add)
            start = end
            faiss.write_index(index, index_path)

            if start % 5000000 == 0:
                print ('Added %d tokens (%d min)' % (start, (time.time()-start_time)/60))

        print ('Adding took {} s'.format(time.time() - start_time))
        return index

    def _get_size(self,):
        return self.index.ntotal



def add_embeddings(index, embeddings, ids, indexing_batch_size, id_offset=0):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    if id_offset:
        ids_toadd = [id + id_offset for id in ids_toadd]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids


def get_index_dir_and_passage_paths(cfg, index_shard_ids=None):
    embedding_args = cfg.datastore.embedding
    index_args = cfg.datastore.index

    # index passages
    index_shard_ids = index_shard_ids if index_shard_ids is not None else index_args.get('index_shard_ids', None)
    if index_shard_ids:
        index_shard_ids = [int(i) for i in index_shard_ids]
        embedding_paths = [os.path.join(embedding_args.embedding_dir, embedding_args.prefix + f"_{shard_id:02d}.pkl")
                       for shard_id in index_shard_ids]

        # name the index dir with all shard ids included in this index, i.e., one index for multiple passage shards
        index_dir_name = '_'.join([str(shard_id) for shard_id in index_shard_ids])
        index_dir = os.path.join(os.path.dirname(embedding_paths[0]), f'index/{index_dir_name}')
        
    else:
        embedding_paths = glob.glob(index_args.passages_embeddings)
        embedding_paths = sorted(embedding_paths, key=lambda x: int(x.split('/')[-1].split(f'{embedding_args.prefix}_')[-1].split('.pkl')[0]))  # must sort based on the integer number
        embedding_paths = embedding_paths if index_args.num_subsampled_embedding_files == -1 else embedding_paths[0:index_args.num_subsampled_embedding_files]
        
        index_dir = os.path.join(os.path.dirname(embedding_paths[0]), f'index')
    
    return index_dir, embedding_paths


def index_encoded_data(index, embedding_paths, indexing_batch_size):
    allids = []
    allembeddings = np.array([])

    id_offset = 0  
    for i, file_path in enumerate(embedding_paths):
        print(f"Loading file {file_path}")
        with open(file_path, "rb") as fin:
            ids, embeddings = pickle.load(fin)
        
        assert min(ids)==0, f'Passage ids start with {min(ids)}, not 0: {file_path}'
        # each embedding shard's ids start with 0, so need to accumulate the id offset
        ids = [id + id_offset for id in ids]
        id_offset = max(ids) + 1

        allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        allids.extend(ids)
        
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)
        
    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    print("Data indexing completed.")


def build_dense_index(cfg):
    index_args = cfg.datastore.index

    if isinstance(index_args.index_shard_ids[0], ListConfig):
        print(f"Multi-index mode: building {len(index_args.index_shard_ids)} index for {index_args.index_shard_ids} sequentially...")
        index_shard_ids_list = index_args.index_shard_ids
    else:
        print(f"Single-index mode: building a single index over {index_args.index_shard_ids} shards...")
        index_shard_ids_list = [index_args.index_shard_ids]
    
    for index_shard_ids in index_shard_ids_list:
        # todo: support PQIVF
        index = Indexer(index_args.projection_size, index_args.n_subquantizers, index_args.n_bits)

        index_dir, embedding_paths = get_index_dir_and_passage_paths(cfg, index_shard_ids)
        logging.info(f"Indexing for passages: {embedding_paths}")
        
        os.makedirs(index_dir, exist_ok=True)
        index_path = os.path.join(index_dir, f"index.faiss")
        if index_args.save_or_load_index and os.path.exists(index_path) and not index_args.overwrite:
            index.deserialize_from(index_dir)
            pass
        else:
            print(f"Indexing passages from files {embedding_paths}")
            start_time_indexing = time.time()
            # index encoded embeddings
            index_encoded_data(index, embedding_paths, index_args.indexing_batch_size)
            print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
            if index_args.save_or_load_index:
                index.serialize(index_dir)


def get_index_passages_and_id_map(cfg, index_shard_ids=None):
    index_args = cfg.datastore.index

    index_shard_ids = index_shard_ids if index_shard_ids else index_args.get('index_shard_ids', None)
    assert index_shard_ids is not None
    
    index_shard_ids = [int(i) for i in index_shard_ids]

    passages = []
    passage_id_map = {}
    offset = 0
    for shard_id in index_shard_ids:
        # shard_passages = fast_load_jsonl_shard(cfg.datastore.embedding, shard_id)
        shard_passages = load_jsonl(
            os.path.join(
                cfg.datastore.embedding.passages_dir,
                f"raw_passages-{shard_id}-of-{cfg.datastore.embedding.num_shards}.jsonl"
            )
        )
        shard_id_map = {str(x["id"]+offset): x for x in shard_passages}
        
        offset += len(shard_passages)
        passages.extend(shard_passages)
        passage_id_map = {**passage_id_map, **shard_id_map}
        
    return passages, passage_id_map


class BM25Index(object):

    def __init__(self, index_dir, data_dir, stopwords):

        if not os.path.exists(index_dir):
            print ("Start building index for %s at %s" % (data_dir, index_dir))
            
            if stopwords:
                command = """python -m pyserini.index.lucene \
                --collection JsonCollection \
                --input '%s' \
                --index '%s' \
                --generator DefaultLuceneDocumentGenerator \
                --storeRaw --threads 1 \
                --stopwords '%s' """ % (data_dir, index_dir, stopwords)
            else:
                command = """python -m pyserini.index.lucene \
                --collection JsonCollection \
                --input '%s' \
                --index '%s' \
                --generator DefaultLuceneDocumentGenerator \
                --storeRaw --threads 1""" % (data_dir, index_dir)

            ret_code = subprocess.run([command],
                                    shell=True,
                                    #stdout=subprocess.DEVNULL,
                                    #stderr=subprocess.STDOUT
                                    )
            if ret_code.returncode != 0:
                print("Failed to build the index")
                exit()
            else:
                print("Successfully built the index")

        self.searcher = LuceneSearcher(index_dir)

    def search(self, query, k, continuation=False, shift=False, raw_only=True):
        # not used for simple raw text retrieval
        hits = self.searcher.search(query, k=k)
        out = []
        for hit in hits:
            docid = hit.docid

            if shift:
                docid = str(int(hit.docid)+1)
            
            raw = self.searcher.doc(docid).raw()
            
            if raw_only:
                if continuation:
                    next_item = self.searcher.doc(str(int(hit.docid)+1))
                    if next_item is not None:
                        next_raw = next_item.raw()
                        raw += next_raw  # todo: verify
                    else:
                        print ("The last block retrieved, so skipping continuation...")
                
                out.append(raw)
            
            else:
                input_ids = json.loads(raw)["input_ids"]

                if continuation:
                    next_item = self.searcher.doc(str(int(hit.docid)+1))
                    if next_item is not None:
                        next_raw = next_item.raw()
                        input_ids += json.loads(next_raw)["input_ids"]
                        raw += next_raw  # todo: verify
                    else:
                        print ("The last block retrieved, so skipping continuation...")

                out.append(input_ids)
        
        return out


def get_bm25_index_dir(cfg, index_shard_ids_list):
    shards_postfix = '_'.join([str(shard_id) for shard_id in index_shard_ids_list])
    index_dir = os.path.join(cfg.datastore.embedding.passages_dir, 'bm25')
    index_dir = os.path.join(index_dir, shards_postfix)
    return index_dir

def build_bm25_index(cfg):
    index_args = cfg.datastore.index
    stopwords = cfg.datastore.index.get("stopwords", None)

    if isinstance(index_args.index_shard_ids[0], ListConfig):
        print(f"Multi-index mode: building a BM25 index over {len(index_args.index_shard_ids)} shards...")
        index_shard_ids_list = [i for index_shards in index_args.index_shard_ids for i in index_shards]
    else:
        print(f"Single-index mode: building a BM25 index over {index_args.index_shard_ids} shards...")
        index_shard_ids_list = index_args.index_shard_ids
    
    bm25_base_path = get_bm25_index_dir(cfg, index_shard_ids_list)
    bm25_data_dir = os.path.join(bm25_base_path, 'data')
    bm25_index_dir = os.path.join(bm25_base_path, 'index')

    if not os.path.exists(bm25_index_dir):
        for index_shard_id in index_shard_ids_list:
            shard_passages, _ = get_index_passages_and_id_map(cfg, [index_shard_id])

            os.makedirs(bm25_data_dir, exist_ok=True)
            bm25_data_path = os.path.join(bm25_data_dir, f"data_{index_shard_id}.jsonl")
            if not os.path.exists(bm25_data_path):
                try:
                    with open(bm25_data_path, "w") as f:
                        for passage in tqdm(shard_passages):
                            f.write(json.dumps({
                                "id": str(passage["id"]),
                                "contents": passage["text"],
                            })+"\n")
                    logging.info(f"Saved passages to {bm25_data_path}.")
                except Exception as e:
                    logging.error(f"An error occurred: {e}")
                    os.remove(bm25_data_path)
                    logging.error(f"File '{bm25_data_path}' has been deleted due to an error.")
            else:
                logging.info(f"{bm25_data_path} exists, skipping..")
    
    logging.info(f'Loading/building bm25 search index from {bm25_index_dir}')
    searcher = BM25Index(bm25_index_dir, bm25_data_dir, stopwords)


def build_index(cfg):
    if cfg.model.get("sparse_retriever", None):
        build_bm25_index(cfg)
    else:
        build_dense_index(cfg)
