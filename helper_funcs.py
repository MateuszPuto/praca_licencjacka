import faiss
import dask.dataframe as dd
import numpy as np
from transformers import pipeline
from operator import itemgetter
import torch
import redis

def tsv_to_df(filename: str, col_name: list=None, index_col: str=None):
    '''Reads .tsv file to dask dataframe'''

    kwargs = {'on_bad_lines': 'skip', 'delimiter': '\t', 'names': col_name}
    df = dd.read_csv(f"./{filename}.tsv", **kwargs)

    return df.set_index(index_col, sorted=True)

def text_to_tensor(sentence: str, model):
    '''Converts string to tensor using provided encoder model'''

    return model.encode(sentence, convert_to_tensor=True)

def generate_queries(text: str, model, tokenizer, prefix="answer2query"):
    '''Generates queries for text; (mostly to be used with T5ForConditionalGeneration)'''

    text = prefix + ": " + text

    input_ids = tokenizer.encode(text, max_length=384, truncation=True, return_tensors='pt')
    outputs = model.generate(
        input_ids=input_ids,
        max_length=64,
        do_sample=True,
        top_p=0.95,
        num_return_sequences=10)

    queries = []
    for i in range(len(outputs)):
        query = tokenizer.decode(outputs[i], skip_special_tokens=True)
        queries.append(query)

    return queries

def create_index(d: int, description: str, gpu: bool):
    '''Creates FAISS search index'''

    index = faiss.index_factory(d, description, faiss.METRIC_INNER_PRODUCT)

    if gpu == True:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    return index

def populate_index(xb, index, to_cpu):
    '''Trains and adds vectors to FAISS index'''

    if to_cpu:
        xb = xb.cpu().numpy()

    faiss.normalize_L2(xb)

    index.train(xb)
    index.add(xb)


def create_redis():
    '''Creates redis database'''

    rds = redis.Redis(host='localhost', port=6379, decode_responses=True)

    return rds

def save_collection(rds, collection, n):
    '''Saves first N elements from collection to redis cache'''

    for key, value in collection.head(n).iterrows():
        rds.set(key, value[0])

def get_entries(filename: str, size: int, encoder, gpu: bool):
    '''Produces torch tensor or numpy array from MS MARCO paragraph dataset encoded with provided encoder'''

    x = []

    with open(filename, 'r') as file:
        for i in range(size):
            paragraph = file.readline().split('\t')[1]

            encoding = text_to_tensor(paragraph, encoder)

            if not gpu:
                vec = encoding.unsqueeze(0).detach().cpu().numpy().astype('float32')
            else:
                vec = encoding.unsqueeze(0)
            x.append(vec)

    if not gpu:
        return np.squeeze(np.stack(x))
    else:
        return torch.squeeze(torch.stack(x))

def get_query_document_pair(query_id, collection, queries, qrels):
        '''Dask get (query, document) pair with specified ID'''

        query = queries.loc[:, query_id].compute()
        rel = qrels.loc[int(query.index.to_numpy()[0]), "document id"].compute().iloc[0]
        document = collection.loc[rel].compute()

        return (query, document)

def query_document_generator(collection, queries, qrels):
    '''Dask (query, document) generator'''

    for rel in qrels.iterrows():
        query_id = rel[0]
        document_id = rel[1].loc["document id"]
        
        query = queries.loc[query_id].compute()
        document = collection.loc[document_id].compute()

        yield (query, document)

def search(query_val: str, num_results: int, encoder, index, normalize=True):
    '''Search query string in FAISS index, returns results and distances'''

    vec = text_to_tensor(query_val, encoder).unsqueeze(0).detach().cpu().numpy().astype('float32')
    
    if normalize:
        faiss.normalize_L2(vec)

    d, i = index.search(vec, num_results)

    return (i, d)

def get_documents(i, collection):
    '''Dask get documents'''

    documents = []
    for result in i[0]:
        if result == -1:
            break
        else:
            doc_number = result
            document_text = collection.loc[doc_number]["paragraph"].compute()[doc_number]
            documents.append(document_text)

    return documents

def get_documents_redis(rds, indices):
    '''Redis get documents'''

    documents = []

    for i in indices[0].tolist():
        doc = str(rds.get(i))

        documents.append(doc)

    return documents

def rerank(query_val: str, documents: list, indices: list, cross_encoder):
    '''Reranks documents with respect to query_val'''

    query_document = zip([query_val for i in range(len(documents))], documents)
    q_d = list(query_document)
    scores = cross_encoder.predict(q_d)

    reranked_results = sorted(zip(zip(indices[0], documents), scores), key=itemgetter(1), reverse=True)

    return reranked_results

def generate_answer(query_val: str, reranked_results):
    '''Generate answer using retrieved results'''
    
    paragraphs = [(reranked_results[i][0][1], reranked_results[i][1]) for i in range(len(reranked_results))]

    qa_model = pipeline("question-answering", model="deepset/tinyroberta-squad2")
    question = query_val
    context = "".join([paragraphs[i][0] + "[SEP]" for i in range(len(reranked_results))])
    answer = qa_model(question = question, context = context)

    return answer

def search_and_rerank(query_val: str, num_results: int, encoder, cross_encoder, index, collection):
    '''Wraps consecutive search and rerank'''

    indices, distances = search(query_val, num_results, encoder, index)
    documents = get_documents(indices, collection)
    reranked_results = rerank(query_val, documents, indices, cross_encoder)

    return reranked_results

def search_and_rerank_redis(query_val: str, num_results: int, encoder, cross_encoder, index, rds):
    '''Search and rerank using redis'''

    indices, distances = search(query_val, num_results, encoder, index)
    documents = get_documents_redis(rds,indices)
    reranked_results = rerank(query_val, documents, indices, cross_encoder)

    return reranked_results


def mrr_at(k: int, result: list, match_id: int):
    '''Calculates reciprocal rank; MRR@k = sum(reciprocal_ranks[:])'''

    if match_id in result[:k]:        
        reciprocal_rank = 1 / (result[:k].index(match_id) + 1)
    else:
        reciprocal_rank = 0

    return reciprocal_rank