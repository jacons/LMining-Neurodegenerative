import json
import requests as requests
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from tqdm.notebook import tqdm as tqdm_notebook
tqdm_notebook.pandas()

PUBMED_BASEURL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&usehistory=y&'
BERN_PUBMED_IDS_LIMIT = 5
BERN_BASEURL = 'http://bern2.korea.ac.kr/pubmed/'


def search_listids(term: str, ret_start: int = 0, ret_max: int = 10000) -> list[str]:
    """
    Function able to search all pubmed id (pmid) of documents related to a given term.
    :param term: Term of search used in the pubmed search engine.
    :param ret_start: Starting offset to skip the first results.
    :param ret_max: Size of pubmed ids to retrieve.
    :return: List of pmed ids retrieved.
    """
    res = requests.get(f'{PUBMED_BASEURL}term={term}&retstart={ret_start}&retmax={ret_max}')
    if res.ok:
        result = res.json()
        return result['esearchresult']['idlist']
    raise ValueError(f'eutils.ncbi error: {res.status_code}')


def pubmed_ner(indexes: list[str]) -> list[dict]:
    """
    Function able to retrieve all annotations given by BERN model for each pubmed id passed in input.
    :param indexes: List of pubmed ids to tag with bern. They should be less or equal than 5.
    :return: List of annotations for each PubMed-id.
    """
    formatted_indexes = ','.join(indexes)
    res = requests.get(f'{BERN_BASEURL}{formatted_indexes}')
    if res.ok:
        result = res.json()
        return result
    raise ValueError(f'bern2 error: {res.status_code}\n{res.text}')


def download_dataset(term='neurodegenerative', folder='dataset/annotations') -> None:
    """
    Function able to download and cache data for a given term.
    :param term: Term to search pubmed ids.
    :param folder: Folder to save the annotations.
    """
    filepath = f'{folder}/{term}.json'
    list_ids = search_listids(term)
    annotations = []

    # if file exists load it to discard already downloaded documents
    if Path(filepath).exists():
        with open(filepath) as file:
            annotations = json.load(file)
            already_stored_ids = [annotation['pmid'] for annotation in annotations]
            list_ids = [id for id in list_ids if id not in already_stored_ids]

    # search by BERN_PUBMED_IDS_LIMIT
    pbar = tqdm(total=len(list_ids))
    while len(list_ids) > 0:
        try:
            ids_batch = list_ids[0:BERN_PUBMED_IDS_LIMIT]
            annotations.extend(pubmed_ner(ids_batch))
            with open(filepath, 'w') as file:
                json.dump(annotations, file)
            pbar.update(BERN_PUBMED_IDS_LIMIT)
        except Exception:
            print(f'Failed download ids: {ids_batch}, retry with single calls')
            for id in ids_batch:
                try:
                    annotations.extend(pubmed_ner([id]))
                    with open(filepath, 'w') as file:
                        json.dump(annotations, file)
                except Exception:
                    print(f'Failed download id: {id}')
                pbar.update(1)
        del list_ids[0:BERN_PUBMED_IDS_LIMIT]
    pbar.close()


def load_dataset(term='neurodegenerative', folder='dataset/annotations') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function able to load the dataset given a term. This function retrieves already cached data
    if there are and provides them through 2 dataframes.
    :param term: Term to search pubmed ids.
    :param folder: Folder to save bern annotations.
    :return: Tuple of dataframes related to:
        - associations pubmed id and texts
        - entity for each pubmed id with bern annotation information
    """
    filepath = f'{folder}/{term}.json'
    if not Path(filepath).exists():
        download_dataset(term=term, folder=folder)
    with open('dataset/annotations/neurodegenerative.json') as file:
        dataset = json.load(file)
        texts = dict(pmid=[], text=[])
        entities = dict(id=[], pmid=[], mention=[], obj=[], prob=[], span_begin=[],
                        span_end=[])
        for data in dataset:
            for key in texts.keys():
                texts[key].append(data[key])
            for annotation in data['annotations']:
                ids = annotation['id']
                ids.sort()
                meshids = list(filter(lambda x: 'mesh:' in x.lower(), ids))
                entities['id'].append(meshids[0] if len(meshids) > 0 else ids[0])
                entities['pmid'].append(data['pmid'])
                for key in ['mention', 'obj', 'prob']:
                    entities[key].append(annotation[key])
                entities['span_begin'].append(annotation['span']['begin'])
                entities['span_end'].append(annotation['span']['end'])
        df_texts = pd.DataFrame.from_dict(texts)
        df_entities = pd.DataFrame.from_dict(entities)
        return df_texts, df_entities
