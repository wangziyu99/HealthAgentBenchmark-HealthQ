"""collection of medical QA datasets on HF that does not need auth.
used for building local vector db for extraction on open dataset"""

from df_as_search_db import get_any_case_df_as_vdb as get_db
from df_as_search_db import lazy_collate, collate_with_metadata, collate_with_metadata_with_output

from datasets import load_dataset

lavita_dataset = load_dataset("lavita/medical-qa-datasets","all-processed")
import pandas as pd 
lavita_df = pd.DataFrame(lavita_dataset['train']).reset_index()

from langchain.schema.document import Document


 
from functools import partial

lavita_w_output = get_db(lavita_df,
    collate_func=collate_with_metadata_with_output,
    # first_run=True,
    db_path="lavita_with_output_vdb")
lavita = get_db(lavita_df,
    collate_func=collate_with_metadata,
    # first_run=True,
    db_path="lavita_cases_vdb")



lavita_w_output_lazy = get_db(lavita_df,
    collate_func=partial(lazy_collate, cols=lavita_df.columns),
    # first_run=True,
    db_path="lavita_w_output_lazy_vdb")

lavita_lazy = get_db(lavita_df,
    collate_func=partial(lazy_collate, cols=lavita_df.columns),
    # first_run=True,
    db_path="lavita_lazy_vdb")


def search_cases_contents(query, db, k=6, sep = '\n'):
    result = db.similarity_search(query, k=k)
    return sep.join(list(map(lambda doc: doc.page_content, result)))

def search_cases_with_output(query, db,df, k=6,id_name = 'id',
                        template = lambda row: collate_with_metadata_with_output(row).page_content,
                        sep = '\n',
                        ):
    result = db.similarity_search(query, k=k)
    r = ''
    # for index in list(map(lambda doc: doc.metadata[id_name], result)):
        # row = df.iloc[index,:]
        # r+=sep+template(row)
    for index in list(map(lambda doc: doc.metadata[id_name], result)):
        row = df[df[id_name] == index]
        r+=sep+template(row)
    return r[len(sep):]




from functools import partial

# print(partial(search_cases_with_output, db = lavita ,sep = '\n', df = lavita_df)('headache'))
