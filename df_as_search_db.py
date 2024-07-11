

# from langchain.llms import LlamaCpp
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain.document_loaders import PyPDFDirectoryLoader


from langchain_community.embeddings import HuggingFaceEmbeddings
instructor_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                              model_kwargs={"device": "cuda"})
# from langchain.embeddings import HuggingFaceInstructEmbeddings
# instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", 
#                                                       model_kwargs={"device": "cuda"})

from langchain.schema.document import Document
    
def lazy_collate(row, cols):
    return Document(page_content = '\n'.join([cols[i] +': '+ str(row.values[i]) for i in range(len(row.values))]),
                    metadata = {'row':row})

def collate_with_metadata(row):
    return Document(page_content = f"{row['input']}",
        metadata =  {"id": row['index']})
    
def collate_with_metadata_with_output(row,
                                      input_name = 'input',
                                      output_name = 'output',
                                      ):
    return Document(page_content = f"Patient Input: {row[input_name]}\nDiagnosis Output: {row[output_name]}",
        metadata =  {"id": row['index']})
    
    


# if first_run:
        
#     faiss_index = FAISS.from_documents(df.apply(collate_with_metadata, axis=1),instructor_embeddings)
#     faiss_index.save_local("lavita_qa_vdb.vdb")
# else:
#     faiss_index = FAISS.load_local("lavita_qa_vdb.vdb", instructor_embeddings)

# faiss_index.similarity_search("headache", k=3)

import os

def get_any_case_df_as_vdb(df= "custom_vdb.vdb",embeddings = instructor_embeddings,
                           collate_func = lazy_collate,
                        #    first_run = False,
                           db_path = "custom_vdb.vdb"):
    """
    Create or load a Vector Database (VDB) from a given DataFrame and embeddings.

    Args:
        df (pandas.DataFrame): The input DataFrame to be converted into a VDB.
        embeddings (Embeddings, optional): The embeddings object to be used for creating the VDB. Defaults to `instructor_embeddings`.
        collate_func (Callable, optional): The function to be used for collating the data from the DataFrame rows. Defaults to `lazy_collate`.
        db_path (str, optional): The path where the VDB should be saved or loaded from. Defaults to "custom_vdb.vdb".

    Returns:
        FAISS: The FAISS index representing the VDB.

    If the specified `db_path` does not exist, a new VDB is created from the input DataFrame `df` using the provided `embeddings` and `collate_func`.
    The resulting VDB is then saved to the specified `db_path`.

    If the `db_path` already exists, the existing VDB is loaded from the specified path using the provided `embeddings`.

    The function returns the FAISS index representing the loaded or created VDB.
    """
    if not os.path.exists(db_path):
        if not type(df) is str:

            faiss_index = FAISS.from_documents(df.apply(collate_func, axis=1),embeddings)
            faiss_index.save_local(db_path)
    else:
        faiss_index = FAISS.load_local(db_path, embeddings)
    return faiss_index


