'''
===========================================
        Module: Util functions
===========================================
'''
import box
import yaml

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings,ModelScopeEmbeddings
from langchain.vectorstores import FAISS
from src.prompts import qa_template
from src.llm import build_llm

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def set_qa_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context', 'question'])
    return prompt


def build_retrieval_qa(llm, prompt, vectordb):
    # dbqa = RetrievalQA.from_chain_type(llm=llm,
    #     chain_type='stuff',
    #     retriever=vectordb.as_retriever(search_kwargs={'k': cfg.VECTOR_COUNT}),
    #     return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
    #     chain_type_kwargs={'prompt': prompt}
    #     )
    dbqa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key='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'),
        chain_type='stuff',
        retriever=vectordb.as_retriever(search_kwargs={'k': cfg.VECTOR_COUNT}),
        return_source_documents=cfg.RETURN_SOURCE_DOCUMENTS,
        chain_type_kwargs={'prompt': prompt}
        )
    return dbqa


def setup_dbqa():
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
    #                                    model_kwargs={'device': 'cpu'})

    model_id = "thomas/text2vec-large-chinese"
    embeddings = ModelScopeEmbeddings(model_id=model_id)
    vectordb = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings)
    llm = build_llm()
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)

    return dbqa
