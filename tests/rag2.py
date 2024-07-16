from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS

prompt = """
Use the following pieces of context to answer the question at the end.

Context: {context}

Question: {question}

Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)
VERBOSE = False

loader = TextLoader("/home/amith/Downloads/text-result.txt")
data = loader.load()

# Instantiate the embedding model
embedder = HuggingFaceEmbeddings()

# Create the vector store
vector = FAISS.from_documents(data, embedder)
# Input
retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
# Define llm
llm = Ollama(model="llama3")

llm_chain = LLMChain(
    llm=llm,
    prompt=QA_CHAIN_PROMPT,
    callbacks=None,
    verbose=VERBOSE)

document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

combine_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
    document_prompt=document_prompt,
    callbacks=None,
)

qa = RetrievalQA(
    combine_documents_chain=combine_documents_chain,
    verbose=VERBOSE,
    retriever=retriever,
    return_source_documents=True,
)
with open('/home/amith/Documents/GitHub/llama3-playground/samples/questions.txt', 'r') as f:
    questions = f.read()

q_prompt_prefix = """
Below are the list of fields specified and beside each field is the key that needs to be used to represent the field in a short form.
Give me the fields and their values as JSON. Make sure to use the exactly the specified field keys.


"""

for q in questions.split("---PAGE-SEP---"):
    # for q in questions.split("\n"):
    q = q_prompt_prefix + q
    print('<--------------------------------')
    print(q)
    print('-------------------------------->')
    print(qa(q)["result"])
    print('=================================')
