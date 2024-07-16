from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS

urls = [
    # "https://www.indiatoday.in/horoscopes/story/horoscope-today-july-16-aries-taurus-gemini-cancer-leo-virgo-libra-scorpio-sagittarius-capricorn-aquarius-pisces-2567053-2024-07-16",
    "https://timesofindia.indiatimes.com/astrology/horoscope/pisces-daily-horoscope-today-july-16-2024-stay-focused-and-positive-to-navigate-through-obstacle/articleshow/111761492.cms"
]

loader = SeleniumURLLoader(urls=urls)
data = loader.load()
print(data)

# Instantiate the embedding model
embedder = HuggingFaceEmbeddings()

# Create the vector store
vector = FAISS.from_documents(data, embedder)
# Input
retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
# Define llm
llm = Ollama(model="llama3")

prompt = """
1. Use the following pieces of context to answer the question at the end.
2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
3. Keep the answer crisp and limited to 3, 4 sentences.
4. If possible, format the data neatly by using bullet points, etc

Context: {context}

Question: {question}

Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

llm_chain = LLMChain(
    llm=llm,
    prompt=QA_CHAIN_PROMPT,
    callbacks=None,
    verbose=False)

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
    verbose=False,
    retriever=retriever,
    return_source_documents=True,
)

question = 'Give me pisces horoscope for today. Summarize it with bullet points and give me details in Career, Health, Lucky Colors, Numbers, etc'
print(qa(question)["result"])
