# Change your import at the top to this:
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Then ensure your call looks exactly like this:
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
#video_id = "J5_-l7WIO_w" # only the ID, not full URL
try:
    #If you don’t care which language, this returns the “best” one
    api = YouTubeTranscriptApi()
    transcript_list = api.fetch("sCx4MqMiYRo", languages=["en"])
    #Flatten it to plain text
    transcript = " ".join(chunk.text for chunk in transcript_list)
    print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")
print(transcript_list)    
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
print(len(chunks))
print(chunks[27])
embeddings = OpenAIEmbeddings(model="openai/text-embedding-3-large")
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.get_by_ids(['2436bdb8-3f5f-49c6-8915-0c654c888700'])
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
print(retriever)
retriever.invoke('What is deepmind')
llm = ChatOpenAI(model="openai/gpt-3.5-turbo-instruct", temperature=0.2)
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)
question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)
print(retrieved_docs)
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
print(context_text)
final_prompt = prompt.invoke({"context": context_text, "question": question})
print(final_prompt)
answer = llm.invoke(final_prompt)
print(answer.content)
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text
parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})
parallel_chain.invoke('Why do the hosts advise against checking your phone immediately after waking up')
print(parallel_chain)
parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser
main_chain.invoke('Can you summarize the video')
print(main_chain)
