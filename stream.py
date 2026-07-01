import re
import streamlit as st

from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="wide",
)

st.title("🎥 YouTube Video Chatbot")
st.write("Ask questions about any YouTube video using RAG + OpenAI + FAISS.")

# --------------------------
# Extract Video ID
# --------------------------
def extract_video_id(url):
    patterns = [
        r"(?:v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    if len(url) == 11:
        return url

    return None


# --------------------------
# Download Transcript
# --------------------------
def get_transcript(video_id):

    api = YouTubeTranscriptApi()

    transcript = api.fetch(
        video_id,
        languages=["en"],
    )

    text = " ".join(chunk.text for chunk in transcript)

    return text


# --------------------------
# Build Vector DB
# --------------------------
@st.cache_resource(show_spinner=False)
def create_vector_store(transcript):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    docs = splitter.create_documents([transcript])

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

    db = FAISS.from_documents(
        docs,
        embeddings,
    )

    return db


# --------------------------
# Prompt
# --------------------------
prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Answer ONLY using the transcript below.

If the answer is not available in the transcript,
say:

"I couldn't find that information in the video transcript."

Transcript:
{context}

Question:
{question}
""",
    input_variables=["context", "question"],
)


# --------------------------
# Format Docs
# --------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# --------------------------
# Sidebar
# --------------------------
st.sidebar.header("Video")

video_url = st.sidebar.text_input(
    "Enter YouTube URL"
)

load_button = st.sidebar.button("Load Video")


if load_button:

    video_id = extract_video_id(video_url)

    if video_id is None:
        st.error("Invalid YouTube URL")
        st.stop()

    with st.spinner("Downloading transcript..."):

        try:

            transcript = get_transcript(video_id)

            db = create_vector_store(transcript)

            retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4},
            )

            llm = ChatOpenAI(
                model="gpt-4.1-mini",
                temperature=0.2,
            )

            parallel_chain = RunnableParallel(
                {
                    "context": retriever
                    | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
            )

            parser = StrOutputParser()

            chain = (
                parallel_chain
                | prompt
                | llm
                | parser
            )

            st.session_state.chain = chain

            st.success("Video Loaded Successfully!")

            st.video(video_url)

        except TranscriptsDisabled:
            st.error("Transcript is disabled for this video.")

        except NoTranscriptFound:
            st.error("No English transcript found.")

        except Exception as e:
            st.error(str(e))


# --------------------------
# Chat Section
# --------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if "chain" in st.session_state:

    question = st.chat_input(
        "Ask anything about the video..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                answer = st.session_state.chain.invoke(
                    question
                )

                st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )