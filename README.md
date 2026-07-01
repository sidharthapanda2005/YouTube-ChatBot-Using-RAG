# 🎥 YouTube Video Chatbot (RAG + OpenAI + FAISS + Streamlit)

A Retrieval-Augmented Generation (RAG) application that allows users to chat with any YouTube video by using its transcript.

The application automatically downloads the transcript of a YouTube video, converts it into embeddings using OpenAI Embeddings, stores those embeddings in a FAISS vector database, retrieves the most relevant transcript chunks for a user's question, and generates an answer using OpenAI GPT.

---

# 🚀 Features

- Extracts YouTube Video ID from multiple URL formats
- Downloads English transcript automatically
- Splits transcript into semantic chunks
- Creates vector embeddings using OpenAI
- Stores embeddings inside FAISS
- Retrieves relevant transcript chunks
- Answers questions using GPT-4.1 Mini
- Interactive Streamlit Chat UI
- Conversation history
- Error handling for missing transcripts

---

# 🏗️ Tech Stack

- Python
- Streamlit
- LangChain
- OpenAI
- FAISS
- YouTube Transcript API
- python-dotenv

---

# 📁 Project Structure

```
project/
│
├── app.py
├── .env
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## Step 1

Clone the repository

```bash
git clone https://github.com/yourusername/youtube-rag-chatbot.git

cd youtube-rag-chatbot
```

---

## Step 2

Create Virtual Environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Mac/Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Step 3

Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4

Create a `.env` file

```
OPENAI_API_KEY=your_openai_api_key
```

---

## Step 5

Run the application

```bash
streamlit run app.py
```

---

# 📚 Required Libraries

```python
import re
import streamlit as st

from dotenv import load_dotenv
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)

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
```

These libraries provide:

| Library | Purpose |
|----------|----------|
| re | Extract Video ID |
| Streamlit | Web UI |
| dotenv | Load API Keys |
| youtube_transcript_api | Download transcript |
| RecursiveCharacterTextSplitter | Split transcript |
| OpenAIEmbeddings | Generate embeddings |
| FAISS | Vector Database |
| ChatOpenAI | GPT Model |
| PromptTemplate | Create Prompt |
| RunnableParallel | LangChain Pipeline |
| RunnableLambda | Format retrieved documents |
| RunnablePassthrough | Pass user question |
| StrOutputParser | Convert LLM output to string |

---

# 🏗️ Code Walkthrough

---

# 1. Load Environment Variables

```python
load_dotenv()
```

Loads environment variables stored inside the `.env` file.

Example

```
OPENAI_API_KEY=xxxxxxxxxxxxxxxx
```

This allows the OpenAI API key to remain secure instead of hardcoding it.

---

# 2. Configure Streamlit

```python
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="wide",
)
```

This customizes the Streamlit application.

- Page title
- Browser icon
- Wide layout

---

# 3. Application Title

```python
st.title(...)
st.write(...)
```

Displays the heading and description on the webpage.

---

# 4. Extract YouTube Video ID

```python
def extract_video_id(url):
```

This function extracts the unique 11-character YouTube video ID.

Supported URL formats:

```
https://www.youtube.com/watch?v=xxxx

https://youtu.be/xxxx

https://www.youtube.com/embed/xxxx
```

If only the video ID is entered, it also accepts that.

Returns

```
Video ID
```

or

```
None
```

---

# 5. Download Transcript

```python
def get_transcript(video_id):
```

Creates a YouTube Transcript API object.

```python
api = YouTubeTranscriptApi()
```

Downloads the English transcript.

```python
api.fetch(
video_id,
languages=["en"]
)
```

Each transcript is returned as multiple chunks.

Example

```
Hello

Welcome

Today we'll discuss AI...
```

These chunks are combined into one large string.

```python
text = " ".join(chunk.text for chunk in transcript)
```

Returns

```
Entire transcript
```

---

# 6. Create Vector Database

```python
@st.cache_resource
```

Caching prevents rebuilding embeddings every time the page refreshes.

---

```python
RecursiveCharacterTextSplitter
```

Splits long transcript into smaller overlapping chunks.

Parameters

```
chunk_size = 1000

chunk_overlap = 200
```

Example

```
Chunk 1

Chunk 2

Chunk 3
```

---

Embeddings

```python
OpenAIEmbeddings(
model="text-embedding-3-large"
)
```

Converts every chunk into a high-dimensional vector.

---

FAISS

```python
FAISS.from_documents(...)
```

Stores all vectors for fast similarity search.

Returns

```
FAISS Vector Database
```

---

# 7. Prompt Template

```python
PromptTemplate(...)
```

Creates the instruction given to GPT.

Prompt

```
Answer ONLY using transcript.

If answer is unavailable,

say

"I couldn't find that information..."
```

This prevents hallucinations.

---

# 8. Format Retrieved Documents

```python
format_docs(docs)
```

Converts retrieved LangChain Documents into plain text.

```
doc1

doc2

doc3
```

↓

```
Single String
```

---

# 9. Sidebar

```python
st.sidebar
```

Creates a sidebar containing

- YouTube URL input
- Load button

---

# 10. Load Video

When the user clicks

```
Load Video
```

The following steps occur:

### Step 1

Extract Video ID

↓

### Step 2

Download Transcript

↓

### Step 3

Create Embeddings

↓

### Step 4

Build FAISS

↓

### Step 5

Create Retriever

↓

### Step 6

Initialize GPT

↓

### Step 7

Build LangChain Pipeline

↓

### Step 8

Store Chain inside Session State

↓

Display Success Message

---

# 11. Retriever

```python
retriever = db.as_retriever(
search_type="similarity",
search_kwargs={"k":4}
)
```

When a question is asked,

FAISS searches for the

Top 4 most similar transcript chunks.

---

# 12. OpenAI Model

```python
ChatOpenAI(
model="gpt-4.1-mini",
temperature=0.2
)
```

Temperature

```
0.2
```

Produces more deterministic and factual responses.

---

# 13. RunnableParallel

```python
RunnableParallel(...)
```

Runs two tasks simultaneously.

Task 1

Retrieve transcript

Task 2

Pass user question

Output

```
{
context: ...

question: ...
}
```

---

# 14. LangChain Pipeline

```python
chain = (

parallel_chain

|

prompt

|

llm

|

parser

)
```

Pipeline Flow

```
User Question

↓

Retriever

↓

Prompt

↓

GPT

↓

Output Parser

↓

Answer
```

---

# 15. Session State

```python
st.session_state.chain
```

Stores the pipeline.

Avoids rebuilding the entire chain after every interaction.

---

# 16. Chat History

```python
st.session_state.messages
```

Stores conversation history.

Each message contains

```
Role

Content
```

Example

```
User

Assistant

User

Assistant
```

---

# 17. Chat Input

```python
st.chat_input(...)
```

Waits for the user's question.

---

# 18. Generate Answer

When the user asks a question

```python
answer = chain.invoke(question)
```

The pipeline performs

```
Question

↓

Similarity Search

↓

Retrieve Context

↓

Prompt GPT

↓

Generate Answer

↓

Display Answer
```

---

# 🔄 Complete Workflow

```
User

↓

Paste YouTube URL

↓

Extract Video ID

↓

Download Transcript

↓

Split Transcript

↓

Generate Embeddings

↓

Store in FAISS

↓

Ask Question

↓

Similarity Search

↓

Retrieve Context

↓

Prompt GPT

↓

Generate Answer

↓

Display Response
```

---

# 🛡️ Error Handling

The application handles several common exceptions:

| Exception | Description |
|------------|-------------|
| Invalid URL | User enters an incorrect YouTube URL |
| TranscriptsDisabled | The video owner has disabled transcripts |
| NoTranscriptFound | No English transcript is available |
| General Exception | Displays any unexpected error |

---

# 📸 Future Improvements

- Support multilingual transcripts
- Chat memory across multiple videos
- Timestamp-based answers
- PDF transcript export
- Whisper fallback for videos without transcripts
- Hybrid search (BM25 + Vector Search)
- Support local embedding models
- Deploy using Streamlit Cloud

---

# 👨‍💻 Author

Your Name

GitHub: https://github.com/yourusername

---

# ⭐ If you like this project

Give this repository a ⭐ on GitHub!
