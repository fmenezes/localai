# Step-by-Step Guide: Building a Local Chatbot with Streamlit, LangChain, Ollama, and MongoDB Atlas

In this tutorial, we'll walk you through setting up a local environment to use MongoDB Atlas Search and local LLMs (Large Language Models) via Ollama. We'll create a simple chatbot using Streamlit, MongoDB, and Ollama, and demonstrate how to enhance user queries with context from chat history.

## Prerequisites
Before we begin, ensure you have the following installed:

* Docker
* Docker Compose

## Step 1: Setting Up the Project
First, create a new directory for your project and navigate into it:

```sh
mkdir localai
cd localai
```

### Project Structure
Your project structure should look like this:

```sh
localai/
├── app.py
├── Dockerfile
├── compose.yaml
└── requirements.txt
```

#### `requirements.txt`
Create a file named `requirements.txt` and add the following dependencies:
```requirements.txt
streamlit
ollama
langchain
langchain_ollama
pymongo
langchain_mongodb
langchain_community
markdownify
```

#### `Dockerfile`
Create a file named `Dockerfile` and add the following content:

```Dockerfile
FROM python:3.12
WORKDIR /opt/app
ADD requirements.txt .
RUN pip install -r requirements.txt
ADD app.py .

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### `compose.yaml`
Create a file named compose.yaml and add the following content:

```yaml
services:
  app:
    build:
      context: .
    ports:
      - 8501:8501/tcp
    environment:
      MONGO_URI: mongodb://root:root@mongo:27017/admin?directConnection=true
  ollama:
    image: ollama/ollama
  mongo:
    image: mongodb/mongodb-atlas-local
    environment:
      - MONGODB_INITDB_ROOT_USERNAME=root
      - MONGODB_INITDB_ROOT_PASSWORD=root
    ports:
       - 27017:27017
```

**Note:** if running on macOs it is recommended to install ollama locally and use this modified version of compose.yaml
```yaml
services:
  app:
    build:
      context: .
    ports:
      - 8501:8501/tcp
    environment:
      OLLAMA_HOST: host.docker.internal:11434
      MONGO_URI: mongodb://root:root@mongo:27017/admin?directConnection=true
    extra_hosts:
      - "host.docker.internal:host-gateway"
  mongo:
    image: mongodb/mongodb-atlas-local
    environment:
      - MONGODB_INITDB_ROOT_USERNAME=root
      - MONGODB_INITDB_ROOT_PASSWORD=root
    ports:
       - 27017:27017
```

#### `app.py`
Create a file named `app.py` and add the following code:

```python
import streamlit as st

st.write('hello world')
```

## Step 2: Building and Running the Docker Containers
With all the files in place, you can now build and run the Docker containers.

Run the following command to build and run the Docker containers:

```sh
docker compose up
```

This will start your app, you can access it by navigating to `http://localhost:8501` in your browser.

You should see a hello world message on the screen.

## Step 3: Incrementally Building the Chatbot

Now, let's incrementally build the chatbot by updating app.py step by step.

### Step 3.1: Setting Up MongoDB and Ollama

First, let's set up MongoDB and Ollama in our `app.py`:

```python
import os
import streamlit as st
from pymongo import MongoClient
import ollama

# Model and embedding configurations
MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

# Pull models from Ollama
ollama.pull(MODEL)
ollama.pull(EMBEDDING_MODEL)

# Initialize MongoDB client
try:
    mongo_client = MongoClient(MONGO_URI)
    collection = mongo_client["bot"]["data"]
except Exception as e:
    st.error(f"Failed to connect to MongoDB: {e}")
    st.stop()
```

To test at each step run docker with `docker compose up --build`

### Step 3.2: Loading Documents and Creating Vector Search Index

Next, we'll load documents and create a vector search index if not already present:

```python
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import MarkdownifyTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch

# Initialize embeddings
embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

collection.drop()

# Load documents and create vector search index if not already present
loaders = [
    WebBaseLoader("https://en.wikipedia.org/wiki/AT%26T"),
    WebBaseLoader("https://en.wikipedia.org/wiki/Bank_of_America")
]
docs = []
for loader in loaders:
    for doc in loader.load():
        docs.append(doc)
md = MarkdownifyTransformer()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
converted_docs = md.transform_documents(docs)
splits = text_splitter.split_documents(converted_docs)
vectorstore = MongoDBAtlasVectorSearch.from_documents(splits, embedding, collection=collection, index_name="default")
vectorstore.create_vector_search_index(768)
```

### Step 3.3: Setting Up the Chat Model

Now, let's set up the chat model:

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_mongodb import MongoDBChatMessageHistory

# Initialize retriever and chat model
retriever = vectorstore.as_retriever()
chat = ChatOllama(model=MODEL)

# Define prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MESSAGE),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

# Define the chain of operations
chain = {
    "context": itemgetter("input") | retriever,
    "input": itemgetter("input"),
    "history": itemgetter("history")
} | prompt_template | chat | StrOutputParser()

# Function to get session history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return MongoDBChatMessageHistory(MONGO_URI, session_id, database_name="bot")

# Initialize history chain
history_chain = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="input", history_messages_key="history")
```

### Step 3.4: Creating the Chat Interface

Next, we'll create the chat interface using Streamlit:

```python
# Streamlit UI
st.title("Chatbot")
st.caption("A Streamlit chatbot")

# Display chat history
history = get_session_history()
for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)

# Handle user input
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            st.write_stream(history_chain.stream({"input": prompt}))
```

At this point you can start prompting like `Who started AT&T?`.

## Conclusion
In this tutorial, we demonstrated how to set up a local environment to use MongoDB Atlas Search and local LLMs via Ollama. We created a simple chatbot using Streamlit, MongoDB, and Ollama, and enhanced user queries with context from chat history. This setup allows you to test and develop your applications locally before deploying them to a production environment.

Feel free to customize and expand this project to suit your needs. Happy coding!

You can see the full code of this project at https://github.com/fmenezes/localai.
