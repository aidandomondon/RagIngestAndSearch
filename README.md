# Ollama RAG Ingest and Search

## Requirements

- [Ollama](https://ollama.com)
- [Docker](https://docs.docker.com/get-started/get-docker/)

## Setup

Make sure Python has certificate verification setup, running the command:
```
<directory-to-python-bundle>/"Install Certificates.command"
```
for example,
```
/Applications/"Python 3.13"/"Install Certificates.command"
```

Then, navigate to the root directory of this project.

Run the setup script.
On Mac/Linux:
```
bash setup.sh
```
On Windows (cmd):
```
setup_env.bat
```

## Running Experiments:

Experiments are run by:
1. Navigating to `src/`
2. Finding, in the table below, the entry that corresponds to the desired experiment and:
    1. Running the dependencies specified by the table entry (all experiments require [Redis-stack](#Redis-stack), [nomic-embed-text](#Nomic-embed-text), and [Llama3.2](#Llama3.2)).
    2. Running the Python script specified by the table entry.

| Experiment Variable(s) | Script | Dependencies |
| - | - | - |
| Chunk size, Chunk overlap size | test-chunking.py | [Redis-stack](#Redis-stack), [nomic-embed-text](#Nomic-embed-text), [Llama3.2](#Llama32)
| Text pre-processing technique | test-preprocessing.py | [Redis-stack](#Redis-stack), [nomic-embed-text](#Nomic-embed-text), [Llama3.2](#Llama32)
| Embedding Model | test-embedding-model.py | [Redis-stack](#Redis-stack), [nomic-embed-text](#Nomic-embed-text), [Llama3.2](#Llama32), [sentence-transformersall-MiniLM-L6-v2](#sentence-transformersall-MiniLM-L6-v2), [sentence-transformersall-mpnet-base-v2](#sentence-transformersall-mpnet-base-v2)
| Vector Database | test-db.py | [Redis-stack](#Redis-stack), [nomic-embed-text](#Nomic-embed-text), [Llama3.2](#Llama32), [Qdrant](#Qdrant)
| System Prompt | test-prompt.py | [Redis-stack](#Redis-stack), [nomic-embed-text](#Nomic-embed-text), [Llama3.2](#Llama32)
| LLM | test-llm.py | [Redis-stack](#Redis-stack), [nomic-embed-text](#Nomic-embed-text), [Llama3.2](#Llama32), [DeepSeek R1 14B](#DeepSeek-R1-14B), [DeepSeek R1 32B](#DeepSeek-R1-32B)

### More About Experiment Dependencies 

#### Redis-stack
For experiments that list Redis-stack as a dependency, you must run a Docker container from the Redis-stack image.
```
docker run -p 6379:6379 redis/redis-stack
```
#### Qdrant
For experiments that list Qdrant as a dependency, you must run a Docker container from the Qdrant image.
```
docker run -p 6333:6333 qdrant/qdrant
```

#### Nomic-embed-text
For experiments that list nomic-embed-text as a dependency, you must install nomic-embed-text on Ollama.
```
ollama pull nomic-embed-text
```
#### sentence-transformers/all-MiniLM-L6-v2
For experiments that list sentence-transformers/all-MiniLM-L6-v2 as a dependency, you must install sentence-transformers/all-MiniLM-L6-v2 on Ollama.
```
ollama pull sentence-transformers/all-MiniLM-L6-v2
```
#### sentence-transformers/all-MiniLM-L6-v2
For experiments that list sentence-transformers/all-mpnet-base-v2 as a dependency, you must install sentence-transformers/all-mpnet-base-v2 on Ollama.
```
ollama pull sentence-transformers/all-mpnet-base-v2
```

#### Llama3.2
For experiments that list Llama3.2 as a dependency, you must install and run Llama3.2 on Ollama.
```
ollama run llama3.2
```
#### DeepSeek R1 14B
For experiments that list DeepSeek R1 14B as a dependency, you must install and run DeepSeek R1 14B on Ollama.
```
ollama run deepseek-r1:14b
```
#### DeepSeek R1 32B
For experiments that list DeepSeek R1 32B as a dependency, you must install and run DeepSeek R1 32B on Ollama.
```
ollama run deepseek-r1:32b
```

## Source Code
- `src/ingest.py` - imports and processes PDF files in `./data` folder. Embeddings and associated information 
stored in Redis-stack
- `src/search.py` - simple question answering using 
