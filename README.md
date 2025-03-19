# Ollama RAG Ingest and Search

## Setup

Make sure Python has certificate verification setup, running the command:
```
$ <directory-to-python-bundle>"Install Certificates.command"
```
for example,
```
$ /Applications/"Python3.13"/"Install Certificates.command"
```

Then, navigate to the root directory of this project.

Run the setup script.
```
$ bash setup.sh
```

## Prerequisites

- Ollama app set up ([Ollama.com](Ollama.com))
- Python with Ollama, Redis-py, and Numpy installed (`pip install ollama redis numpy`)
- Redis Stack running (Docker container is fine) on port 6379.  If that port is mapped to another port in 
Docker, change the port number in the creation of the Redis client in both python files in `src`.

## Source Code
- `src/ingest.py` - imports and processes PDF files in `./data` folder. Embeddings and associated information 
stored in Redis-stack
- `src/search.py` - simple question answering using 
