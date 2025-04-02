from textpreparers.TextPreparer import TextPreparer
from chunkers.BasicChunker import BasicChunker
from vectordbs.RedisVectorDB import RedisVectorDB
from prompts.BasicPrompt import BasicPrompt
from prompts.OutsideKnowledgePrompt import OutsideKnowledgePrompt
from prompts.StrictPrompt import StrictPrompt
from llms.OllamaLLM import OllamaLLM
from embeddors.MiniLMEmbeddor import MiniLMEmbeddor
from embeddors.MiniLM33Embeddor import MiniLM33Embeddor
from embeddors.NomicEmbeddor import NomicEmbeddor
from measured_ingest import measure_ingest
from measured_query import measure_query
from os.path import join
from json import dumps

# Text preparation steps, 
# chunk size & chunk overlap size, 
# the embedding model,
# the vector database,
# and the LLM are all held constant
text_preparer = TextPreparer(
    remove_whitespace=False,
    remove_punctuation=False,
    remove_stopwords=False,
    lemmatize=False
)
chunker = BasicChunker()
embeddor = NomicEmbeddor()
vector_db = RedisVectorDB(-1)
llm = OllamaLLM('llama3.2')

# The embedding model is varied
prompt_templates = {
    "basic-prompt": BasicPrompt(),
    "outside-knowledge-prompt": OutsideKnowledgePrompt(),
    "strict-prompt": StrictPrompt()
}

test_queries = [
    "List three differences between Redis and MongoDB.",
    "Name one way in which a linked-list is better than a continguous array.",
    "What is the name of the library that allows us to interact with a MongoDB in Python?",
    "What is the name of a query language for Neo4J?",
    "What is the difference between a B tree and a B+ tree?"
]

# Ingest the documents
measure_ingest(text_preparer=text_preparer, chunker=chunker, vector_db=vector_db, embeddor=embeddor)

results = {}
for prompt_template_name, prompt_template in prompt_templates.items():
    print(f'Testing the prompt template {prompt_template_name}...')
    stats = {}
    stats.update(
        measure_query(
            queries=test_queries, embeddor=embeddor, vector_db=vector_db, llm=llm,
            prompt = prompt_template
        )
    )
    results[prompt_template_name] = stats
    print(f'Done testing {prompt_template_name}.')

# Print results
print("Prompt Template\tQuery Time")
for prompt_template_name, result in results.items(): 
    with open(join(".", "llm_responses", f"test-prompt-{prompt_template_name}.json"), 'w') as file:
        file.write(dumps(result['responses']))
    print(f"{prompt_template_name}\t{result['query_time']}")