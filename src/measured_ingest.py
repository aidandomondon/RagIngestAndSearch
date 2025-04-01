from subprocess import run
from json import loads
from Ingestor import Ingestor
from textpreparers.TextPreparer import TextPreparer
from chunkers.Chunker import Chunker
from embeddors.Embeddor import Embeddor
from vectordbs.VectorDB import VectorDB
from time import perf_counter_ns

def _get_local_docker_usage(container_name: str) -> str:
    """
    Returns the memory usage statistic of the Docker container with the given name.
    Assumes the Docker container is running on this machine.
    """
    command = [
        'docker', 'container', 'stats', 
        container_name, 
        '--no-stream', 
        '--format', 'json'
    ]
    command_result = run(command, capture_output=True)
    stats = loads(command_result.stdout.decode())
    memory_usage = stats['MemUsage']
    return memory_usage

def measure_ingest(
    text_preparer: TextPreparer, 
    chunker: Chunker, 
    embeddor: Embeddor, 
    vector_db: VectorDB,
    docker_container_name: str = "DS4300-redis-stack"
) -> dict:
    """
    Returns the time (in milliseconds) it takes to ingest 
    all documents in `./data` with the given settings.
    """
    # Clean/reset and prepare the database for a fresh test
    vector_db.clean_and_reinit(embeddor.get_dimension())
    
    ingestor = Ingestor(text_preparer, chunker, embeddor, vector_db)

    start = perf_counter_ns()
    ingestor.ingest('../data/')
    end = perf_counter_ns()
    ingestion_time = int((end - start) / 1e6)

    # memory_usage = _get_local_docker_usage(docker_container_name)
    memory_usage = vector_db.memory_usage()

    return {
        "ingestion_time": ingestion_time, 
        "memory_usage": memory_usage
    }