DATA_DIR_PATH = "data/sustainability_reports"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 200
EMBEDDER = "BAAI/bge-base-en-v1.5"
DEVICE = "cpu"
PROMPT_TEMPLATE = '''
Answer the question based only on the following context:
{context}

Question: {question}
Do provide only helpful answers

Helpful answer:
'''
INP_VARS = ['context', 'question']
CHAIN_TYPE = "stuff"
SEARCH_KWARGS = {'k': 2}
MODEL_CKPT = "res/mistral-7b-openorca.Q8_0.gguf"
MODEL_TYPE = "llama"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.3
