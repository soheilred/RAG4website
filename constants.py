OUTPUT_FILE = "eluvio_index"
DB_FILE = "DB_VEC"
DOCS_BASE_URL = "https://docs.eluv.io"
BASE_URL = "eluv.io"
START_URLS = ["https://eluv.io/"]
LANGUAGE_MODEL = "llama3.2:3b"
# EMBEDDING_MODEL = "intfloat/e5-small-v2" #
EMBEDDING_MODEL = "BAAI/bge-m3"
JSONFILE = 'eluvio_content.jsonl'
SKIP_EXTENSIONS = ('.sh', 'bin', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.zip', '.mp4')
BANNED_SUBSTRINGS = ["advanced/playout", "privacy", "terms"]
PDF_DIR = DB_FILE + '/pdfs'

