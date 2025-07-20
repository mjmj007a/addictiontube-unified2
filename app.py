from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from openai import OpenAI, APIError
import weaviate
from weaviate.auth import AuthApiKey
import os
import re
import json
import logging
from logging.handlers import RotatingFileHandler
import tiktoken
from dotenv import load_dotenv
import random
import nltk
import time
import contextlib
import timeout_decorator

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://addictiontube.com", "http://addictiontube.com"]}})

logger = logging.getLogger('addictiontube')
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler('/tmp/unified_search.log', maxBytes=10485760, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logger.addHandler(handler)

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
    headers_enabled=True,
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_CLUSTER_URL = os.getenv("WEAVIATE_CLUSTER_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

if not all([OPENAI_API_KEY, WEAVIATE_CLUSTER_URL, WEAVIATE_API_KEY]):
    logger.error("Missing one or more required environment variables.")
    raise EnvironmentError("Missing one or more required environment variables.")

try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    raise

weaviate_client = None
for attempt in range(3):
    try:
        weaviate_client = weaviate.connect_to_wcs(
            cluster_url=WEAVIATE_CLUSTER_URL,
            auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
            headers={"X-OpenAI-Api-Key": OPENAI_API_KEY},
            skip_init_checks=True
        )
        logger.info("Weaviate client initialized successfully")
        break
    except Exception as e:
        logger.error(f"Weaviate client initialization attempt {attempt + 1} failed: {str(e)}")
        time.sleep(2)
if weaviate_client is None:
    logger.error("Failed to initialize Weaviate client after 3 attempts")
    raise EnvironmentError("Weaviate client initialization failed")

try:
    with open('songs_revised_with_songs-july06.json', 'r', encoding='utf-8') as f:
        song_dict = {item['video_id']: item['song'] for item in json.load(f)}
    with open('videos_revised_with_poems-july04.json', 'r', encoding='utf-8') as f:
        poem_dict = {item['video_id']: item['poem'] for item in json.load(f)}
    with open('stories.json', 'r', encoding='utf-8') as f:
        story_dict = {item['id']: item['text'] for item in json.load(f)}
    logger.info("Content dictionaries loaded successfully")
except Exception as e:
    logger.error(f"Failed to load content dictionaries: {str(e)}")
    raise

lemmatizer = None
try:
    nltk.download('wordnet', quiet=True, raise_on_error=True)
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    logger.info("NLTK WordNetLemmatizer initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize NLTK WordNetLemmatizer: {str(e)}. Falling back to no lemmatization.")

def strip_html(text):
    return re.sub(r'<[^>]+>', '', text or '') if text else ''

def preprocess_query(query):
    if lemmatizer:
        words = query.lower().split()
        lemmatized = [lemmatizer.lemmatize(word, pos='n') for word in words]
        processed = ' '.join(lemmatized)
        logger.info(f"Processed query: {query} -> {processed}")
        return processed
    logger.warning(f"No lemmatizer available, using raw query: {query}")
    return query.lower()

@contextlib.contextmanager
def weaviate_connection():
    try:
        yield weaviate_client
    finally:
        if weaviate_client:
            weaviate_client.close()
            logger.info("Weaviate client connection closed")

@app.route('/', methods=['GET', 'HEAD'])
def health_check():
    logger.info("Health check endpoint accessed")
    return jsonify({"status": "ok", "message": "AddictionTube Unified API is running"}), 200

@app.errorhandler(429)
def ratelimit_handler(e):
    logger.warning(f"Rate limit exceeded: {e.description}")
    return jsonify({
        "error": "Rate limit exceeded",
        "details": f"Too many requests. Please wait and try again. Limit: {e.description}"
    }), 429

@timeout_decorator.timeout(60, timeout_exception=TimeoutError)
@app.route('/search_content', methods=['GET'])
@limiter.limit("60 per hour")
def search_content():
    query = re.sub(r'[^\w\s.,!?]', '', request.args.get('q', '')).strip()
    content_type = request.args.get('content_type', '').strip()
    category = request.args.get('category', 'all').strip()
    page = max(1, int(request.args.get('page', 1)))
    size = max(1, min(100, int(request.args.get('per_page', 5))))

    logger.info(f"Search request: query={query}, content_type={content_type}, category={category}, page={page}, per_page={size}")

    if not query or content_type not in ['songs', 'poems', 'stories']:
        logger.warning(f"Invalid request: query={query}, content_type={content_type}")
        return jsonify({"error": "Invalid or missing query or content type"}), 400

    try:
        start_time = time.time()
        processed_query = preprocess_query(query)
        embedding = client.embeddings.create(input=processed_query, model="text-embedding-3-small")
        vector = embedding.data[0].embedding

        with weaviate_connection():
            collection = weaviate_client.collections.get("Content")
            from weaviate.classes.query import Filter
            filters = Filter.by_property("type").equal(content_type)

            results = collection.query.near_vector(
                near_vector=vector,
                limit=size * page + 10,
                filters=filters,
                return_metadata=["distance"]
            )

            paginated = results.objects[(page - 1) * size: page * size]
            items = []
            for obj in paginated:
                item = {
                    "id": obj.properties.get("content_id", str(obj.uuid)),
                    "score": obj.metadata.distance,
                    "title": strip_html(obj.properties.get("title", "N/A")),
                    "description": strip_html(obj.properties.get("description", "")),
                    "image": obj.properties.get("image", "") if content_type == 'stories' else ""
                }
                items.append(item)

        response = {"results": items, "total": len(results.objects)}
        logger.info(f"Search success: {len(items)} items returned, total={len(results.objects)}, time={time.time() - start_time:.2f}s")
        return jsonify(response)
    except TimeoutError:
        logger.error("Search timed out after 60 seconds")
        return jsonify({"error": "Search timed out", "details": "Request took too long to process"}), 504
    except Exception as e:
        logger.error(f"Weaviate search error: {str(e)}")
        return jsonify({"error": "Search failed", "details": str(e)}), 500

@timeout_decorator.timeout(60, timeout_exception=TimeoutError)
@app.route('/rag_answer_content', methods=['GET'])
@limiter.limit("30 per hour")
def rag_answer_content():
    query = re.sub(r'[^\w\s.,!?]', '', request.args.get('q', '')).strip()
    content_type = request.args.get('content_type', '').strip()
    category = request.args.get('category', 'all').strip()
    reroll = request.args.get('reroll', '').lower().startswith('yes')

    logger.info(f"RAG request: query={query}, content_type={content_type}, category={category}, reroll={reroll}")

    if not query or content_type not in ['songs', 'poems', 'stories']:
        logger.warning(f"Invalid RAG request: query={query}, content_type={content_type}")
        return jsonify({"error": "Invalid or missing query or content type"}), 400

    try:
        start_time = time.time()
        processed_query = preprocess_query(query)
        embedding = client.embeddings.create(input=processed_query, model="text-embedding-3-small")
        vector = embedding.data[0].embedding

        with weaviate_connection():
            collection = weaviate_client.collections.get("Content")
            from weaviate.classes.query import Filter
            filters = Filter.by_property("type").equal(content_type)

            results = collection.query.near_vector(
                near_vector=vector,
                limit=10,
                filters=filters,
                return_metadata=["distance"]
            )

            matches = results.objects
            if reroll:
                random.shuffle(matches)

        content_dict = {'songs': song_dict, 'poems': poem_dict, 'stories': story_dict}
        encoding = tiktoken.get_encoding("cl100k_base")
        max_tokens = 16384 - 1000
        context_docs = []
        total_tokens = 0

        for obj in matches:
            doc = strip_html(content_dict[content_type].get(obj.properties.get("content_id", str(obj.uuid)), obj.properties.get("description", "")))[:3000]
            token_len = len(encoding.encode(doc))
            if total_tokens + token_len <= max_tokens:
                context_docs.append(doc)
                total_tokens += token_len
            else:
                break

        if not context_docs:
            logger.warning(f"No usable context data found for query={query}, content_type={content_type}")
            return jsonify({"error": "No usable context data found"}), 404

        context_text = "\n\n---\n\n".join(context_docs)
        system_prompt = f"You are an expert assistant for addiction recovery {content_type}."
        user_prompt = f"""Use the following {content_type} to answer the question.\n\n{context_text}\n\nQuestion: {query}\nAnswer:"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1000
        )
        answer = response.choices[0].message.content
        logger.info(f"RAG success: Answer generated for query={query}, content_type={content_type}, time={time.time() - start_time:.2f}s")
        return jsonify({"answer": answer})
    except TimeoutError:
        logger.error("RAG timed out after 60 seconds")
        return jsonify({"error": "RAG timed out", "details": "Request took too long to process"}), 504
    except Exception as e:
        logger.error(f"RAG failed: {str(e)}")
        return jsonify({"error": "RAG processing failed", "details": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)