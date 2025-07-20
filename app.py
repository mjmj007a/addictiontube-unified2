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
from tenacity import retry, stop_after_attempt, wait_exponential

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

def get_client():
    for attempt in range(3):
        try:
            weaviate_client = weaviate.Client(
                url=WEAVIATE_CLUSTER_URL,
                auth_client_secret=AuthApiKey(WEAVIATE_API_KEY),
                additional_headers={"X-OpenAI-Api-Key": OPENAI_API_KEY}
            )
            if weaviate_client.is_ready():
                logger.info("Weaviate client initialized and ready")
                return weaviate_client
            else:
                logger.warning("Weaviate client initialized but not ready")
                weaviate_client.close()
        except Exception as e:
            logger.error(f"Weaviate client initialization attempt {attempt + 1} failed: {str(e)}", exc_info=True)
            time.sleep(2)
    logger.error("Failed to initialize Weaviate client after 3 attempts")
    raise EnvironmentError("Weaviate client initialization failed")

try:
    with open('songs_revised_with_songs-july06.json', 'r', encoding='utf-8') as f:
        song_dict = {item['video_id']: item['song'] for item in json.load(f)}
    with open('videos_revised_with_poems-july04.json', 'r', encoding='utf-8') as f:
        poem_dict = {item['video_id']: item['poem'] for item in json.load(f)}
    with open('stories.json', 'r', encoding='utf-8') as f:
        story_dict = {item['id']: item['text'] for item in json.load(f)}
    logger.info(f"Loaded dictionaries: songs={len(song_dict)}, poems={len(poem_dict)}, stories={len(story_dict)}")
except Exception as e:
    logger.error(f"Failed to load content dictionaries: {str(e)}", exc_info=True)
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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def get_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    logger.info(f"OpenAI embedding headers: {response.headers}")
    return response.data[0].embedding

@app.route('/', methods=['GET', 'HEAD'])
def health_check():
    logger.info("Health check endpoint accessed")
    client = get_client()
    try:
        if not client.is_ready():
            logger.warning("Weaviate client not ready")
            return jsonify({"error": "Weaviate client not ready", "details": "Client initialized but not ready"}), 503
        try:
            embedding = get_embedding("test")
            logger.info("OpenAI embedding test successful")
        except APIError as e:
            logger.error(f"OpenAI health check failed: {str(e)}", exc_info=True)
            return jsonify({"error": "OpenAI health check failed", "details": str(e)}), 503
        collections = client.collections.list_all()
        logger.info(f"Weaviate collections: {collections}")
        return jsonify({"status": "ok", "message": "AddictionTube Unified API is running", "weaviate_collections": list(collections.keys())}), 200
    except Exception as e:
        logger.error(f"Weaviate health check failed: {str(e)}", exc_info=True)
        return jsonify({"error": "Weaviate health check failed", "details": str(e)}), 503
    finally:
        client.close()
        logger.info("Weaviate client closed after health check")

@app.errorhandler(429)
def ratelimit_handler(e):
    logger.warning(f"Rate limit exceeded: {e.description}, remote_addr={get_remote_address()}")
    return jsonify({
        "error": "Rate limit exceeded",
        "details": f"Too many requests. Please wait and try again. Limit: {e.description}"
    }), 429

@app.route('/search_content', methods=['GET'])
@limiter.limit("60 per hour")
def search_content():
    query = re.sub(r'[^\w\s.,!?]', '', request.args.get('q', '')).strip()
    content_type = request.args.get('content_type', '').strip()
    page = max(1, int(request.args.get('page', 1)))
    size = max(1, min(100, int(request.args.get('per_page', 5))))

    logger.info(f"Search request: query={query}, content_type={content_type}, page={page}, per_page={size}")

    if not query or content_type not in ['songs', 'poems', 'stories']:
        logger.warning(f"Invalid request: query={query}, content_type={content_type}")
        return jsonify({"error": "Invalid or missing query or content type"}), 400

    client = get_client()
    try:
        start_time = time.time()
        processed_query = preprocess_query(query)
        try:
            vector = get_embedding(processed_query)
        except APIError as e:
            logger.error(f"OpenAI embedding failed: {str(e)}", exc_info=True)
            return jsonify({"error": "Embedding generation failed", "details": str(e)}), 500

        try:
            collection = client.collections.get("Content")
            from weaviate.classes.query import Filter
            filters = Filter.by_property("type").equal(content_type)
            results = collection.query.near_vector(
                near_vector=vector,
                limit=size * page + 10,
                filters=filters,
                return_metadata=["distance"]
            )
        except Exception as e:
            logger.error(f"Weaviate query failed: {str(e)}", exc_info=True)
            return jsonify({"error": "Weaviate query failed", "details": str(e)}), 500

        content_dict = {'songs': song_dict, 'poems': poem_dict, 'stories': story_dict}
        paginated = results.objects[(page - 1) * size:page * size]
        items = []
        for obj in paginated:
            content_id = obj.properties.get("content_id", str(obj.uuid))
            if content_id not in content_dict.get(content_type, {}):
                logger.warning(f"Content ID {content_id} not found in {content_type} dictionary")
            item = {
                "id": content_id,
                "score": obj.metadata.distance,
                "title": strip_html(obj.properties.get("title", "N/A")),
                "description": strip_html(obj.properties.get("description", "")),
                "image": obj.properties.get("image", "") if content_type == 'stories' else ""
            }
            items.append(item)

        response = {"results": items, "total": len(results.objects)}
        logger.info(f"Search success: {len(items)} items returned, total={len(results.objects)}, time={time.time() - start_time:.2f}s")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Unexpected error in search_content: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error", "details": str(e)}), 500
    finally:
        client.close()
        logger.info("Weaviate client closed after search_content")

@app.route('/rag_answer_content', methods=['GET'])
@limiter.limit("30 per hour")
def rag_answer_content():
    query = re.sub(r'[^\w\s.,!?]', '', request.args.get('q', '')).strip()
    content_type = request.args.get('content_type', '').strip()
    reroll = request.args.get('reroll', '').lower().startswith('yes')

    logger.info(f"RAG request: query={query}, content_type={content_type}, reroll={reroll}")

    if not query or content_type not in ['songs', 'poems', 'stories']:
        logger.warning(f"Invalid RAG request: query={query}, content_type={content_type}")
        return jsonify({"error": "Invalid or missing query or content type"}), 400

    client = get_client()
    try:
        start_time = time.time()
        processed_query = preprocess_query(query)
        try:
            vector = get_embedding(processed_query)
        except APIError as e:
            logger.error(f"OpenAI embedding failed: {str(e)}", exc_info=True)
            return jsonify({"error": "Embedding generation failed", "details": str(e)}), 500

        try:
            collection = client.collections.get("Content")
            from weaviate.classes.query import Filter
            filters = Filter.by_property("type").equal(content_type)
            results = collection.query.near_vector(
                near_vector=vector,
                limit=10,
                filters=filters,
                return_metadata=["distance"]
            )
        except Exception as e:
            logger.error(f"Weaviate query failed: {str(e)}", exc_info=True)
            return jsonify({"error": "Weaviate query failed", "details": str(e)}), 500

        matches = results.objects
        if reroll:
            random.shuffle(matches)

        content_dict = {'songs': song_dict, 'poems': poem_dict, 'stories': story_dict}
        encoding = tiktoken.get_encoding("cl100k_base")
        max_tokens = 16384 - 1000
        context_docs = []
        total_tokens = 0

        for obj in matches:
            content_id = obj.properties.get("content_id", str(obj.uuid))
            doc = strip_html(content_dict[content_type].get(content_id, obj.properties.get("description", "")))[:3000]
            if not doc:
                logger.warning(f"No content found for content_id={content_id}, content_type={content_type}")
                continue
            try:
                token_len = len(encoding.encode(doc))
                if total_tokens + token_len <= max_tokens:
                    context_docs.append(doc)
                    total_tokens += token_len
                else:
                    break
            except Exception as e:
                logger.error(f"Token encoding failed for content_id={content_id}: {str(e)}", exc_info=True)
                continue

        if not context_docs:
            logger.warning(f"No usable context data found for query={query}, content_type={content_type}")
            return jsonify({"error": "No usable context data found"}), 404

        context_text = "\n\n---\n\n".join(context_docs)
        system_prompt = f"You are an expert assistant for addiction recovery {content_type}."
        user_prompt = f"""Use the following {content_type} to answer the question.\n\n{context_text}\n\nQuestion: {query}\nAnswer:"""

        try:
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
        except APIError as e:
            logger.error(f"OpenAI completion failed: {str(e)}", exc_info=True)
            return jsonify({"error": "RAG answer generation failed", "details": str(e)}), 500
    except Exception as e:
        logger.error(f"Unexpected error in rag_answer_content: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error", "details": str(e)}), 500
    finally:
        client.close()
        logger.info("Weaviate client closed after rag_answer_content")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)