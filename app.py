from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from openai import OpenAI, APIError
import weaviate
from weaviate.classes.init import Auth
import os
import re
import json
import logging
from logging.handlers import RotatingFileHandler
import tiktoken
from bs4 import BeautifulSoup
import random
from dotenv import load_dotenv
import nltk
from tenacity import retry, stop_after_attempt, wait_exponential

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://addictiontube.com", "http://addictiontube.com"]}})

# Configure logging
logger = logging.getLogger('addictiontube')
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler('/tmp/unified_search.log', maxBytes=10485760, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logger.addHandler(handler)

# Initialize Flask-Limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
    headers_enabled=True,
)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_CLUSTER_URL = os.getenv("WEAVIATE_CLUSTER_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# Validate environment variables
missing_vars = []
if not OPENAI_API_KEY:
    missing_vars.append("OPENAI_API_KEY")
if not WEAVIATE_CLUSTER_URL:
    missing_vars.append("WEAVIATE_CLUSTER_URL")
if not WEAVIATE_API_KEY:
    missing_vars.append("WEAVIATE_API_KEY")
if missing_vars:
    error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
    logger.error(error_msg)
    raise EnvironmentError(error_msg)

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)

def get_weaviate_client():
    for attempt in range(3):
        try:
            weaviate_client = weaviate.connect_to_weaviate_cloud(
                cluster_url=WEAVIATE_CLUSTER_URL,
                auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
                headers={"X-OpenAI-Api-Key": OPENAI_API_KEY},
                skip_init_checks=False
            )
            if weaviate_client.is_ready():
                logger.info("Weaviate client initialized and ready")
                return weaviate_client
            else:
                logger.warning("Weaviate client initialized but not ready")
                weaviate_client.close()
        except Exception as e:
            logger.error(f"Weaviate client initialization attempt {attempt + 1} failed: {str(e)}")
    logger.error("Failed to initialize Weaviate client after 3 attempts")
    raise EnvironmentError("Weaviate client initialization failed")

# Load metadata
try:
    with open('songs_revised_with_songs-july06.json', 'r', encoding='utf-8') as f:
        song_dict = {item['video_id']: item['song'] for item in json.load(f)}
    with open('videos_revised_with_poems-july04.json', 'r', encoding='utf-8') as f:
        poem_dict = {item['video_id']: item['poem'] for item in json.load(f)}
    with open('stories.json', 'r', encoding='utf-8') as f:
        story_dict = {item['id']: item['text'] for item in json.load(f)}
    logger.info("Metadata JSON files loaded successfully")
except Exception as e:
    logger.error(f"Failed to load metadata JSON files: {str(e)}")
    raise

# Initialize NLTK lemmatizer
try:
    nltk.download('wordnet', quiet=True, raise_on_error=True)
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    logger.info("NLTK WordNetLemmatizer initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize NLTK WordNetLemmatizer: {str(e)}. Falling back to no lemmatization.")
    lemmatizer = None

def strip_html(text):
    try:
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        logger.error(f"HTML stripping error: {str(e)}")
        return text or ''

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def get_embedding(text):
    try:
        response = client.embeddings.create(input=text, model="text-embedding-3-small")
        return response.data[0].embedding
    except APIError as e:
        logger.error(f"OpenAI embedding failed: {str(e)}")
        raise

def preprocess_query(query):
    if lemmatizer:
        words = query.lower().split()
        lemmatized = [lemmatizer.lemmatize(word, pos='n') for word in words]
        processed = ' '.join(lemmatized)
        logger.info(f"Processed query: {query} -> {processed}")
        return processed
    logger.warning(f"No lemmatizer available, using raw query: {query}")
    return query.lower()

@app.route('/', methods=['GET', 'HEAD'])
def health_check():
    logger.info("Health check endpoint accessed")
    weaviate_client = get_weaviate_client()
    try:
        if not weaviate_client.is_ready():
            return jsonify({"error": "Weaviate client not ready"}), 503
        collections = weaviate_client.collections.list_all()
        if 'Content' not in collections:
            return jsonify({"error": "Content collection not found"}), 503
        try:
            embedding = get_embedding("test")
            logger.info("OpenAI embedding test successful")
        except APIError as e:
            return jsonify({"error": "OpenAI health check failed", "details": str(e)}), 503
        return jsonify({"status": "ok", "message": "AddictionTube Unified API is running"}), 200
    finally:
        weaviate_client.close()

@app.errorhandler(429)
def ratelimit_handler(e):
    logger.warning(f"Rate limit exceeded: {e.description}")
    return jsonify({
        "error": "Rate limit exceeded",
        "details": f"Too many requests. Please wait and try again. Limit: {e.description}"
    }), 429

@app.route('/search_content', methods=['GET'])
@limiter.limit("60 per hour")
def search_content():
    query = re.sub(r'[^\w\s.,!?]', '', request.args.get('q', '')).strip()
    content_type = request.args.get('content_type', '').strip().lower()
    page = max(1, int(request.args.get('page', 1)))
    size = max(1, min(100, int(request.args.get('per_page', 1))))

    if not query or not content_type or content_type not in ['songs', 'poems', 'stories']:
        logger.error(f"Invalid request: query='{query}', content_type='{content_type}'")
        return jsonify({"error": "Invalid or missing query or content type"}), 400

    weaviate_client = get_weaviate_client()
    try:
        processed_query = preprocess_query(query)
        try:
            vector = get_embedding(processed_query)
        except APIError as e:
            logger.error(f"OpenAI embedding failed: {str(e)}")
            return jsonify({"error": "Embedding service unavailable", "details": str(e)}), 500

        try:
            collection = weaviate_client.collections.get("Content")
            from weaviate.classes.query import Filter
            filters = Filter.by_property("type").equal(content_type)
            properties = ["content_id", "title", "description", "url"]
            results = collection.query.near_vector(
                near_vector=vector,
                limit=50,  # Increased to fetch all relevant results
                filters=filters,
                return_metadata=["distance"],
                return_properties=properties
            )
            logger.info(f"Query results: {len(results.objects)} objects found for query='{query}', content_type='{content_type}'")
            total = len(results.objects)
            paginated = results.objects[(page - 1) * size:page * size]

            items = []
            for obj in paginated:
                content_id = obj.properties.get("content_id", str(obj.uuid))
                logger.info(f"Processing item: Content ID: {content_id}, Distance: {obj.metadata.distance}")
                item = {
                    "content_id": content_id,
                    "score": obj.metadata.distance,
                    "title": strip_html(obj.properties.get("title", "N/A")),
                    "description": strip_html(obj.properties.get("description", ""))
                }
                if content_type == 'stories':
                    item['image'] = obj.properties.get("url", "")
                elif content_type in ['songs', 'poems']:
                    item['url'] = obj.properties.get("url", "")
                items.append(item)
                logger.info(f"Item: {item}")

            logger.info(f"Search completed: query='{query}', content_type='{content_type}', page={page}, total={total}, returned={len(items)}")
            return jsonify({"results": items, "total": total})
        except Exception as e:
            logger.error(f"Weaviate query failed for {content_type}: {str(e)}")
            return jsonify({"error": "Search service unavailable", "details": str(e)}), 500
    finally:
        weaviate_client.close()

@app.route('/rag_answer_content', methods=['GET'])
@limiter.limit("30 per hour")
def rag_answer_content():
    query = re.sub(r'[^\w\s.,!?]', '', request.args.get('q', '')).strip()
    content_type = request.args.get('content_type', '').strip().lower()
    reroll = request.args.get('reroll', '').lower().startswith('yes')

    if not query or not content_type or content_type not in ['songs', 'poems', 'stories']:
        logger.error(f"Invalid RAG request: query='{query}', content_type='{content_type}'")
        return jsonify({"error": "Invalid or missing query or content type"}), 400

    weaviate_client = get_weaviate_client()
    try:
        processed_query = preprocess_query(query)
        try:
            vector = get_embedding(processed_query)
        except APIError as e:
            logger.error(f"OpenAI embedding failed: {str(e)}")
            return jsonify({"error": "Embedding service unavailable", "details": str(e)}), 500

        try:
            collection = weaviate_client.collections.get("Content")
            from weaviate.classes.query import Filter
            filters = Filter.by_property("type").equal(content_type)
            results = collection.query.near_vector(
                near_vector=vector,
                limit=50,  # Increased for RAG context
                filters=filters,
                return_metadata=["distance"],
                return_properties=["content_id", "text", "description"]
            )
            logger.info(f"RAG query results: {len(results.objects)} objects found for query='{query}', content_type='{content_type}'")
        except Exception as e:
            logger.error(f"Weaviate query failed: {str(e)}")
            return jsonify({"error": "Weaviate query failed", "details": str(e)}), 500

        matches = results.objects
        if reroll:
            random.shuffle(matches)

        if not matches:
            logger.warning(f"No matches found for query='{query}', content_type='{content_type}'")
            return jsonify({"error": "No relevant context found"}), 404

        encoding = tiktoken.get_encoding("cl100k_base")
        max_tokens = 16384 - 1000
        context_docs = []
        total_tokens = 0
        content_dict = {'songs': song_dict, 'poems': poem_dict, 'stories': story_dict}

        for match in matches:
            text = content_dict[content_type].get(match.properties.get("content_id", ""), match.properties.get("text", match.properties.get("description", "")))
            if not text:
                logger.warning(f"Match {match.properties.get('content_id')} has no text metadata in {content_type}")
                continue
            doc = strip_html(text)[:3000]
            doc_tokens = len(encoding.encode(doc))
            if total_tokens + doc_tokens <= max_tokens:
                context_docs.append(doc)
                total_tokens += doc_tokens
            else:
                break

        if not context_docs:
            logger.warning(f"No usable context data for query='{query}', content_type='{content_type}'")
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
            logger.info(f"RAG answer generated for query='{query}', content_type='{content_type}'")
            return jsonify({"answer": answer})
        except APIError as e:
            logger.error(f"OpenAI completion failed: {str(e)}")
            return jsonify({"error": "RAG processing failed", "details": str(e)}), 500
    finally:
        weaviate_client.close()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)