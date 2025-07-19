
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import re
import json
import logging
from logging.handlers import RotatingFileHandler
from openai import OpenAI, APIError
from weaviate import connect_to_weaviate_cloud
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery
from bs4 import BeautifulSoup
import tiktoken
import random

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://addictiontube.com", "http://addictiontube.com"]}})

# Configure logging
logger = logging.getLogger('addictiontube_weaviate')
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler('unified_search_weaviate.log', maxBytes=10485760, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logger.addHandler(handler)

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
    headers_enabled=True,
)

# Load API keys
try:
    import config_v
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", config_v.OPENAI_API_KEY)
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", config_v.WEAVIATE_API_KEY)
    WEAVIATE_CLUSTER_URL = os.getenv("WEAVIATE_CLUSTER_URL", config_v.WEAVIATE_CLUSTER_URL)
except ImportError:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    WEAVIATE_CLUSTER_URL = os.getenv("WEAVIATE_CLUSTER_URL")

# Validate environment
missing_vars = []
if not OPENAI_API_KEY: missing_vars.append("OPENAI_API_KEY")
if not WEAVIATE_API_KEY: missing_vars.append("WEAVIATE_API_KEY")
if not WEAVIATE_CLUSTER_URL: missing_vars.append("WEAVIATE_CLUSTER_URL")
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

client = OpenAI(api_key=OPENAI_API_KEY)
weaviate_client = connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_CLUSTER_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    headers={"X-OpenAI-Api-Key": OPENAI_API_KEY},
    skip_init_checks=True
)
collection = weaviate_client.collections.get("Content")

# Load metadata for RAG
with open('songs_revised_with_songs-july06.json', 'r', encoding='utf-8') as f:
    song_dict = {item['video_id']: item['song'] for item in json.load(f)}
with open('videos_revised_with_poems-july04.json', 'r', encoding='utf-8') as f:
    poem_dict = {item['video_id']: item['poem'] for item in json.load(f)}
with open('stories.json', 'r', encoding='utf-8') as f:
    story_dict = {item['id']: item['text'] for item in json.load(f)}

def strip_html(text):
    return re.sub(r'<[^>]+>', '', text or '') if text else ''

@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "Weaviate Unified Search is live"}), 200

@app.route('/search_content', methods=['GET'])
@limiter.limit("60 per hour")
def search_content():
    query = re.sub(r'[^\w\s.,!?]', '', request.args.get('q', '')).strip()
    content_type = request.args.get('content_type', '')
    category = request.args.get('category', 'all')
    page = max(1, int(request.args.get('page', 1)))
    size = max(1, min(100, int(request.args.get('per_page', 5))))

    if not query or content_type not in ['songs', 'poems', 'stories']:
        return jsonify({"error": "Invalid query or content_type"}), 400

    filters = Filter.by_property("type").equal(content_type)
    if category != 'all':
        filters = filters.and_filter(Filter.by_property("category_id").equal(category))

    try:
        response = collection.query.near_text(
            query=query,
            filters=filters,
            limit=200,
            return_metadata=MetadataQuery(distance=True)
        )
        start = (page - 1) * size
        matches = response.objects[start:start + size]

        results = []
        for obj in matches:
            props = obj.properties
            results.append({
                "id": obj.uuid,
                "score": 1.0 - obj.metadata.distance,
                "title": strip_html(props.get("title", "N/A")),
                "description": strip_html(props.get("description", "")),
                "image": props.get("image", "") if content_type == "stories" else ""
            })
        return jsonify({"results": results, "total": len(response.objects)})
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({"error": "Search failed", "details": str(e)}), 500

@app.route('/rag_answer_content', methods=['GET'])
@limiter.limit("30 per hour")
def rag_answer():
    query = re.sub(r'[^\w\s.,!?]', '', request.args.get('q', '')).strip()
    content_type = request.args.get('content_type', '')
    category = request.args.get('category', 'all')
    reroll = request.args.get('reroll', '').lower().startswith('yes')

    if not query or content_type not in ['songs', 'poems', 'stories']:
        return jsonify({"error": "Invalid or missing query/content_type"}), 400

    filters = Filter.by_property("type").equal(content_type)
    if category != 'all':
        filters = filters.and_filter(Filter.by_property("category_id").equal(category))

    try:
        results = collection.query.near_text(
            query=query,
            filters=filters,
            limit=5
        ).objects

        if reroll:
            random.shuffle(results)

        if not results:
            return jsonify({"error": "No relevant context found"}), 404

        content_dict = {'songs': song_dict, 'poems': poem_dict, 'stories': story_dict}
        encoding = tiktoken.get_encoding("cl100k_base")
        max_tokens = 16384 - 1000
        context_docs, total_tokens = [], 0

        for obj in results:
            text = content_dict[content_type].get(obj.uuid, obj.properties.get("description", ""))
            clean = strip_html(text)[:3000]
            doc_tokens = len(encoding.encode(clean))
            if total_tokens + doc_tokens <= max_tokens:
                context_docs.append(clean)
                total_tokens += doc_tokens
            else:
                break

        if not context_docs:
            return jsonify({"error": "No usable context data found"}), 404

        context = "

---

".join(context_docs)
        messages = [
            {"role": "system", "content": f"You are an expert assistant for addiction recovery {content_type}."},
            {"role": "user", "content": f"Use the following {content_type} to answer the question.

{context}

Question: {query}
Answer:"}
        ]
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=1000
        )
        return jsonify({"answer": response.choices[0].message.content})
    except Exception as e:
        logger.error(f"RAG error: {str(e)}")
        return jsonify({"error": "RAG processing failed", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
