from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from openai import OpenAI, APIError
from weaviate.classes.init import Auth
from weaviate import WeaviateClient
from weaviate.auth import AuthApiKey
from weaviate.config import Config
import weaviate
import os
import re
import json
import logging
from logging.handlers import RotatingFileHandler
import tiktoken
from dotenv import load_dotenv
import random

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://addictiontube.com", "http://addictiontube.com"]}})

# Logging
logger = logging.getLogger('addictiontube')
logger.setLevel(logging.DEBUG)
handler = RotatingFileHandler('unified_search.log', maxBytes=10485760, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logger.addHandler(handler)

# Rate Limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
    headers_enabled=True,
)

# Load env vars and clients
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_CLUSTER_URL = os.getenv("WEAVIATE_CLUSTER_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

if not all([OPENAI_API_KEY, WEAVIATE_CLUSTER_URL, WEAVIATE_API_KEY]):
    raise EnvironmentError("Missing one or more required environment variables.")

client = OpenAI(api_key=OPENAI_API_KEY)

auth = Auth.api_key(WEAVIATE_API_KEY)

weaviate_client = WeaviateClient(
    url=WEAVIATE_CLUSTER_URL,
    auth_credentials=auth,
    headers={"X-OpenAI-Api-Key": OPENAI_API_KEY},
    skip_init_checks=True
)

# Load content dictionaries
with open('songs_revised_with_songs-july06.json', 'r', encoding='utf-8') as f:
    song_dict = {item['video_id']: item['song'] for item in json.load(f)}
with open('videos_revised_with_poems-july04.json', 'r', encoding='utf-8') as f:
    poem_dict = {item['video_id']: item['poem'] for item in json.load(f)}
with open('stories.json', 'r', encoding='utf-8') as f:
    story_dict = {item['id']: item['text'] for item in json.load(f)}

def strip_html(text):
    return re.sub(r'<[^>]+>', '', text or '') if text else ''

@app.route('/', methods=['GET', 'HEAD'])
def health_check():
    return jsonify({"status": "ok", "message": "AddictionTube Unified API is running"}), 200

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Rate limit exceeded",
        "details": f"Too many requests. Please wait and try again. Limit: {e.description}"
    }), 429

@app.route('/search_content', methods=['GET'])
@limiter.limit("60 per hour")
def search_content():
    query = re.sub(r'[^\w\s.,!?]', '', request.args.get('q', '')).strip()
    content_type = request.args.get('content_type', '').strip()
    category = request.args.get('category', 'all').strip()
    page = max(1, int(request.args.get('page', 1)))
    size = max(1, min(100, int(request.args.get('per_page', 5))))

    if not query or content_type not in ['songs', 'poems', 'stories']:
        return jsonify({"error": "Invalid or missing query or content type"}), 400

    valid_categories = {
        'songs': ['1074'],
        'poems': ['1082'],
        'stories': ['1028', '1042']
    }
    if category != 'all' and category not in valid_categories[content_type]:
        return jsonify({"error": "Invalid category for selected content type"}), 400

    try:
        embedding = client.embeddings.create(input=query, model="text-embedding-ada-002")
        vector = embedding.data[0].embedding

        collection = weaviate_client.collections.get("Content")
        filters = {}
        if category != 'all':
            key = "category_id" if content_type in ['songs', 'poems'] else "category"
            filters = {"path": [key], "operator": "Equal", "valueText": category}

        results = collection.query.near_vector(vector=vector, limit=size * page, filters=filters)

        paginated = results.objects[(page - 1)*size : page*size]
        items = []
        for o in paginated:
            item = {
                "id": o.uuid,
                "score": o.distance,
                "title": strip_html(o.properties.get("title", "")),
                "description": strip_html(o.properties.get("description", "")),
                "image": o.properties.get("image", "")
            }
            items.append(item)

        return jsonify({"results": items, "total": len(results.objects)})
    except Exception as e:
        logger.error(f"Weaviate search error: {str(e)}")
        return jsonify({"error": "Search failed", "details": str(e)}), 500

@app.route('/rag_answer_content', methods=['GET'])
@limiter.limit("30 per hour")
def rag_answer_content():
    query = re.sub(r'[^\w\s.,!?]', '', request.args.get('q', '')).strip()
    content_type = request.args.get('content_type', '').strip()
    category = request.args.get('category', 'all').strip()
    reroll = request.args.get('reroll', '').lower().startswith('yes')

    if not query or content_type not in ['songs', 'poems', 'stories']:
        return jsonify({"error": "Invalid or missing query or content type"}), 400

    valid_categories = {
        'songs': ['1074'],
        'poems': ['1082'],
        'stories': ['1028', '1042']
    }
    if category != 'all' and category not in valid_categories[content_type]:
        return jsonify({"error": "Invalid category for selected content type"}), 400

    try:
        embedding = client.embeddings.create(input=query, model="text-embedding-ada-002")
        vector = embedding.data[0].embedding

        collection = weaviate_client.collections.get("Content")
        filters = {}
        if category != 'all':
            key = "category_id" if content_type in ['songs', 'poems'] else "category"
            filters = {"path": [key], "operator": "Equal", "valueText": category}

        results = collection.query.near_vector(vector=vector, limit=5, filters=filters)
        matches = results.objects
        if reroll:
            random.shuffle(matches)

        content_dict = {'songs': song_dict, 'poems': poem_dict, 'stories': story_dict}
        encoding = tiktoken.get_encoding("cl100k_base")
        max_tokens = 16384 - 1000
        total_tokens = 0
        context_docs = []

        for obj in matches:
            doc = strip_html(content_dict[content_type].get(obj.uuid, obj.properties.get("description", "")))[:3000]
            token_len = len(encoding.encode(doc))
            if total_tokens + token_len <= max_tokens:
                context_docs.append(doc)
                total_tokens += token_len
            else:
                break

        if not context_docs:
            return jsonify({"error": "No usable context data found"}), 404

        context_text = "\n\n---\n\n".join(context_docs)
        system_prompt = f"You are an expert assistant for addiction recovery {content_type}."
        user_prompt = f"""Use the following {content_type} to answer the question.\n\n{context_text}\n\nQuestion: {query}\nAnswer:"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1000
        )
        answer = response.choices[0].message.content
        return jsonify({"answer": answer})
    except Exception as e:
        logger.error(f"RAG failed: {str(e)}")
        return jsonify({"error": "RAG processing failed", "details": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
