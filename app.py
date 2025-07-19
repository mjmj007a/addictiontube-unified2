from flask import Flask, request, jsonify
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery
from dotenv import load_dotenv
import os

# Initialize Flask app
app = Flask(__name__)
load_dotenv()

# Connect to Weaviate
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),
    auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
    skip_init_checks=True
)

@app.route('/search_content', methods=['GET'])
def search_content():
    query = request.args.get('q', '')
    content_type = request.args.get('content_type', '')
    category = request.args.get('category', '')
    per_page = int(request.args.get('per_page', 10))

    collection = client.collections.get("Content")

    filters = []
    if content_type:
        filters.append(Filter.by_property("type").equal(content_type))
    if category:
        filters.append(Filter.by_property("category").equal(category))
    combined_filter = Filter.all_of(filters) if filters else None

    response = collection.query.near_text(
        query=query,
        limit=per_page,
        filters=combined_filter,
        return_properties=["title", "type", "category", "url", "date", "schema_version"],
        return_metadata=MetadataQuery(creation_time=True, distance=True)
    )

    results = sorted(
        [{
            "title": obj.properties.get("title", ""),
            "type": obj.properties.get("type", ""),
            "category": obj.properties.get("category", ""),
            "url": obj.properties.get("url", ""),
            "date": obj.properties.get("date", ""),
            "distance": obj.metadata.distance,
            "created": obj.metadata.creation_time
        } for obj in response.objects],
        key=lambda x: x["date"],
        reverse=True
    )

    return jsonify({"results": results})


@app.route('/rag_answer_content', methods=['GET'])
def rag_answer_content():
    query = request.args.get('q', '')
    content_type = request.args.get('content_type', '')
    category = request.args.get('category', '')

    collection = client.collections.get("Content")

    filters = []
    if content_type:
        filters.append(Filter.by_property("type").equal(content_type))
    if category:
        filters.append(Filter.by_property("category").equal(category))
    combined_filter = Filter.all_of(filters) if filters else None

    response = collection.generate.near_text(
        query=query,
        limit=3,
        filters=combined_filter,
        single_prompt="Summarize the content titled {title} with text {text} in 50 words, focusing on recovery themes."
    )

    results = sorted(
        [{
            "title": obj.properties.get("title", ""),
            "summary": obj.generated,
            "date": obj.properties.get("date", "")
        } for obj in response.objects],
        key=lambda x: x["date"],
        reverse=True
    )

    return jsonify({"results": results})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
