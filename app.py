import os
import requests
import streamlit as st
from dotenv import load_dotenv
import chromadb
from PyPDF2 import PdfReader
import time

# Load environment variables
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Setup Vector DB (Chroma)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="user_docs", metadata={"hnsw:space": "cosine"}
)

# ---------------------- Ollama Helpers ----------------------

def request_with_retry(url, payload, retries=5, timeout=60, stream=False):
    """Generic POST request with retries + exponential backoff"""
    for attempt in range(retries):
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=timeout,
                stream=stream
            )
            response.raise_for_status()
            return response
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                st.warning(f"⚠️ Request failed ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"❌ Request failed after {retries} attempts: {e}")


def get_embedding(text: str):
    """Get embeddings from Ollama"""
    try:
        response = request_with_retry(
            "http://localhost:11434/api/embeddings",
            {"model": "nomic-embed-text", "prompt": text, "stream": False},
            timeout=60,
        )
        data = response.json()
        return data.get("embedding", None)
    except Exception as e:
        st.error(f"⚠️ Embedding failed: {e}")
        return None


def chat_with_ollama(context, query, model_name):
    """Chat with Ollama (streaming response)"""
    try:
        # Log the request for debugging
        st.session_state.setdefault('debug_logs', []).append(f"Sending request to Ollama with model: {model_name}")
        
        # Check if Ollama is running
        try:
            # Simple check to see if Ollama is available
            check_response = requests.get("http://localhost:11434/api/version", timeout=5)
            check_response.raise_for_status()
            st.session_state['debug_logs'].append("Ollama server is running")
        except Exception as e:
            st.error(f"⚠️ Ollama server not available: {e}. Make sure Ollama is running on http://localhost:11434")
            yield "⚠️ Ollama server not available. Make sure Ollama is running on http://localhost:11434"
            return
            
        # Check if model is available
        try:
            models_response = requests.get("http://localhost:11434/api/tags", timeout=10)
            models_response.raise_for_status()
            models_data = models_response.json()
            model_available = any(model['name'] == model_name for model in models_data.get('models', []))
            
            if not model_available:
                st.warning(f"Model {model_name} may not be available. Attempting to use it anyway.")
        except Exception:
            # Continue even if we can't check models
            pass
            
        # Prepare the chat request
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "Answer concisely using the context provided."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"},
            ],
            "stream": True,
        }
        
        response = request_with_retry(
            "http://localhost:11434/api/chat",
            payload,
            timeout=300,
            stream=True,
        )

        # Stream the response tokens
        answer = ""
        token_count = 0
        
        for line in response.iter_lines():
            if line:
                try:
                    data = line.decode("utf-8")
                    # Parse JSON properly instead of using eval
                    import json
                    try:
                        json_data = json.loads(data)
                        if json_data.get("message", {}).get("content"):
                            part = json_data["message"]["content"]
                            answer += part
                            token_count += 1
                            yield answer
                    except json.JSONDecodeError:
                        # Handle partial JSON
                        if '"content":' in data:
                            # Extract content between quotes after "content":
                            import re
                            content_match = re.search(r'"content":\s*"([^"]*)"', data)
                            if content_match:
                                part = content_match.group(1)
                                answer += part
                                token_count += 1
                                yield answer
                except Exception as e:
                    st.session_state['debug_logs'].append(f"Error parsing line: {str(e)}")
                    continue
        
        if token_count == 0:
            yield "No response generated. Please check if the model is working correctly."
        return
    except Exception as e:
        st.session_state['debug_logs'].append(f"Chat error: {str(e)}")
        yield f"⚠️ Chat failed: {e}"


# ---------------------- News Fetch ----------------------

def fetch_news(topic, api_key):
    try:
        url = (
            f"https://newsapi.org/v2/everything?q={topic}&sortBy=publishedAt&apiKey={api_key}"
        )
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "articles" in data:
            return [
                f"{a['title']} - {a['description']} ({a['publishedAt']})"
                for a in data["articles"]
                if a.get("title") and a.get("description")
            ]
        return []
    except Exception as e:
        st.error(f"⚠️ Failed to fetch news: {e}")
        return []


# ---------------------- Utils ----------------------

def split_text(text, chunk_size=1000, overlap=100):
    """Split text into chunks for embeddings"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# ---------------------- Streamlit UI ----------------------
st.set_page_config(page_title="Real-Time RAG with Ollama", layout="wide")
st.title("📡 Real-Time RAG with Ollama")

# Model selection
model_choice = st.selectbox("Choose Ollama Model:", ["llama2:7b", "mistral:7b"])

# Mode selection
mode = st.radio("Choose Mode:", ["📂 Upload Docs", "📰 News Search"])

# Show API Key status
st.write("🔑 News API Key loaded:", NEWS_API_KEY is not None)

# ---------------------- Upload Docs Mode ----------------------
if mode == "📂 Upload Docs":
    # Initialize session state for debug logs if not exists
    if 'debug_logs' not in st.session_state:
        st.session_state['debug_logs'] = []
    
    # Add a debug toggle in sidebar
    with st.sidebar:
        debug_mode = st.checkbox("Show Debug Info", value=False)
        
        if debug_mode and st.button("Clear Debug Logs"):
            st.session_state['debug_logs'] = []
            
        if debug_mode and st.session_state['debug_logs']:
            st.subheader("Debug Logs")
            for log in st.session_state['debug_logs'][-10:]:  # Show last 10 logs
                st.text(log)
    
    uploaded_files = st.file_uploader(
        "Upload documents (txt/pdf)", type=["txt", "pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Processing documents..."):
            for file in uploaded_files:
                try:
                    if file.type == "application/pdf":
                        pdf = PdfReader(file)
                        file_content = " ".join(
                            [page.extract_text() for page in pdf.pages if page.extract_text()]
                        )
                    else:
                        file_content = file.read().decode("utf-8")

                    # Split into chunks & add embeddings
                    chunks = split_text(file_content)
                    st.info(f"Processing {len(chunks)} chunks from {file.name}")
                    
                    # Log for debugging
                    st.session_state['debug_logs'].append(f"Created {len(chunks)} chunks from {file.name}")
                    
                    successful_chunks = 0
                    for i, chunk in enumerate(chunks):
                        emb = get_embedding(chunk)
                        if emb:
                            # Check if embedding is valid (not empty and has expected dimensions)
                            if isinstance(emb, list) and len(emb) > 0:
                                collection.add(
                                    documents=[chunk],
                                    embeddings=[emb],
                                    ids=[f"{file.name}_{i}_{hash(chunk)}"]
                                )
                                successful_chunks += 1
                            else:
                                st.warning(f"Invalid embedding format for chunk {i} in {file.name}")
                                st.session_state['debug_logs'].append(f"Invalid embedding format: {type(emb)}")
                        else:
                            st.warning(f"Failed to get embedding for chunk {i} in {file.name}")
                    
                    st.session_state['debug_logs'].append(f"Successfully embedded {successful_chunks}/{len(chunks)} chunks")
                    
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
                    st.session_state['debug_logs'].append(f"File processing error: {str(e)}")
            
            st.success("✅ Documents indexed into ChromaDB")

    # Display collection stats
    try:
        collection_count = collection.count()
        if collection_count > 0:
            st.info(f"📚 Your knowledge base contains {collection_count} document chunks")
    except Exception as e:
        st.session_state['debug_logs'].append(f"Error getting collection count: {str(e)}")

    query = st.text_input("💬 Ask a question based on your documents:")
    if query:
        with st.spinner("Processing your question..."):
            # Log the query
            st.session_state['debug_logs'].append(f"Processing query: {query}")
            
            query_emb = get_embedding(query)
            if query_emb:
                try:
                    # Check if embedding is valid
                    if not isinstance(query_emb, list) or len(query_emb) == 0:
                        st.error("Invalid query embedding format")
                        st.session_state['debug_logs'].append(f"Invalid query embedding: {type(query_emb)}")
                    else:
                        # Log embedding success
                        st.session_state['debug_logs'].append(f"Query embedding successful, length: {len(query_emb)}")
                        
                        # Get collection count to verify we have documents
                        collection_count = collection.count()
                        if collection_count == 0:
                            st.warning("Your knowledge base is empty. Please upload documents first.")
                            st.session_state['debug_logs'].append("Query attempted on empty collection")
                        else:
                            # Query the collection
                            results = collection.query(query_embeddings=[query_emb], n_results=min(3, collection_count))
                            
                            # Log query results
                            st.session_state['debug_logs'].append(f"Query returned {len(results.get('documents', [[]]))} document sets")
                            
                            if results and results.get("documents") and len(results["documents"]) > 0 and len(results["documents"][0]) > 0:
                                context = " ".join([doc for doc in results["documents"][0]])
                                
                                # Truncate context if too long
                                if len(context) > 8000:
                                    context = context[:8000] + "..."
                                    st.session_state['debug_logs'].append("Context truncated to 8000 chars")
                                
                                st.subheader("📖 Answer:")
                                answer_box = st.empty()
                                
                                # Log context length
                                st.session_state['debug_logs'].append(f"Context length: {len(context)} chars")
                                
                                answer_generated = False
                                for partial in chat_with_ollama(context, query, model_choice):
                                    answer_box.markdown(partial)
                                    answer_generated = True
                                
                                if not answer_generated:
                                    st.error("Failed to generate an answer. Please try again.")
                                    st.session_state['debug_logs'].append("No answer generated from Ollama")
                            else:
                                st.warning("No relevant documents found. Please upload more documents or try a different question.")
                                st.session_state['debug_logs'].append("No relevant documents found in query results")
                except Exception as e:
                    st.error(f"Error querying documents: {str(e)}")
                    st.session_state['debug_logs'].append(f"Query error: {str(e)}")
            else:
                st.error("Failed to process your question. Please try again.")
                st.session_state['debug_logs'].append("Failed to generate query embedding")

# ---------------------- News Mode ----------------------
elif mode == "📰 News Search":
    topic = st.text_input("Enter a news topic:")
    if st.button("Fetch News") and NEWS_API_KEY:
        articles = fetch_news(topic, NEWS_API_KEY)
        if articles:
            st.success(f"✅ Retrieved {len(articles)} news articles.")
            for art in articles:
                for chunk in split_text(art):
                    emb = get_embedding(chunk)
                    if emb:
                        collection.add(
                            documents=[chunk],
                            embeddings=[emb],
                            ids=[topic + str(hash(chunk))],
                        )
        else:
            st.warning("⚠️ No articles found.")

    query = st.text_input("💬 Ask a question about the news:")
    if query:
        query_emb = get_embedding(query)
        if query_emb:
            results = collection.query(query_embeddings=[query_emb], n_results=3)
            context = " ".join([doc for doc in results["documents"][0]])
            st.subheader("📰 Answer:")
            answer_box = st.empty()
            for partial in chat_with_ollama(context, query, model_choice):
                answer_box.write(partial)
