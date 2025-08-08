# Podcast RAG (YouTube Timestamp Jump) — FastAPI + LangChain + Streamlit

This proof‑of‑concept lets you:
- Ingest **YouTube videos** (fetch transcript, chunk, embed w/ OpenAI).
- **Search** for relevant chunks and **jump** to the exact moment in YouTube via `?start=<seconds>`.

## Stack
- **Backend**: FastAPI, LangChain 0.3, OpenAIEmbeddings (`text-embedding-3-small`), Pinecone (cloud vector database)
- **YouTube**: `youtube-transcript-api` for transcript (public transcripts; no API key), `yt-dlp` for duration/title
- **Frontend**: Streamlit tiles with embedded YouTube iframes that skip to the timestamp

> Uses Pinecone as a managed vector database for efficient similarity search and scalability.

---

## Quickstart

```bash
# 1) Create venv however you like, then:
pip install -r requirements.txt

# 2) Copy env and set your keys
cp .env .env
# Set OPENAI_API_KEY, PINECONE_API_KEY, and optional INDEX_NAME
# Set DEV=true for local development or DEV=false for production

# 3) Run backend (FastAPI)
uvicorn backend.main:app --reload --port 8000

# 4) Run frontend (Streamlit)
streamlit run frontend/streamlit_app.py
```

### Setting up Pinecone

1. Sign up for a Pinecone account at [pinecone.io](https://www.pinecone.io/)
2. Create a new index with the following settings:
   - Dimensions: 3072 (for OpenAI's `text-embedding-3-large`)
   - Metric: Cosine similarity
3. Copy your Pinecone API key and index name to your `.env` file:
   ```
   PINECONE_API_KEY=your-api-key
   INDEX_NAME=your-index-name
   ```

### Ingesting Data

**YouTube URLs**
- In the Streamlit sidebar paste 1+ URLs and click **Ingest YouTube URLs**.
- We try English (and auto) transcripts; chunk at ~800 chars with overlap; each chunk's **timestamp** is the first segment's start time in that chunk.
- **Limitations**: Transcripts may be disabled/unavailable; some videos don't permit transcript retrieval; newly uploaded videos may take time before captions are available.
- **IP Bans**: YouTube may block requests from cloud provider IPs (AWS, GCP, Azure). Use the proxy configuration to work around this issue.

**Environment Configuration**
- The app uses a `DEV` environment variable to toggle between local and production URLs:
  ```
  DEV=true  # Uses http://127.0.0.1:8000 as API_BASE
  DEV=false # Uses https://morphus-rag-chat.vercel.app as API_BASE
  ```
- This simplifies deployment and testing by eliminating the need to manually change API_BASE URLs.

**Proxy Configuration**
- To work around YouTube IP bans, the app supports Webshare rotating residential proxies.
- Configure in your `.env` file:
  ```
  USE_PROXY=true
  WEBSHARE_PROXY_USERNAME=your-username
  WEBSHARE_PROXY_PASSWORD=your-password
  ```
- Sign up for a Webshare account and purchase a "Residential" proxy package (not "Proxy Server" or "Static Residential").
- This helps avoid RequestBlocked or IpBlocked exceptions when deploying to cloud providers.

**Clearing Data**
- In the Streamlit sidebar, click **Clear Index** to remove all data from the Pinecone index.
- A confirmation button will appear to prevent accidental deletion.

### Searching
- Enter a question and hit **Search**.
- You'll see tiles with:
  - Title
  - Jump time (e.g., `09:12`)
  - YouTube **embed** starting at the exact second
  - "Open on YouTube" link with `&t=<seconds>`

---

## API (FastAPI)

- `GET /healthz` – smoke test
- `POST /ingest/youtube` – body: `{ "urls": ["https://youtu.be/..."] }`
- `GET /search?q=...&k=6` – returns top‑k results with `video_url`, `start_seconds`

---

## Notes / Design Choices
- **Pinecone** used as a cloud-based vector database for efficient similarity search and scalability.
- **Embeddings** via `text-embedding-3-small` to keep cost low. Upgrade if needed.
- **Timestamps** for YouTube transcripts are exact (from API).
- **Security**: This is a local POC. Add auth, quotas, and CORS rules before deploying.
- **Observability**: Wire in LangSmith if you want traces.

---

## Project Layout
```
podcast-rag/
  backend/
    ingest.py        # YouTube ingestion + timestamp logic
    main.py          # FastAPI app + /search
    models.py        # Pydantic schemas
    settings.py      # env config
    store.py         # Pinecone vector store
    yt.py            # YouTube helpers (id, meta)
  frontend/
    streamlit_app.py # UI: ingest controls + result tiles with embeds
  shared/
    utils.py         # time utils
  requirements.txt
  .env.example
  README.md
```

---

## License
MIT for the template — adjust as needed.
