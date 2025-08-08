from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import time

from .data_models import IngestYouTubeRequest, SearchResponse, SearchResponseItem
from .store import vs
from .ingest import ingest_youtube_urls
from .utils import seconds_to_hms
from .settings import logger, DEBUG_LOGGING

app = FastAPI(title="Podcast RAG API", version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    # Log the request
    logger.info(f"Request: {request.method} {request.url.path}")
    if DEBUG_LOGGING:
        logger.debug(f"Request headers: {request.headers}")
        body = await request.body()
        if body:
            try:
                logger.debug(f"Request body: {body.decode()}")
            except:
                logger.debug(f"Request body: {body} (binary)")

    # Process the request
    response = await call_next(request)

    # Log the response
    process_time = time.time() - start_time
    logger.info(f"Response: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s")

    return response

@app.get("/healthz")
def health():
    logger.debug("Health check endpoint called")
    return {"ok": True}

@app.post("/ingest/youtube")
def ingest_youtube(req: IngestYouTubeRequest):
    url_count = len(req.urls)
    logger.info(f"Ingesting {url_count} YouTube URLs")

    if DEBUG_LOGGING:
        for i, url in enumerate(req.urls):
            logger.debug(f"URL {i+1}: {url}")

    try:
        results = ingest_youtube_urls([str(u) for u in req.urls])

        # Log summary of results
        success_count = len([r for r in results if r.get("status") == "ok"])
        logger.info(f"YouTube ingestion completed: {success_count}/{url_count} URLs successful")

        if DEBUG_LOGGING:
            for result in results:
                status = result.get("status")
                url = result.get("url", "Unknown URL")
                if status == "ok":
                    logger.debug(f"Success for {url}: {result.get('chunks')} chunks")
                else:
                    logger.debug(f"Failed for {url}: {result.get('error_type')} - {result.get('error_details')}")

        # If all URLs failed, return a 400 Bad Request status
        if success_count == 0 and url_count > 0:
            error_details = [
                {
                    "url": r.get("url", "Unknown URL"),
                    "error_type": r.get("error_type", "unknown_error"),
                    "error_details": r.get("error_details", "Unknown error")
                }
                for r in results if r.get("status") == "error"
            ]
            logger.error(f"All {url_count} URLs failed to process")
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Failed to process all YouTube URLs",
                    "errors": error_details,
                    "results": results
                }
            )

        return {"results": results}
    except Exception as e:
        logger.error(f"Error in YouTube ingestion endpoint: {str(e)}")
        raise


@app.get("/search", response_model=SearchResponse)
def search(q: str = Query(..., description="User query"), k: int = 6):
    logger.info(f"Searching for: '{q}' with k={k}")

    try:
        start_time = time.time()
        pairs = vs.search(q, k=k)
        search_time = time.time() - start_time

        logger.info(f"Search returned {len(pairs)} results in {search_time:.3f}s")

        results: List[SearchResponseItem] = []
        for doc, score in pairs:
            meta = doc.metadata or {}
            title = str(meta.get("title") or "Untitled")
            video_url = meta.get("video_url")
            start_seconds = int(meta.get("start_seconds") or 0)

            results.append(
                SearchResponseItem(
                    score=float(score),
                    title=title,
                    snippet=doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""),
                    video_url=video_url,
                    start_seconds=start_seconds,
                    start_hms=seconds_to_hms(start_seconds),
                )
            )

            if DEBUG_LOGGING:
                logger.debug(f"Result: score={score:.4f}, title='{title}', start={start_seconds}s")
                if video_url:
                    logger.debug(f"  video_url: {video_url}")

        return SearchResponse(query=q, results=results)
    except Exception as e:
        logger.error(f"Error in search endpoint: {str(e)}")
        raise
