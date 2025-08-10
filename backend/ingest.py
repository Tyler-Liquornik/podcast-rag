import os
from typing import List, Dict, Any

from langchain.schema import Document

from store import vs
from yt import _get_video_metadata, _fetch_youtube_transcript_chunks
from settings import logger, DEBUG_LOGGING


# Metadata:
# - "video_url": str
# - "title": str
# - "start_seconds": int
# - "source": "youtube_api" | "md_estimated"
# - "raw_path": str

def ingest_youtube_urls(urls: List[str]) -> List[Dict[str, Any]]:
    logger.info(f"Starting ingestion of {len(urls)} YouTube URLs")
    if DEBUG_LOGGING:
        logger.debug(f"URLs to ingest: {urls}")

    results = []
    for i, url in enumerate(urls):
        logger.info(f"Processing URL {i+1}/{len(urls)}: {url}")
        try:
            # Step 1: Get video metadata
            try:
                logger.info(f"Fetching metadata for URL: {url}")
                vid, duration, title = _get_video_metadata(url)
                logger.info(f"Successfully fetched metadata: video_id={vid}, duration={duration}s, title='{title}'")
            except Exception as e:
                error_msg = f"Failed to fetch metadata for URL: {url}. Error: {str(e)}"
                logger.error(error_msg)
                results.append({
                    "url": url, 
                    "status": "error", 
                    "error": str(e),
                    "error_type": "metadata_fetch_failed",
                    "error_details": "Failed to fetch video metadata. The video might be unavailable or restricted."
                })
                continue

            # Step 2: Get transcript chunks
            try:
                logger.info(f"Fetching transcript for video_id: {vid}")
                transcript_chunks = _fetch_youtube_transcript_chunks(vid, url, title)
                if not transcript_chunks:
                    error_msg = f"No transcript available for video_id: {vid}"
                    logger.error(error_msg)
                    results.append({
                        "url": url, 
                        "status": "error", 
                        "error_type": "no_transcript",
                        "error_details": "No transcript available for this video. It might be private, age-restricted, or have no captions."
                    })
                    continue
                logger.info(f"Successfully fetched and chunked transcript: {len(transcript_chunks)} chunks")

                if DEBUG_LOGGING:
                    logger.debug(f"Transcript chunk details for {url}:")
                    for i, chunk in enumerate(transcript_chunks[:3]):  # Log first 3 chunks as sample
                        logger.debug(f"  Chunk {i+1} start: {chunk.metadata.get('start_seconds')}s")
                        logger.debug(f"  Chunk {i+1} length: {len(chunk.page_content)} chars")
                    if len(transcript_chunks) > 3:
                        logger.debug(f"  ... and {len(transcript_chunks) - 3} more chunks")
            except Exception as e:
                error_msg = f"Failed to fetch transcript for video_id: {vid}. Error: {str(e)}"
                logger.error(error_msg)
                results.append({
                    "url": url, 
                    "status": "error", 
                    "error": str(e),
                    "error_type": "transcript_fetch_failed",
                    "error_details": "Failed to fetch video transcript. The video might be unavailable or have no captions."
                })
                continue

            # Step 3: Add to vector store
            try:
                logger.info(f"Adding {len(transcript_chunks)} chunks to vector store for URL: {url}")
                vs.add_documents(transcript_chunks)
                logger.info(f"Successfully added {len(transcript_chunks)} chunks to vector store for URL: {url}")
                results.append({"url": url, "status": "ok", "chunks": len(transcript_chunks)})
            except Exception as e:
                error_msg = f"Failed to add chunks to vector store for URL: {url}. Error: {str(e)}"
                logger.error(error_msg)
                results.append({
                    "url": url, 
                    "status": "error", 
                    "error": str(e),
                    "error_type": "embedding_failed",
                    "error_details": "Failed to create embeddings. Check your OpenAI API key format in .env file."
                })
        except Exception as e:
            error_msg = f"Unexpected error processing URL: {url}. Error: {str(e)}"
            logger.error(error_msg)
            results.append({
                "url": url, 
                "status": "error", 
                "error": str(e),
                "error_type": "unknown_error",
                "error_details": "An unexpected error occurred during processing."
            })

    logger.info(f"Completed ingestion of {len(urls)} YouTube URLs")
    success_count = len([r for r in results if r.get("status") == "ok"])
    logger.info(f"Successfully processed {success_count}/{len(urls)} URLs")

    return results
