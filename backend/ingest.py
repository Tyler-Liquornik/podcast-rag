import os
import re
from typing import List, Dict, Any

from langchain.schema import Document

from .store import vs
from .yt import _get_video_metadata, _fetch_youtube_transcript_chunks
from .settings import logger, DEBUG_LOGGING


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


def ingest_markdown_folder(folder_path: str) -> List[Dict[str, Any]]:
    """
    Parse all .md files in folder.
    Optional YAML-ish header supporting "youtube_url: ...".
    We assign timestamps by evenly spacing across duration; if duration is unknown,
    we approximate assuming ~160 words/min (~0.375 sec/word).
    """
    logger.info(f"Starting ingestion of markdown files from folder: {folder_path}")

    folder = os.path.abspath(folder_path)
    if not os.path.isdir(folder):
        logger.error(f"Folder not found: {folder}")
        return [{"folder": folder_path, "status": "error", "error": "folder not found"}]

    logger.info(f"Scanning for markdown files in: {folder}")

    ingested = []
    md_files_count = 0

    for root, _, files in os.walk(folder):
        md_files = [name for name in files if name.lower().endswith(".md")]
        md_files_count += len(md_files)

        for name in md_files:
            path = os.path.join(root, name)
            logger.info(f"Processing markdown file: {path}")

            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = f.read()

                logger.debug(f"File size: {len(raw)} bytes")

                youtube_url, body = _extract_header_url(raw)
                title = os.path.splitext(os.path.basename(path))[0]

                if youtube_url:
                    logger.info(f"Found YouTube URL in header: {youtube_url}")
                else:
                    logger.info(f"No YouTube URL found in header")

                duration = 0
                if youtube_url:
                    try:
                        logger.info(f"Fetching video metadata for URL: {youtube_url}")
                        _, dur, yt_title = _get_video_metadata(youtube_url)
                        duration = dur or 0
                        if yt_title:
                            logger.info(f"Using YouTube title: '{yt_title}' instead of file name")
                            title = yt_title
                        logger.info(f"Video duration: {duration}s")
                    except Exception as e:
                        logger.warning(f"Failed to fetch video metadata: {str(e)}")

                logger.info(f"Splitting content into sentences")
                sentences = _naive_sentence_split(body)

                if not sentences:
                    logger.warning(f"No content found in file: {path}")
                    ingested.append({"file": path, "status": "empty"})
                    continue

                logger.info(f"Found {len(sentences)} sentences")

                logger.info(f"Chunking content")
                chunks = _chunk_by_tokens(sentences, max_chars=800, overlap_chars=120)
                logger.info(f"Created {len(chunks)} chunks")

                n = len(chunks)
                if duration <= 0:
                    words = len(body.split())
                    duration = int(words * 0.375)
                    logger.info(f"Estimated duration from word count: {duration}s ({words} words)")

                logger.info(f"Creating document objects")
                docs = []
                for i, ch in enumerate(chunks):
                    start = int((i / max(1, (n))) * duration)
                    docs.append(Document(
                        page_content=ch,
                        metadata={
                            "video_url": youtube_url,
                            "title": title,
                            "start_seconds": max(0, start),
                            "source": "md_estimated",
                            "raw_path": path,
                        }
                    ))

                if DEBUG_LOGGING:
                    logger.debug(f"Document details:")
                    total_chars = sum(len(doc.page_content) for doc in docs)
                    logger.debug(f"  Total content length: {total_chars} chars")
                    logger.debug(f"  Average chunk size: {total_chars / len(docs):.1f} chars")

                logger.info(f"Adding {len(docs)} documents to vector store")
                vs.add_documents(docs)
                logger.info(f"Successfully added {len(docs)} documents to vector store")

                ingested.append({"file": path, "status": "ok", "chunks": len(docs)})

            except Exception as e:
                logger.error(f"Error processing file {path}: {str(e)}")
                ingested.append({"file": path, "status": "error", "error": str(e)})

    success_count = len([r for r in ingested if r.get("status") == "ok"])
    empty_count = len([r for r in ingested if r.get("status") == "empty"])
    error_count = len([r for r in ingested if r.get("status") == "error"])

    logger.info(f"Completed ingestion of markdown folder: {folder_path}")
    logger.info(f"Found {md_files_count} markdown files")
    logger.info(f"Successfully processed: {success_count}, Empty: {empty_count}, Errors: {error_count}")

    return ingested

def _extract_header_url(raw: str):
    """
    Very simple front-matter parser. Supports a header block like:

    ---
    youtube_url: https://www.youtube.com/watch?v=dQw4w9WgXcQ
    ---
    (transcript...)

    If missing, just returns (None, raw)
    """
    lines = raw.splitlines()
    if len(lines) >= 3 and lines[0].strip() == "---":
        for i in range(1, min(30, len(lines))):
            if lines[i].strip() == "---":
                header = "\n".join(lines[1:i])
                body = "\n".join(lines[i+1:])
                yt = None
                for hl in header.splitlines():
                    if hl.lower().startswith("youtube_url:"):
                        yt = hl.split(":", 1)[1].strip()
                        break
                return yt, body
    return None, raw


def _naive_sentence_split(text: str) -> List[str]:
    """
    Very small, dependency-free sentence splitter.
    Splits on punctuation followed by whitespace.
    """
    text = text.strip()
    if not text:
        return []
    parts = SENTENCE_REGEX.split(text)
    # Merge tiny fragments to avoid super short "sentences"
    merged = []
    buf = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(buf) < 60:
            buf = (buf + " " + part).strip()
        else:
            if buf:
                merged.append(buf)
            buf = part
    if buf:
        merged.append(buf)
    return merged


SENTENCE_REGEX = re.compile(r'(?<=[.!?])\s+')


def _chunk_by_tokens(sentences: List[str], max_chars: int = 800, overlap_chars: int = 120) -> List[str]:
    """
    Chunk by approximate character count (keeps sentence boundaries).
    Adds simple char-based overlap between chunks.
    """
    chunks = []
    cur = ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= max_chars:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            # start new chunk, with overlap from tail of previous
            tail = cur[-overlap_chars:] if cur else ""
            cur = (tail + " " + s).strip() if tail else s
    if cur:
        chunks.append(cur)
    return chunks
