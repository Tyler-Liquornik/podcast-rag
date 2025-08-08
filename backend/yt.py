import re
import time
from typing import Optional, Tuple, List
import yt_dlp
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.proxies import WebshareProxyConfig

from .settings import (
    logger, DEBUG_LOGGING,
    USE_PROXIES, WEBSHARE_PROXY_USERNAME, WEBSHARE_PROXY_PASSWORD
)

YOUTUBE_ID_RE = re.compile(r"(?:v=|/)([0-9A-Za-z_-]{11}).*")

def extract_video_id(url: str) -> Optional[str]:
    logger.debug(f"Extracting video ID from URL: {url}")
    m = YOUTUBE_ID_RE.search(url)
    video_id = m.group(1) if m else None

    if video_id:
        logger.debug(f"Successfully extracted video ID: {video_id}")
    else:
        logger.warning(f"Failed to extract video ID from URL: {url}")

    return video_id

def _get_video_metadata(url: str) -> Tuple[str, int, str]:
    """
    Return (video_id, length_seconds, title) using yt-dlp.
    """
    logger.info(f"Fetching video metadata with yt-dlp for URL: {url}")

    try:
        # First extract the video ID to verify the URL format
        video_id = extract_video_id(url)
        if not video_id:
            logger.error(f"Invalid YouTube URL format: {url}")
            raise ValueError(f"Invalid YouTube URL format: {url}")

        logger.debug(f"Initializing yt-dlp for video ID: {video_id}")

        # Configure yt-dlp options
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            'extract_flat': True,
        }

        # Extract video information
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            # Extract the metadata we need
            video_id = info.get('id', video_id)
            length = int(info.get('duration', 0))
            title = info.get('title', '')

        logger.info(f"Successfully fetched metadata with yt-dlp: video_id={video_id}, length={length}s, title='{title}'")

        if DEBUG_LOGGING:
            logger.debug(f"Additional video metadata from yt-dlp:")
            logger.debug(f"  Channel: {info.get('channel', 'Unknown')}")
            logger.debug(f"  Upload date: {info.get('upload_date', 'Unknown')}")
            logger.debug(f"  View count: {info.get('view_count', 'Unknown')}")
            logger.debug(f"  Average rating: {info.get('average_rating', 'Unknown')}")

        return video_id, length, title

    except Exception as e:
        logger.error(f"Error fetching video metadata with yt-dlp for URL {url}: {str(e)}")
        raise


def _fetch_youtube_transcript_chunks(video_id: str, url: str, title: str):
    """
    Pull transcript via YouTubeTranscriptApi and chunk.
    Use the start time of the first segment in each chunk as the jump timestamp.
    """
    try:
        logger.info(f"Fetching transcript for video_id: {video_id}")
        segments = None

        # Initialize YouTubeTranscriptApi with proxy if enabled
        if USE_PROXIES and WEBSHARE_PROXY_USERNAME and WEBSHARE_PROXY_PASSWORD:
            logger.info("Using Webshare proxy for transcript fetching")

            # Configure proxy
            proxy_config = WebshareProxyConfig(
                proxy_username=WEBSHARE_PROXY_USERNAME,
                proxy_password=WEBSHARE_PROXY_PASSWORD
            )

            ytt_api = YouTubeTranscriptApi(proxy_config=proxy_config)
            logger.debug("YouTubeTranscriptApi initialized with Webshare proxies")
        else:
            logger.debug("Using YouTubeTranscriptApi without Webshare proxies")
            ytt_api = YouTubeTranscriptApi()

        # Add a delay to prevent 429 Too Many Requests errors
        time.sleep(1)  # 1 second delay before making API calls

        # Try to get all available transcripts
        try:
            logger.debug(f"Listing all available transcripts for video_id: {video_id}")
            transcript_list = ytt_api.list(video_id)

            # First try to find manual English transcript
            try:
                logger.debug(f"Trying to find manual English transcript")
                transcript = transcript_list.find_transcript(['en'])
                # Add a small delay before fetching to prevent rate limiting
                time.sleep(0.5)
                fetched_transcript = transcript.fetch()
                logger.info(f"Successfully fetched English transcript (manual)")
                segments = fetched_transcript.to_raw_data()
            except Exception as e:
                logger.debug(f"No manual English transcript found: {str(e)}")

                # Fall back to auto-generated English transcript
                try:
                    logger.debug(f"Trying to find auto-generated English transcript")
                    transcript = transcript_list.find_generated_transcript(['en'])
                    # Add a small delay before fetching to prevent rate limiting
                    time.sleep(0.5)
                    fetched_transcript = transcript.fetch()
                    logger.info(f"Successfully fetched auto-generated English transcript")
                    segments = fetched_transcript.to_raw_data()
                except Exception as e:
                    logger.debug(f"No auto-generated English transcript found: {str(e)}")
                    logger.warning(f"Could not find any English transcripts for video_id: {video_id}")
        except Exception as e:
            logger.error(f"Error listing or fetching transcripts: {str(e)}")
            logger.warning(f"Failed to retrieve transcript for video_id: {video_id}")

        if not segments:
            logger.error(f"No transcript available for video_id: {video_id} in any language")
            return []

        logger.info(f"Processing {len(segments)} transcript segments")

        sents = []
        for s in segments:
            text = s.get("text", "").strip()
            start = int(float(s.get("start", 0)))
            if text:
                sents.append((start, text))

        logger.info(f"Extracted {len(sents)} text segments from transcript")

        if DEBUG_LOGGING and sents:
            logger.debug(f"First few transcript segments:")
            for i, (start, text) in enumerate(sents[:3]):  # Log first 3 segments as sample
                logger.debug(f"  Segment {i+1} at {start}s: '{text[:50]}...'")

        chunks = []
        cur_text = ""
        cur_start = None
        MAX = 800
        OVERLAP = 120

        logger.debug(f"Chunking transcript with MAX={MAX} chars, OVERLAP={OVERLAP} chars")

        def push_chunk():
            if cur_text:
                chunks.append((cur_start if cur_start is not None else 0, cur_text.strip()))

        for (start, text) in sents:
            if not cur_text:
                cur_text = text
                cur_start = start
            elif len(cur_text) + 1 + len(text) <= MAX:
                cur_text = f"{cur_text} {text}"
            else:
                push_chunk()
                tail = cur_text[-OVERLAP:]
                cur_text = f"{tail} {text}".strip()
                cur_start = start

        push_chunk()

        logger.info(f"Created {len(chunks)} chunks from transcript")

        docs = []
        for (start_sec, content) in chunks:
            docs.append(Document(
                page_content=content,
                metadata={
                    "video_url": url,
                    "title": title,
                    "start_seconds": int(start_sec),
                    "source": "youtube_api",
                }
            ))

        logger.info(f"Created {len(docs)} Document objects from chunks")

        if DEBUG_LOGGING and docs:
            logger.debug(f"Document details:")
            total_chars = sum(len(doc.page_content) for doc in docs)
            logger.debug(f"  Total content length: {total_chars} chars")
            logger.debug(f"  Average chunk size: {total_chars / len(docs):.1f} chars")

        return docs

    except Exception as e:
        logger.error(f"Error in _fetch_youtube_transcript_chunks for video_id {video_id}: {str(e)}")
        return []
