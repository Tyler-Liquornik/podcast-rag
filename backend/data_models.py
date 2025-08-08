from typing import List, Optional
from pydantic import BaseModel, HttpUrl

class IngestYouTubeRequest(BaseModel):
    urls: List[HttpUrl]

class IngestFolderRequest(BaseModel):
    folder_path: str

class SearchResponseItem(BaseModel):
    score: float
    title: str
    snippet: str
    video_url: Optional[str] = None
    start_seconds: int
    start_hms: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResponseItem]
