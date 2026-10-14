"""Shared data structures."""
from dataclasses import dataclass, field


@dataclass
class Document:
    """A piece of extracted content plus where it came from.

    Common metadata keys:
        source        PDF file name or URL the content belongs to
        source_type   pdf | web | youtube | attachment
        content_type  text | table | page_ocr | image_ocr | audio_transcript | video_transcript | web_page
        page          page number inside the PDF (if any)
        url           URL the content was scraped from (if any)
        found_in      the PDF (and page) where a link was discovered
    """

    text: str
    metadata: dict = field(default_factory=dict)
