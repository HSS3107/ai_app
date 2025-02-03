from typing import Dict, List, Optional, Tuple, Any
import re
import logging
from dataclasses import dataclass
from enum import Enum
import nltk
from nltk.tokenize import sent_tokenize
import spacy
from dataclasses import dataclass
from enum import Enum
import json
import requests
from django.conf import settings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import asyncio
import nest_asyncio
from asgiref.sync import async_to_sync
import threading
from datetime import datetime
import os
import traceback
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
    TokenTextSplitter,
    CharacterTextSplitter,
    SpacyTextSplitter
)


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


logger = logging.getLogger(__name__)


class SmartTextSplitter:
    """A smart text splitter that chooses the optimal splitting strategy based on document type and content."""
    
    def __init__(self):
        
        
        # Configure markdown headers
        self.markdown_headers = [
            ("######", 6),  # Add level 6
            ("#####", 5),   # Add level 5
            ("####", 4),    # Add level 4
            ("###", 3),
            ("##", 2),
            ("#", 1)
        ]
        
        # Configure HTML headers
        self.html_headers = [
            ("h1", 1),
            ("h2", 2),
            ("h3", 3),
            ("h4", 4)
        ]
        
        # Initialize specialized splitters
        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.markdown_headers)
        self.html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=self.html_headers)
        
        # Initialize token-based splitter for semantic coherence
        self.token_splitter = TokenTextSplitter(
            chunk_size=4000,
            chunk_overlap=100,
            encoding_name="cl100k_base"  # OpenAI's tiktoken encoding
        )
        
        # Initialize spaCy splitter for linguistic coherence
        self.spacy_splitter = SpacyTextSplitter(
            chunk_size=4000,
            chunk_overlap=100,
            separator=" "
        )
        
        # Fallback character splitter
        self.char_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=4000,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False
        )
        
        self.semantic_splitter = TokenTextSplitter(
            chunk_size=2500,
            chunk_overlap=300,
            encoding_name="cl100k_base"
        )
    
    def split_text(self, text: str, file_type: str = None, structure_type: str = None) -> List[str]:
        # Try header-preserving split first
        try:
            chunks = []
            current_chunk = []
            for line in text.split('\n'):
                if re.match(r'^#+\s+', line):
                    if current_chunk:
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = []
                current_chunk.append(line)
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
            return chunks
        except Exception as e:
            logger.error(f"Header-based split failed: {str(e)}")
        
        # Fallback to paragraph-based splitting
        return [p for p in text.split('\n\n') if p.strip()]


class DocumentStructureType(Enum):
    TOC_BASED = "toc_based"
    HEADER_BASED = "header_based"
    SEMANTIC_BASED = "semantic_based"

@dataclass
class Section:
    title: str
    level: int
    content: str
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    chapter_num: Optional[str] = None
    subsections: List['Section'] = None

    def __post_init__(self):
        if self.subsections is None:
            self.subsections = []

    def to_dict(self) -> Dict:
        """Convert section to dictionary format."""
        return {
            'title': self.title,
            'level': self.level,
            'content': self.content,
            'start_page': self.start_page,
            'end_page': self.end_page,
            'chapter_num': self.chapter_num,
            'subsections': [sub.to_dict() for sub in self.subsections] if self.subsections else []
        }

    @property
    def normalized_title(self):
        """Create search-friendly title version"""
        return re.sub(r'\W+', ' ', self.title).lower().strip()

# Pydantic models for LLM output parsing
class TOCItem(BaseModel):
    """Model for a single TOC item."""
    title: str = Field(..., description="The title of the section")
    level: int = Field(..., description="The hierarchical level (1 for chapters, 2 for sections, etc.)")
    chapter_num: Optional[str] = Field(None, description="Chapter or section number if available")
    page_number: Optional[int] = Field(None, description="Starting page number if available")

class TOCStructure(BaseModel):
    """Model for the complete TOC structure."""
    sections: List[TOCItem] = Field(..., description="List of all sections in the document")

class SmartDocumentProcessor:
    """Smart document processor with LLM-based TOC extraction."""

    def __init__(self):
        try:
            # Add verbose logging
            self.logger = logging.getLogger('SmartDocumentProcessor')
            self.logger.setLevel(logging.DEBUG)

            # Add console handler if not already present
            if not self.logger.handlers:
                ch = logging.StreamHandler()
                ch.setLevel(logging.DEBUG)
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)

            # Initialize NLP components
            self.nlp = spacy.load("en_core_web_sm")

            self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_retries=3,
            request_timeout=60,
            max_tokens=4000  # Ensure we get complete JSON responses
        )

            # Initialize output parser
            self.parser = PydanticOutputParser(pydantic_object=TOCStructure)

            # TOC extraction prompts
            # Update the TOC extraction prompt to be more explicit
            self.toc_extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", """Extract ALL document headers with surgical precision. Follow these rules:

**Header Types to Capture:**
- Markdown headers (e.g., `## Leadership Principles`, `###### "THE SAFETY VALVE"`)
- Numbered sections (e.g., "22: If You Must Find Fault...")
- Quoted/phrase headers (e.g., `###### "A DROP OF HONEY"`)
- Part/chapter markers (e.g., `# PART FOUR`, `### Fundamental Techniques`)
- Appendices/prefaces (e.g., `###### Preface To Revised Edition`)

**Level Assignment Guidelines:**
1. Markdown: `#`=1, `##`=2, `###`=3, `####`=4, `#####`=5 and `######`=6.
2. Parts (`# PART ONE`) = Level 1 
3. Chapter titles (`### How To Win People...`) = Level 2
4. Subheaders (`###### "GIVE A DOG A GOOD NAME"`) = Level 3
5. Numbered principles ("24: Let the other person...") = Level 4

**Special Cases:**
- For quoted headers, preserve exact capitalization/punctuation
- Treat "Part One"/"Part Two" as top-level sections
- Capture continuation lines (e.g., headers split across lines)
- Include page numbers in titles if present (e.g., "Appendix......123")

**Output Format DEMAND:**
```json
{
  "sections": [
    {
      "title": "EXACT header text with formatting",
      "level": <1-6>, 
      "chapter_num": "<extracted number>",
      "page_number": <int>
    }
  ]
}""")])

            self.toc_refinement_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are tasked with refining and validating a document's table of contents structure.
                Review the extracted TOC and improve it by:
                1. Ensuring consistent numbering
                2. Validating hierarchical relationships
                3. Checking for missing sections
                4. Standardizing section titles
                5. Verifying page numbers if present

                Return the refined structure in the same JSON format.
                """),
                ("human", "Here is the extracted TOC to refine:\n\n{toc_data}")
            ])

            # Semantic section detection patterns
            self.semantic_markers = {
                'topic_shift': [
                    'however', 'nevertheless', 'in contrast',
                    'furthermore', 'moreover', 'in addition',
                    'therefore', 'thus', 'consequently',
                    'in conclusion', 'to summarize', 'finally'
                ],
                'section_indicators': [
                    'introduction', 'background', 'methodology',
                    'results', 'discussion', 'conclusion',
                    'summary', 'recommendations', 'references'
                ] 
            }

            # Initialize text splitter
            self.splitter = SmartTextSplitter()

            self.initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize processor: {str(e)}")
            self.initialized = False
            raise

    def _clean_text(self, text: str) -> str:
        """Improved text normalization with PDF-specific cleaning"""
        # Remove PDF artifacts
        text = re.sub(r'\s*\d{1,3}\s*\n', '\n', text)  # Page numbers
        text = re.sub(r'\b\w{1,3}\b', '', text)  # Isolated numbers/letters
        text = re.sub(r'\*\d+\*', '', text)  # **92**-style page markers
        # Remove legal/copyright blocks
        text = re.sub(r'Editors disclaim all liability.*?END OF LICENSE', '', text, flags=re.DOTALL)
        # Normalize section headers
        text = re.sub(r'_{2,}', '', text)  # Remove underline headers
        return super()._clean_text(text)

    async def extract_toc_with_llm(self, text: str) -> List[Section]:
        try:
            logger.info(f"Starting LLM Based ToC extraction")
            cleaned_text = self._clean_text(text[:45000])

            # Use more explicit prompt template
            prompt_template = ChatPromptTemplate.from_template("""
            Analyze this document and extract its table of contents. Follow these rules:
            1. Preserve exact header text including any markdown
            2. Maintain hierarchy based on header levels
            3. Include page numbers if available
            4. Output MUST be valid JSON with this structure:
            {{
                "sections": [
                    {{
                        "title": "Section Title",
                        "level": 1,
                        "chapter_num": "1.1",
                        "page_number": 12
                    }}
                ]
            }}
            Document Content:
            {text}
            """)

            messages = prompt_template.format_messages(text=cleaned_text)

            # Add retry mechanism
            for attempt in range(3):
                try:
                    response = await asyncio.wait_for(
                        self.llm.agenerate([messages]),
                        timeout=60
                    )
                    raw_json = response.generations[0][0].text
                    logger.debug(f"LLM Response (Attempt {attempt+1}):\n{raw_json}")
                    break
                except asyncio.TimeoutError:
                    if attempt == 2:
                        raise
                    continue

            # Validate and parse response
            parsed = self.parser.parse(self._extract_json(raw_json))
            return self._convert_toc_items_to_sections(parsed.sections)

        except Exception as e:
            logger.error(f"LLM TOC extraction failed: {str(e)}")
            logger.debug(f"Error traceback:\n{traceback.format_exc()}")
            return []

    def _extract_json(self, text: str) -> str:
        """Extract JSON from potential markdown code blocks."""
        try:
            # First try direct JSON parsing
            json.loads(text)
            return text
        except json.JSONDecodeError:
            # Extract JSON from code blocks
            code_blocks = re.findall(r'```json\n(.*?)\n```', text, re.DOTALL)
            if code_blocks:
                return max(code_blocks, key=len).strip()  # Get longest code block

            # Fallback to extracting first complete JSON object
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json_match.group()

            raise ValueError("No valid JSON found in response")

    def _section_to_dict(self, section: Section) -> dict:
        """Convert a Section object to dictionary format."""
        return {
            'title': section.title,
            'level': section.level,
            'chapter_num': section.chapter_num,
            'page_range': f"{section.start_page}-{section.end_page}" if section.start_page and section.end_page else None,
            'subsections': [self._section_to_dict(subsec) for subsec in section.subsections] if section.subsections else []
        }

    def find_toc_items(self, text: str) -> List[TOCItem]:
        """
        Find potential table of contents items in the text using pattern matching.
        This is used as a backup when LLM-based extraction fails.
        
        Args:
            text (str): The document text to analyze
            
        Returns:
            List[TOCItem]: List of found TOC items
        """
        items = []
        lines = text.split('\n')

        # Patterns for different types of headers and TOC entries
        patterns = {
            # Markdown headers (level 1-6)
            'markdown': r'^(#{1,6})\s*(?:Chapter\s+)?(\d+)?\.?\s*(.+?)(?:\s*\|\s*(\d+))?$',
            
            # Traditional chapter headers
            'chapter': r'^(?:Chapter|CHAPTER)\s+(\d+)[.:]?\s*(.+?)(?:\s*\|\s*(\d+))?$',
            
            # Numbered sections
            'numbered': r'^(\d+(?:\.\d+)?)\s+(.+?)(?:\s*\|\s*(\d+))?$',
            
            # Roman numeral sections
            'roman': r'^([IVX]+)[.:]?\s*(.+?)(?:\s*\|\s*(\d+))?$'
        }

        for line in lines:
            line = line.strip()
            if not line:
                continue

            for pattern_type, pattern in patterns.items():
                match = re.match(pattern, line)
                if match:
                    if pattern_type == 'markdown':
                        hashes, chapter_num, title, page = match.groups()
                        level = len(hashes) if hashes else 1
                    elif pattern_type == 'chapter':
                        chapter_num, title, page = match.groups()
                        level = 1
                    elif pattern_type == 'numbered':
                        chapter_num, title, page = match.groups()
                        level = 2 if '.' in str(chapter_num) else 1
                    else:  # roman
                        chapter_num, title, page = match.groups()
                        level = 1

                    # Clean up the title
                    title = re.sub(r'\s+', ' ', title).strip()

                    # Convert page number to int if present
                    page_num = int(page) if page else None

                    # Create TOC item
                    item = TOCItem(
                        title=title,
                        level=level,
                        chapter_num=str(chapter_num) if chapter_num else None,
                        page_number=page_num
                    )
                    items.append(item)
                    break  # Stop checking other patterns if one matches

        # Post-process the items to ensure proper hierarchy
        processed_items = self._process_toc_hierarchy(items)

        return processed_items

    def _process_toc_hierarchy(self, items: List[TOCItem]) -> List[TOCItem]:
        """
        Process TOC items to ensure proper hierarchical relationships.
        Adjusts levels based on context and numbering patterns.
        
        Args:
            items (List[TOCItem]): Raw TOC items
            
        Returns:
            List[TOCItem]: Processed TOC items with corrected hierarchy
        """
        if not items:
            return []

        # First pass: Find the minimum level to normalize levels
        min_level = min(item.level for item in items)

        # Adjust levels relative to minimum
        normalized_items = []
        for item in items:
            # Create new item with adjusted level
            adjusted_level = item.level - min_level + 1
            normalized_items.append(TOCItem(
                title=item.title,
                level=adjusted_level,
                chapter_num=item.chapter_num,
                page_number=item.page_number
            ))

        # Second pass: Ensure subsections are properly nested
        for i in range(1, len(normalized_items)):
            current = normalized_items[i]
            previous = normalized_items[i-1]

            # Check if current item should be a subsection based on numbering
            if current.chapter_num and previous.chapter_num:
                # If current number is like "1.1" and previous is "1"
                if '.' in str(current.chapter_num) and \
                str(current.chapter_num).split('.')[0] == str(previous.chapter_num):
                    current.level = previous.level + 1

            # Ensure no level jumps by more than 1
            if current.level > previous.level + 1:
                current.level = previous.level + 1

        return normalized_items

    def _convert_toc_items_to_sections(self, toc_items: List[TOCItem]) -> List[Section]:
        sections = []
        for i, item in enumerate(toc_items):
            # Calculate end_page based on next item's start_page
            end_page = item.page_number
            if i < len(toc_items) - 1:
                next_item = toc_items[i + 1]
                end_page = next_item.page_number - 1 if next_item.page_number else item.page_number

            section = Section(
                title=item.title,
                level=item.level,
                content="",
                chapter_num=item.chapter_num or "N/A",  # Default for missing chapter
                start_page=item.page_number,
                end_page=end_page
            )
            sections.append(section)
        return sections

    def _extract_fallback_sections(self, text: str) -> List[Section]:
        """Extract sections using pattern matching as fallback."""
        sections = []
        lines = text.split('\n')
        current_section = None

        # Updated header patterns to include markdown headers
        header_patterns = [
            r'^#{1,6}\s*(?:Chapter|CHAPTER)\s+\d+',  # Matches markdown chapter headers
            r'^#{1,6}\s*\d+\.\s+[A-Z]',              # Numbered sections with markdown
            r'^(?:Chapter|CHAPTER)\s+\d+',            # Plain chapter headers
            r'^\d+\.\s+[A-Z]',                       # Numbered sections
            r'^[IVX]+\.\s+[A-Z]',                    # Roman numeral sections
            r'^\d+\.\d+\s+[A-Z]'                     # Subsections
        ]

        for line in lines:
            line = line.strip()
            if not line:
                continue

            is_header = any(re.match(pattern, line) for pattern in header_patterns)
            if is_header:
                if current_section:
                    sections.append(current_section)

                # Determine level based on pattern
                level = 1
                if re.match(r'^#{6}', line):  # Six hashes indicates chapter
                    level = 1
                elif re.match(r'^#{5}', line):  # Five hashes indicates major section
                    level = 2
                elif re.match(r'^#{3,4}', line):  # 3-4 hashes indicates subsection
                    level = 3
                elif re.match(r'^\d+\.\d+', line):  # x.y format indicates subsection
                    level = 2

                # Clean title by removing markdown symbols
                title = re.sub(r'^#{1,6}\s*', '', line)

                current_section = Section(
                    title=title,
                    level=level,
                    content="",
                    chapter_num=self._extract_section_number(title)
                )
            elif current_section:
                current_section.content += line + "\n"

        # Add the last section
        if current_section:
            sections.append(current_section)

        return sections

    def _extract_section_number(self, header: str) -> Optional[str]:
        """Extract section number from header text."""
        patterns = [
            r'(?:Chapter|CHAPTER)\s+(\d+)',    # Regular chapter numbers
            r'^(\d+)\.?\s+',                   # Simple numbers
            r'^([IVX]+)\.?\s+',                # Roman numerals
            r'^(\d+\.\d+)\s+'                  # Decimal numbers
        ]

        header = re.sub(r'^#{1,6}\s*', '', header)  # Remove markdown symbols before matching

        for pattern in patterns:
            match = re.match(pattern, header)
            if match:
                return match.group(1)
        return None

    def process_document(self, text: str, resource_id: Optional[str] = None) -> Tuple[List[Dict], Dict]:
        """Thread-safe synchronous wrapper for async document processing."""
        try:
            # Create a new event loop for this thread if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Enable nested event loop if needed
            nest_asyncio.apply()

            # Run the async processing
            try:
                return async_to_sync(self.process_document_async)(text, resource_id)
            finally:
                # Clean up if we created a new loop
                if not loop.is_running():
                    loop.close()

        except Exception as e:
            logger.error(f"Error in process_document: {str(e)}")
            return [], {}

    async def process_document_async(self, text: str, resource_id: Optional[str] = None) -> Tuple[List[Dict], Dict]:
        try:
            sections = await self.extract_toc_with_llm(text)
            if not sections:
                sections = self._extract_fallback_sections(text)

            # Get document title from the first top-level section
            document_title = next((s.title for s in sections if s.level == 1), "Unknown Document")

            # Populate content and validate sections
            sections = self._populate_section_content(text, sections)
            chunks = self.create_chunks_from_sections(sections, resource_id or "N/A", document_title)

            return chunks, self._create_optimized_content_map(sections, resource_id)
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return [], {}

    def _filter_pdf_structures(self, text: str) -> str:
        """Enhanced PDF artifact removal"""
        # Remove PDF line numbers and markers
        text = re.sub(r'\s*\d{1,3}\s*\n', '\n', text) 
        # Remove lonely header-like lines
        text = re.sub(r'^\W+$', '', text, flags=re.MULTILINE)
        # Remove 2+ line separator sequences
        text = re.sub(r'-{3,}', '', text)
        return text

    def _create_optimized_content_map(self, sections: List[Section], resource_id: Optional[str] = None) -> Dict:
        """Create an optimized content map that minimizes token usage."""
        def section_to_map_entry(section: Section) -> Dict:
            return {
                'title': section.title,
                'level': section.level,
                'chapter_num': section.chapter_num,
                'metadata': {
                    'start_page': section.start_page,
                    'end_page': section.end_page,
                    'resource_id': resource_id
                },
                # Only include a brief summary instead of full content
                'summary': create_section_summary(section.content) if section.content else "",
                'subsections': [
                    section_to_map_entry(subsec) 
                    for subsec in section.subsections
                ] if section.subsections else []
            }

        def create_section_summary(content: str, max_length: int = 200) -> str:
            """Create a brief summary of section content."""
            if not content:
                return ""
            # Get the first paragraph or sentence that gives the main idea
            first_para = content.split('\n\n')[0] if '\n\n' in content else content
            if len(first_para) > max_length:
                return first_para[:max_length].rsplit(' ', 1)[0] + '...'
            return first_para

        try:
            content_map = {
                'structure_type': 'llm_generated',
                'document_stats': {
                    'total_sections': len(sections),
                    'max_depth': max(s.level for s in sections),
                    'has_subsections': any(s.subsections for s in sections)
                },
                'main_sections': [
                    {
                        'title': section.title,
                        'chapter_num': section.chapter_num
                    }
                    for section in sections if section.level == 1
                ],
                'sections': [section_to_map_entry(section) for section in sections]
            }

            if resource_id:
                content_map['metadata'] = {
                    'resource_id': resource_id,
                    'generated_at': datetime.now().isoformat(),
                    'processor_version': '2.0'
                }

            return content_map

        except Exception as e:
            logger.error(f"Error creating content map: {str(e)}")
            # Return minimal content map in case of error
            return {
                'structure_type': 'llm_generated',
                'sections': [
                    {
                        'title': s.title,
                        'level': s.level,
                        'chapter_num': s.chapter_num
                    } for s in sections
                ]
            }

    def _normalize_title(self, title: str) -> str:
        """Normalize section title for comparison."""
        # Remove special characters but keep basic structure
        title = re.sub(r'[^\w\s\-:.]', '', title)
        # Remove leading/trailing spaces
        title = title.strip()
        # Convert to lowercase
        title = title.lower()
        return title

    def _find_section_boundaries(self, text: str, sections: List[Section]) -> List[Tuple[int, int, Section]]:
        """Find section boundaries with enhanced markdown support."""
        try:
            lines = text.split('\n')
            boundaries = []
            processed_positions = set()

            # Create normalized version of each line once
            normalized_lines = [self._normalize_title(line) for line in lines]

            # First pass: Find explicit markdown headers
            for i, line in enumerate(lines):
                if i in processed_positions:
                    continue

                # Check for markdown headers
                if match := re.match(r'^(#{1,6})\s+(.+)$', line.strip()):
                    header_level = len(match.group(1))
                    header_text = match.group(2).strip()

                    # Find matching section
                    for section in sections:
                        if self._titles_match(header_text, section.title):
                            boundaries.append((i, -1, section))
                            processed_positions.add(i)
                            break

            # Second pass: Find numbered sections
            for i, line in enumerate(lines):
                if i in processed_positions:
                    continue

                # Check for numbered sections
                if match := re.match(r'^\d+:\s*(.+)$', line.strip()):
                    section_text = match.group(1).strip()

                    for section in sections:
                        if self._titles_match(section_text, section.title):
                            boundaries.append((i, -1, section))
                            processed_positions.add(i)
                            break

            # Third pass: Find part headers
            for i, line in enumerate(lines):
                if i in processed_positions:
                    continue

                if re.match(r'^Part\s+[A-Z\d]+:', line.strip()):
                    for section in sections:
                        if self._titles_match(line.strip(), section.title):
                            boundaries.append((i, -1, section))
                            processed_positions.add(i)
                            break

            # Sort boundaries by position
            boundaries.sort(key=lambda x: x[0])

            # Calculate end positions
            processed_boundaries = []
            for i, (start, _, section) in enumerate(boundaries):
                end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(lines)
                processed_boundaries.append((start, end, section))

            return processed_boundaries

        except Exception as e:
            self.logger.error(f"Error finding section boundaries: {str(e)}")
            return []

    def _titles_match(self, title1: str, title2: str) -> bool:
        """Enhanced title matching with multiple strategies."""
        def normalize(text: str) -> str:
            # Remove markdown symbols, punctuation, and normalize whitespace
            text = re.sub(r'^#+\s*', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            return text.lower().strip()

        normalized1 = normalize(title1)
        normalized2 = normalize(title2)

        # Try different matching strategies
        return any([
            normalized1 == normalized2,  # Exact match after normalization
            normalized1 in normalized2 or normalized2 in normalized1,  # Substring match
            # Fuzzy match for longer titles
            len(normalized1) > 10 and len(normalized2) > 10 and (
                normalized1[:10] == normalized2[:10] or
                normalized1[-10:] == normalized2[-10:]
            )
        ])

    def _populate_section_content(self, text: str, sections: List[Section]) -> List[Section]:
        """Enhanced section content population with better markdown handling."""
        try:
            self.logger.info(f"Starting content population for {len(sections)} sections")

            # Split text into lines for processing
            lines = text.split('\n')
            current_position = 0
            current_section = None

            # Define header patterns for better markdown matching
            header_patterns = [
                (r'^#{1,6}\s*(.+?)$', lambda m: m.group(1).strip()),  # Markdown headers
                (r'^(\d+:\s*.+?)$', lambda m: m.group(1).strip()),    # Numbered sections
                (r'^(Part\s+[A-Z\d]+:.+?)$', lambda m: m.group(1).strip()), # Part headers
                (r'^([A-Z][^.!?]+)$', lambda m: m.group(1).strip())   # All-caps headers
            ]

            # Track processed sections to avoid duplicates
            processed_sections = set()
            section_content = {}

            # First pass: Identify section boundaries
            for i, line in enumerate(lines):
                # Check each line against header patterns
                for pattern, extractor in header_patterns:
                    if match := re.match(pattern, line.strip()):
                        header_text = extractor(match)

                        # Find matching section
                        matching_section = None
                        for section in sections:
                            # Try different matching approaches
                            if any([
                                header_text == section.title,
                                header_text == self._normalize_title(section.title),
                                header_text in section.title or section.title in header_text
                            ]):
                                matching_section = section
                                break

                        if matching_section and matching_section.title not in processed_sections:
                            # If we found a section boundary
                            if current_section:
                                # Store content for previous section
                                content = '\n'.join(lines[current_position:i]).strip()
                                if self._validate_section_content(content):
                                    section_content[current_section.title] = content

                            current_section = matching_section
                            current_position = i + 1
                            processed_sections.add(current_section.title)
                            break

            # Handle final section
            if current_section and current_section.title not in section_content:
                content = '\n'.join(lines[current_position:]).strip()
                if self._validate_section_content(content):
                    section_content[current_section.title] = content

            # Second pass: Apply content to sections
            for section in sections:
                if section.title in section_content:
                    section.content = section_content[section.title]
                    self.logger.debug(
                        f"Populated content for section '{section.title}' "
                        f"({len(section.content)} chars)"
                    )
                else:
                    self.logger.warning(f"No content found for section: {section.title}")

            # Log results
            populated_count = len(section_content)
            self.logger.info(
                f"Successfully populated content for {populated_count}/{len(sections)} sections"
            )

            return sections

        except Exception as e:
            self.logger.error(f"Error in content population: {str(e)}")
            self.logger.debug(f"Error traceback:\n{traceback.format_exc()}")
            return sections

    def _validate_section_content(self, content: str) -> bool:
        """Validate section content is meaningful."""
        if not content:
            return False

        content = content.strip()
        if len(content) < 50:  # Minimum content length
            return False

        # Check for actual text content (not just numbers/punctuation)
        text_content = re.sub(r'[\d\W]+', '', content)
        if not text_content:
            return False

        # Check for common garbage patterns
        garbage_patterns = [
            r'^\s*page\s+\d+\s*$',
            r'^\s*chapter\s+\d+\s*$',
            r'^\s*[-_=]{3,}\s*$'
        ]
        if any(re.match(pattern, content, re.IGNORECASE) for pattern in garbage_patterns):
            return False

        return True

    def _fuzzy_match(self, target: str, paragraphs: List[str]) -> Optional[str]:
        """More flexible matching"""
        try:
            # Allow partial matches and synonyms
            synonyms = {
                'introduction': ['preface', 'overview'],
                'methodology': ['approach', 'methods'],
                # Add other domain-specific synonyms
            }

            search_terms = [target] + synonyms.get(target, [])

            for term in search_terms:
                pattern = re.compile(rf'\b{re.escape(term)}\b', re.IGNORECASE)
                for p in paragraphs:
                    if pattern.search(p):
                        return p
            return None
        except re.error as e:
            logger.error(f"Regex error: {str(e)}")
            return None

    def _sequential_match(self, all_sections: List[Section], current: Section, paragraphs: List[str]) -> Optional[str]:
        """Find content based on section position"""
        idx = all_sections.index(current)
        if idx < len(paragraphs):
            return paragraphs[idx]
        return None

    def _keyword_fallback(self, target: str, paragraphs: List[str]) -> Optional[str]:
        """Fallback to keyword presence"""
        keywords = set(target.split())
        for p in paragraphs:
            if keywords.issubset(p.split()):
                return p
        return None

    async def _store_toc(self, resource_id: str, content_map: Dict) -> None:
        """Store TOC in the API."""
        try:
            url = f"{settings.API_BASE_URL}/resources/{resource_id}/toc/"
            response = requests.post(url, json=content_map)
            response.raise_for_status()
            logger.info(f"Successfully stored TOC for resource {resource_id}")
        except Exception as e:
            logger.error(f"Error storing TOC: {str(e)}")

    def create_chunks_from_sections(self, sections: List[Section], document_id: str, document_title: str) -> List[Dict]:
        """Create chunks with improved content handling and validation."""
        try:
            self.logger.info(f"Starting chunk creation with {len(sections)} sections")
            chunks = []

            # Configuration
            max_chunk_size = 200000
            min_chunk_size = 100
            overlap_size = 200

            def create_chunk(content: str, section: Section, chunk_index: int) -> Dict:
                return {
                    "document_id": document_id,
                    "document_title": document_title,
                    "chapter_no": section.chapter_num or "N/A",
                    "chapter_title": section.title,
                    "start_page_no": section.start_page or 1,
                    "end_page_no": section.end_page or section.start_page or 1,
                    "chapter_content": content,
                    "chunk_index": chunk_index,
                    "chunk_no": len(chunks) + 1,
                    "total_chunks": 0  # Will be updated at the end
                }

            for section in sections:
                if not section.content:
                    self.logger.warning(f"Skipping empty section: {section.title}")
                    continue

                content = section.content.strip()

                # Handle small sections
                if len(content) <= max_chunk_size:
                    if len(content) >= min_chunk_size:
                        chunk_content = f"## {section.title}\n\n{content}"
                        chunks.append(create_chunk(chunk_content, section, len(chunks) + 1))
                    continue

                # Split large sections into chunks
                paragraphs = content.split('\n\n')
                current_chunk = [f"## {section.title}\n"]
                current_size = len(current_chunk[0])

                for paragraph in paragraphs:
                    paragraph = paragraph.strip()
                    if not paragraph:
                        continue

                    paragraph_size = len(paragraph)

                    if current_size + paragraph_size > max_chunk_size:
                        # Create chunk if we have enough content
                        if current_size >= min_chunk_size:
                            chunk_content = '\n\n'.join(current_chunk)
                            chunks.append(create_chunk(chunk_content, section, len(chunks) + 1))

                        # Start new chunk with overlap
                        current_chunk = [f"## {section.title}\n"]
                        if chunks and chunks[-1]["chapter_content"]:
                            # Add overlap from previous chunk
                            last_paragraphs = chunks[-1]["chapter_content"].split('\n\n')[-2:]
                            current_chunk.extend(last_paragraphs)

                        current_chunk.append(paragraph)
                        current_size = sum(len(p) for p in current_chunk)
                    else:
                        current_chunk.append(paragraph)
                        current_size += paragraph_size + 2  # +2 for newlines

                # Add final chunk if enough content remains
                if current_chunk and current_size >= min_chunk_size:
                    chunk_content = '\n\n'.join(current_chunk)
                    chunks.append(create_chunk(chunk_content, section, len(chunks) + 1))

            # Update total chunks count
            total_chunks = len(chunks)
            for chunk in chunks:
                chunk["total_chunks"] = total_chunks

            self.logger.info(f"Created {total_chunks} chunks")
            return chunks

        except Exception as e:
            self.logger.error(f"Error creating chunks: {str(e)}")
            return []

    def is_valid_content(self, text: str) -> bool:
        """Validate chunk content with updated criteria."""
        if not text or not isinstance(text, str):
            return False

        text = text.strip()
        if not text:
            return False

        # Basic validation criteria
        min_chars = 50  # Increased minimum character length
        min_words = 10  # Increased minimum word count

        words = text.split()

        return all([
            len(text) >= min_chars,
            len(words) >= min_words,
            any(c.isalpha() for c in text),  # Contains at least one letter
            not text.isdigit()  # Not just numbers
        ])
        # def process_section(section: Section):
        #     if not section.content or not section.content.strip():
        #         logger.warning(f"Empty content in section: {section.title}")
        #         return

        #     # Clean content
        #     content = section.content.strip()
        #     content = re.sub(r'\s+', ' ', content)

        #     # Handle small sections
        #     if len(content) < min_chunk_size:
        #         if len(content) > 0:
        #             chunks.append(create_chunk(content, section, 1))
        #         return

        #     # Split into sentences
        #     try:
        #         sentences = sent_tokenize(content)
        #     except Exception as e:
        #         logger.error(f"Error tokenizing sentences: {str(e)}")
        #         sentences = content.split('. ')

        #     current_chunk = []
        #     current_size = 0
        #     chunk_num = 1

        #     for sentence in sentences:
        #         sentence = sentence.strip()
        #         if not sentence:
        #             continue

        #         sentence_size = len(sentence)

        #         if current_chunk and (current_size + sentence_size > max_chunk_size):
        #             if current_size >= min_chunk_size:
        #                 chunk_text = ' '.join(current_chunk)
        #                 chunks.append(create_chunk(chunk_text, section, chunk_num))
        #                 chunk_num += 1
        #             current_chunk = []
        #             current_size = 0

        #         current_chunk.append(sentence)
        #         current_size += sentence_size

        #     # Handle remaining content
        #     if current_chunk:
        #         chunk_text = ' '.join(current_chunk)
        #         if len(chunk_text) >= min_chunk_size:
        #             chunks.append(create_chunk(chunk_text, section, chunk_num))

        #     # Process subsections
        #     for subsection in section.subsections:
        #         process_section(subsection)

        # # Process all sections
        # for section in sections:
        #     process_section(section)

        # logger.info(f"Created {len(chunks)} chunks")
        # return chunks

    async def chunk_and_analyze(self, text: str, document_info: Dict) -> Dict[str, Any]:
        """Perform detailed analysis of document chunks using LLM."""
        try:
            analysis_prompt = ChatPromptTemplate.from_messages([
                ("system", """Analyze this document segment as an educational expert. For this segment:
                1. Identify main themes and concepts
                2. Extract key terms and definitions
                3. Note important arguments or findings
                4. Highlight any notable quotes
                5. Mark connections to other sections if apparent

                Format your response as a structured analysis with clear sections."""),
                ("human", "Document Title: {title}\nSegment Text:\n{text}")
            ])

            chunks = self.create_chunks_from_sections(await self.extract_toc_with_llm(text))
            analyses = []

            for chunk in chunks:
                prompt = analysis_prompt.format_messages(
                    title=document_info.get('title', 'Unknown'),
                    text=chunk['content']
                )

                response = await self.llm.agenerate([prompt])
                analyses.append({
                    'chunk_metadata': chunk['metadata'],
                    'analysis': response.generations[0][0].text
                })

            return {
                'document_info': document_info,
                'chunk_analyses': analyses
            }

        except Exception as e:
            logger.error(f"Error in chunk analysis: {str(e)}")
            raise

    async def summarize_document(self, text: str, document_info: Dict) -> str:
        """Generate a comprehensive document summary using LLM."""
        try:
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", """As a master educator, create a comprehensive summary of this document that:
                1. Captures the main thesis and key arguments
                2. Outlines the document's structure and flow
                3. Identifies major themes and concepts
                4. Notes significant findings or conclusions
                5. Highlights practical applications or implications

                Your summary should help readers quickly grasp the document's essence and value."""),
                ("human", "Document Title: {title}\n\nDocument Text:\n{text}")
            ])

            # Use first chunk for high-level summary
            initial_text = text[:8000]  # First 8000 chars for initial summary

            prompt = summary_prompt.format_messages(
                title=document_info.get('title', 'Unknown'),
                text=initial_text
            )

            response = await self.llm.agenerate([prompt])
            return response.generations[0][0].text

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Error generating document summary"

    def _clean_text(self, text: str) -> str:
        """Less aggressive cleaning"""
        # Preserve section breaks and bullet points
        text = re.sub(r'\n{3,}', '\n\n', text)  # Normalize excessive newlines
        text = re.sub(r'\s*\d{1,3}\s*\n', '\n', text)  # Remove page numbers
        text = re.sub(r'\*\d+\*', '', text)  # Remove page markers
        return text.strip()

    async def chunk_and_analyze(self, text: str, document_info: Dict) -> Dict[str, Any]:
        """Perform detailed analysis of document chunks using LLM."""
        try:
            analysis_prompt = ChatPromptTemplate.from_messages([
                ("system", """Analyze this document segment as an educational expert. For this segment:
                1. Identify main themes and concepts
                2. Extract key terms and definitions
                3. Note important arguments or findings
                4. Highlight any notable quotes
                5. Mark connections to other sections if apparent

                Format your response as a structured analysis with clear sections."""),
                ("human", "Document Title: {title}\nSegment Text:\n{text}")
            ])

            chunks = self.create_chunks_from_sections(await self.extract_toc_with_llm(text))
            analyses = []

            for chunk in chunks:
                prompt = analysis_prompt.format_messages(
                    title=document_info.get('title', 'Unknown'),
                    text=chunk['content']
                )

                response = await self.llm.agenerate([prompt])
                analyses.append({
                    'chunk_metadata': chunk['metadata'],
                    'analysis': response.generations[0][0].text
                })

            return {
                'document_info': document_info,
                'chunk_analyses': analyses
            }

        except Exception as e:
            logger.error(f"Error in chunk analysis: {str(e)}")
            raise

    async def summarize_document(self, text: str, document_info: Dict) -> str:
        """Generate a comprehensive document summary using LLM."""
        try:
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", """As a master educator, create a comprehensive summary of this document that:
                1. Captures the main thesis and key arguments
                2. Outlines the document's structure and flow
                3. Identifies major themes and concepts
                4. Notes significant findings or conclusions
                5. Highlights practical applications or implications

                Your summary should help readers quickly grasp the document's essence and value."""),
                ("human", "Document Title: {title}\n\nDocument Text:\n{text}")
            ])

            # Use first chunk for high-level summary
            initial_text = text[:8000]  # First 8000 chars for initial summary

            prompt = summary_prompt.format_messages(
                title=document_info.get('title', 'Unknown'),
                text=initial_text
            )

            response = await self.llm.agenerate([prompt])
            return response.generations[0][0].text

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Error generating document summary"

    def _clean_text(self, text: str) -> str:
        """Clean and normalize document text."""
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Normalize line breaks
        text = re.sub(r'^\s*\d+\s*', '', text, flags=re.MULTILINE)  # Remove page numbers
        return text.strip()

    def _calculate_text_stats(self, text: str) -> Dict[str, Any]:
        """Calculate text statistics for better processing."""
        return {
            'length': len(text),
            'sentences': len(sent_tokenize(text)),
            'paragraphs': len([p for p in text.split('\n\n') if p.strip()]),
            'has_numbers': bool(re.search(r'\d+', text)),
            'has_headers': bool(re.search(r'^[A-Z][^.!?]*', text, re.MULTILINE))
        }
