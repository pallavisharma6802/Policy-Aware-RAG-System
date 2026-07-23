import json
import re
import uuid
from pathlib import Path
from typing import List, Dict

def get_policy_url(section_name: str, metadata: Dict) -> str:
    """
    Returns the specific policy URL for a section from metadata,
    otherwise returns the default document URL.
    
    First checks exact match, then checks if section name is contained in any key.
    """
    section_urls = metadata.get('section_urls', {})
    
    # Try exact match first
    if section_name in section_urls:
        return section_urls[section_name]
    
    # Try partial match (case-insensitive)
    section_lower = section_name.lower()
    for key, url in section_urls.items():
        if section_lower in key.lower() or key.lower() in section_lower:
            return url
    
    # Fall back to document URL
    return metadata['url']

def load_markdown_file(filepath: Path) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def load_metadata(filepath: Path) -> Dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_sections(content: str) -> List[Dict]:
    sections = []
    lines = content.split('\n')
    
    current_section = None
    current_level = None
    current_text = []
    section_stack = []
    
    for line in lines:
        h2_match = re.match(r'\[SECTION-H2\]\s+(.+)', line)
        h3_match = re.match(r'\[SECTION-H3\]\s+(.+)', line)
        
        if h2_match:
            if current_section and current_text:
                sections.append({
                    'section': current_section,
                    'level': current_level,
                    'hierarchy': list(section_stack),
                    'text': '\n'.join(current_text).strip()
                })
            
            current_section = h2_match.group(1)
            current_level = 'H2'
            section_stack = [current_section]
            current_text = []
        
        elif h3_match:
            if current_section and current_text:
                sections.append({
                    'section': current_section,
                    'level': current_level,
                    'hierarchy': list(section_stack),
                    'text': '\n'.join(current_text).strip()
                })
            
            h3_title = h3_match.group(1)
            section_stack = [section_stack[0] if section_stack else '', h3_title]
            current_section = h3_title
            current_level = 'H3'
            current_text = []
        
        elif line.strip() and not line.startswith('[SECTION-H1]'):
            current_text.append(line)
    
    if current_section and current_text:
        sections.append({
            'section': current_section,
            'level': current_level,
            'hierarchy': list(section_stack),
            'text': '\n'.join(current_text).strip()
        })
    
    return sections

BOILERPLATE_PATTERN = re.compile(r'\s*Learn more about the.*?policy\.\s*', re.IGNORECASE)

def strip_boilerplate(text: str) -> str:
    """Removes the 'Learn more about the <X> policy.' CTA link caption left over from scraping."""
    return BOILERPLATE_PATTERN.sub(' ', text).strip()

def create_chunks(sections: List[Dict], metadata: Dict, max_tokens: int = 500) -> List[Dict]:
    chunks = []
    chunk_index = 0

    versioned_doc_id = f"{metadata['doc_id']}_{metadata['downloaded_at'][:10]}"

    for section in sections:
        text = strip_boilerplate(section['text'])

        if not text or len(text.strip()) < 20:
            continue
        
        hierarchy_text = ' > '.join(filter(None, section['hierarchy']))
        chunk_text = f"[{hierarchy_text}]\n\n{text}"
        
        # Get section-specific URL from metadata or fall back to document URL
        section_url = get_policy_url(section['section'], metadata)
        
        estimated_tokens = len(chunk_text.split()) / 0.75
        
        if estimated_tokens > max_tokens:
            paragraphs = text.split('\n\n')
            current_chunk = []
            current_size = 0
            
            for para in paragraphs:
                para_tokens = len(para.split())
                
                if current_size + para_tokens > max_tokens and current_chunk:
                    chunk_content = f"[{hierarchy_text}]\n\n" + '\n\n'.join(current_chunk)
                    chunks.append({
                        'chunk_id': str(uuid.uuid4()),
                        'doc_id': versioned_doc_id,
                        'chunk_index': chunk_index,
                        'chunk_text': chunk_content,
                        'policy_section': section['section'],
                        'policy_section_level': section['level'],
                        'policy_path': hierarchy_text,
                        'doc_url': section_url,
                        'platform': metadata['platform'],
                        'category': metadata['category']
                    })
                    chunk_index += 1
                    current_chunk = [para]
                    current_size = para_tokens
                else:
                    current_chunk.append(para)
                    current_size += para_tokens
            
            if current_chunk:
                chunk_content = f"[{hierarchy_text}]\n\n" + '\n\n'.join(current_chunk)
                chunks.append({
                    'chunk_id': str(uuid.uuid4()),
                    'doc_id': versioned_doc_id,
                    'chunk_index': chunk_index,
                    'chunk_text': chunk_content,
                    'policy_section': section['section'],
                    'policy_section_level': section['level'],
                    'policy_path': hierarchy_text,
                    'doc_url': section_url,
                    'platform': metadata['platform'],
                    'category': metadata['category']
                })
                chunk_index += 1
        else:
            chunks.append({
                'chunk_id': str(uuid.uuid4()),
                'doc_id': versioned_doc_id,
                'chunk_index': chunk_index,
                'chunk_text': chunk_text,
                'policy_section': section['section'],
                'policy_section_level': section['level'],
                'policy_path': hierarchy_text,
                'doc_url': section_url,
                'platform': metadata['platform'],
                'category': metadata['category']
            })
            chunk_index += 1
    
    return chunks

def save_chunks(chunks: List[Dict], doc_id: str):
    output_dir = Path(__file__).parent.parent / "data" / "processed_chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{doc_id}_chunks.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(chunks)} chunks to {output_file.name}")

def process_all_documents():
    print("Starting chunking process...")
    
    raw_docs_dir = Path(__file__).parent.parent / "data" / "raw_docs"
    
    md_files = list(raw_docs_dir.glob("*.md"))
    
    total_chunks = 0
    
    for md_file in md_files:
        doc_name = md_file.stem
        metadata_file = raw_docs_dir / f"{doc_name}_metadata.json"
        
        if not metadata_file.exists():
            print(f"Skipping {md_file.name}: no metadata file")
            continue
        
        print(f"\nProcessing: {doc_name}")
        
        content = load_markdown_file(md_file)
        metadata = load_metadata(metadata_file)
        
        sections = extract_sections(content)
        print(f"  Extracted {len(sections)} sections")
        
        chunks = create_chunks(sections, metadata)
        print(f"  Created {len(chunks)} chunks")
        
        save_chunks(chunks, metadata['doc_id'])
        
        total_chunks += len(chunks)
    
    print(f"\nChunking complete. Total chunks: {total_chunks}")

if __name__ == "__main__":
    process_all_documents()
