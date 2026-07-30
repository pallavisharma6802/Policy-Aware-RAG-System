import os
import time
import json
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime

POLICY_URLS = {
    "google_ads_base_hub": {
        "url": "https://support.google.com/adspolicy/answer/6008942",
        "platform": "google",
        "category": "overview"
    },
    "google_ads_misrepresentation": {
        "url": "https://support.google.com/adspolicy/answer/6020955",
        "platform": "google",
        "category": "misrepresentation"
    },
    "google_ads_restricted_products": {
        "url": "https://support.google.com/adspolicy/answer/6014299",
        "platform": "google",
        "category": "restricted"
    },
    "google_ads_prohibited_content": {
        "url": "https://support.google.com/adspolicy/answer/6015406",
        "platform": "google",
        "category": "prohibited"
    },
    "google_ads_editorial_technical": {
        "url": "https://support.google.com/adspolicy/answer/6021546",
        "platform": "google",
        "category": "editorial"
    }
}

def fetch_page(url, delay=2):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        time.sleep(delay)
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_structured_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.decompose()
    
    structured_lines = []
    current_section = None
    
    for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'ul', 'ol']):
        text = element.get_text(strip=True)
        
        if not text:
            continue
        
        if element.name in ['h1', 'h2', 'h3', 'h4']:
            level = element.name[1]
            current_section = text
            structured_lines.append(f"\n[SECTION-H{level}] {text}\n")
        
        elif element.name == 'li':
            structured_lines.append(f"  - {text}")
        
        elif element.name == 'p':
            if current_section:
                structured_lines.append(f"{text}")
            else:
                structured_lines.append(text)
        
        elif element.name in ['ul', 'ol']:
            continue
    
    return "\n".join(structured_lines)

def extract_metadata(soup, url, platform, category):
    metadata = {
        "doc_id": f"{platform}_{category}",
        "url": url,
        "platform": platform,
        "category": category,
        "downloaded_at": datetime.now().isoformat(),
        "title": None,
        "sections": [],
        "section_urls": {}  #maps section names to their specific URLs
    }
    
    title_tag = soup.find('title')
    if title_tag:
        metadata["title"] = title_tag.get_text(strip=True)
    
    h1_tag = soup.find('h1')
    if h1_tag and not metadata["title"]:
        metadata["title"] = h1_tag.get_text(strip=True)
    
    for header in soup.find_all(['h1', 'h2', 'h3', 'h4']):
        section_text = header.get_text(strip=True)
        if section_text:
            metadata["sections"].append({
                "section": section_text,
                "policy_level": header.name.upper()
            })

            link = header.find('a', href=True)
            if not link:
                next_elem = header.find_next_sibling()
                if next_elem:
                    link = next_elem.find('a', href=True)

            if not link:
                current = header.find_next()
                search_limit = 3
                while current and search_limit > 0:
                    if current.name in ['h1', 'h2', 'h3', 'h4']:
                        break

                    links = current.find_all('a', href=True) if hasattr(current, 'find_all') else []
                    for l in links:
                        href = l.get('href', '')
                        if 'adspolicy/answer/' in href and 'answer_' not in href:
                            if href.startswith('http'):
                                link = l
                            elif href.startswith('/'):
                                link = l
                                href = f"https://support.google.com{href}"
                                l['href'] = href
                            break

                    if link:
                        break
                    current = current.find_next()
                    search_limit -= 1

            if link and link.get('href'):
                href = link.get('href')
                if '?' in href:
                    href = href.split('?')[0]
                if 'adspolicy/answer/' in href:
                    metadata["section_urls"][section_text] = href
    
    return metadata

def save_document(content, filename, doc_type="html"):
    base_dir = Path(__file__).parent.parent / "data" / "raw_docs"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = base_dir / f"{filename}.{doc_type}"
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"Saved: {filepath.name}")
    return filepath

def download_policies():
    print("Starting policy document download...")
    
    for policy_name, policy_info in POLICY_URLS.items():
        url = policy_info["url"]
        platform = policy_info["platform"]
        category = policy_info["category"]
        
        print(f"\nFetching: {policy_name}")
        
        html_content = fetch_page(url)
        
        if html_content:
            save_document(html_content, f"{policy_name}_raw", "html")
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            text_content = extract_structured_text(html_content)
            save_document(text_content, policy_name, "md")
            
            metadata = extract_metadata(soup, url, platform, category)
            save_document(json.dumps(metadata, indent=2), f"{policy_name}_metadata", "json")
        else:
            print(f"Failed to download: {policy_name}")
    
    print("\nDownload complete.")

if __name__ == "__main__":
    download_policies()
