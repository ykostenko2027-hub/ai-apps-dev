import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Chunk:
    id: str
    text: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

def parse_frontmatter(content: str) -> tuple[Dict[str, Any], str]:
    metadata = {}
    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            raw_meta = parts[1].strip()
            body = parts[2].strip()
            for line in raw_meta.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    metadata[key.strip()] = val.strip().strip('"').strip("'")
    return metadata, body

def load_documents(docs_dir: str = "docs") -> List[Document]:
    documents = []
    if not os.path.exists(docs_dir):
        return documents

    for filename in sorted(os.listdir(docs_dir)):
        if not filename.endswith(".md"):
            continue
        filepath = os.path.join(docs_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        metadata, body = parse_frontmatter(content)
        metadata["file_name"] = filename
        if "title" not in metadata:
            metadata["title"] = filename.replace(".md", "")

        doc_id = filename.replace(".md", "")
        documents.append(Document(id=doc_id, content=body, metadata=metadata))

    return documents

def split(body: str, source: str, metadata: Dict[str, Any]) -> List[Chunk]:
    chunks = []
    sections = re.split(r'\n(?=#{1,3}\s+)', body.strip())

    for idx, sec in enumerate(sections):
        sec = sec.strip()
        if not sec:
            continue

        lines = sec.split('\n', 1)
        header_match = re.match(r'^#{1,3}\s+(.+)', lines[0])
        section_title = header_match.group(1).strip() if header_match else "Загальне"

        chunk_meta = dict(metadata)
        chunk_meta["section"] = section_title
        chunk_meta["source"] = source
        chunk_meta["file_name"] = source

        chunk_id = f"{source}#{idx}"
        chunks.append(Chunk(
            id=chunk_id,
            text=sec,
            source=source,
            metadata=chunk_meta
        ))

    return chunks

def chunk_document(doc: Document) -> List[Chunk]:
    return split(doc.content, doc.metadata.get("file_name", doc.id), doc.metadata)

def load_chunks(docs_dir: str = "docs") -> List[Chunk]:
    docs = load_documents(docs_dir)
    chunks = []
    for doc in docs:
        chunks.extend(chunk_document(doc))
    return chunks