"""
Real Local RAG Engine with SentenceTransformer + FAISS
100% On-Premise Vector Database and Dynamic Document Ingestion
Zero Cloud Egress - Air-Gap Sovereign Operation

Supports any document type: PDF, DOCX, PPTX, TXT, CSV, MD
Uses local all-MiniLM-L6-v2 for 384-dim semantic embeddings
"""

import os
import re
import io
import hashlib
import logging
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from datetime import datetime

logger = logging.getLogger("RAG_ENGINE")

# ── Lazy-loaded embedding model (loaded once on first use) ─────────────────────
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading local SentenceTransformer (all-MiniLM-L6-v2)...")
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Embedding model loaded successfully (100% local, no WAN)")
        except Exception as e:
            logger.warning(f"SentenceTransformer unavailable ({e}), using hash projection fallback")
    return _embedder


# ── Document parsers ────────────────────────────────────────────────────────────

def _parse_pdf(file_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        parts = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                parts.append(f"[Page {i+1}]\n{text.strip()}")
        return "\n\n".join(parts)
    except Exception as e:
        logger.error(f"PDF parse error: {e}")
        return ""


def _parse_docx(file_bytes: bytes) -> str:
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        parts = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                parts.append(text)
        return "\n\n".join(parts)
    except Exception as e:
        logger.error(f"DOCX parse error: {e}")
        return ""


def _parse_pptx(file_bytes: bytes) -> str:
    try:
        from pptx import Presentation
        prs = Presentation(io.BytesIO(file_bytes))
        parts = []
        for i, slide in enumerate(prs.slides):
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text.append(shape.text.strip())
            if slide_text:
                parts.append(f"[Slide {i+1}]\n" + "\n".join(slide_text))
        return "\n\n".join(parts)
    except Exception as e:
        logger.error(f"PPTX parse error: {e}")
        return ""


def _parse_text(file_bytes: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return file_bytes.decode(enc)
        except Exception:
            continue
    return ""


def parse_document(filename: str, file_bytes: bytes) -> str:
    """Parse any supported file type to plain text."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
    if ext == "pdf":
        return _parse_pdf(file_bytes)
    elif ext in ("docx", "doc"):
        return _parse_docx(file_bytes)
    elif ext in ("pptx", "ppt"):
        return _parse_pptx(file_bytes)
    elif ext in ("txt", "md", "csv", "log", "yaml", "json", "xml"):
        return _parse_text(file_bytes)
    else:
        # Try text decode as last resort
        return _parse_text(file_bytes)


# ── Text chunking ───────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 60) -> List[str]:
    """
    Chunk text into overlapping windows at sentence/paragraph boundaries.
    Preserves context near boundaries so thresholds & limits stay in one chunk.
    """
    # Split by paragraphs first
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    
    chunks = []
    current = []
    current_len = 0
    
    for para in paragraphs:
        words = para.split()
        if current_len + len(words) > chunk_size and current:
            # Save current chunk
            chunks.append(" ".join(current))
            # Keep overlap
            overlap_words = current[-overlap:] if len(current) > overlap else current[:]
            current = overlap_words
            current_len = len(overlap_words)
        current.extend(words)
        current_len += len(words)
    
    if current:
        chunks.append(" ".join(current))
    
    # Safety: if any single paragraph is massive, split it by sentences
    result = []
    for chunk in chunks:
        if len(chunk.split()) > chunk_size * 2:
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            sub = []
            sub_len = 0
            for sent in sentences:
                sw = sent.split()
                if sub_len + len(sw) > chunk_size and sub:
                    result.append(" ".join(sub))
                    sub = sw[-overlap:] if len(sw) > overlap else sw[:]
                    sub_len = len(sub)
                sub.extend(sw)
                sub_len += len(sw)
            if sub:
                result.append(" ".join(sub))
        else:
            result.append(chunk)
    
    return [c for c in result if len(c.strip()) > 20]


# ── SHA-256 checksum ─────────────────────────────────────────────────────────────

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── Fallback hash-projection embedder (when sentence-transformers not available) ──

def _hash_embed(text: str, dim: int = 384) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    words = re.findall(r"\w+", text.lower())
    for word in words:
        h = hash(word)
        i1 = abs(h) % dim
        i2 = abs(h >> 7) % dim
        sign = 1.0 if (h & 1) else -1.0
        vec[i1] += sign * 1.5
        vec[i2] += 0.75
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-6 else vec


# ── Main RAG engine ──────────────────────────────────────────────────────────────

class LocalFaissRag:
    EMBED_DIM = 384  # all-MiniLM-L6-v2

    def __init__(self):
        self.index = faiss.IndexFlatIP(self.EMBED_DIM)
        self.chunks: List[Dict[str, Any]] = []   # each entry = one chunk
        self.documents: List[Dict[str, Any]] = []  # document-level metadata
        self._init_seed_knowledge()

    # ── Embedding ──────────────────────────────────────────────────────────────

    def _embed(self, texts: List[str]) -> np.ndarray:
        embedder = _get_embedder()
        if embedder is not None:
            vecs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return vecs.astype(np.float32)
        else:
            # Fallback: hash projection
            vecs = np.array([_hash_embed(t, self.EMBED_DIM) for t in texts], dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms < 1e-6, 1.0, norms)
            return (vecs / norms).astype(np.float32)

    def _embed_one(self, text: str) -> np.ndarray:
        return self._embed([text])[0]

    # ── Seed knowledge base ───────────────────────────────────────────────────

    def _init_seed_knowledge(self):
        """Seed the pre-loaded motor SOPs so the system works without any upload."""
        seed_docs = [
            {
                "doc_id": "SOP-CNC-402",
                "filename": "CNC Machine Maintenance SOP.pdf",
                "section": "Section 4.2: Spindle Motor Baseline Standards",
                "classification": "Proprietary Tier 1",
                "content": (
                    "Standard Operating Procedure for CNC Spindle Motor-014. "
                    "Continuous operating vibration limit is 3.5 mm/s RMS. "
                    "Stage-1 warning threshold is 4.2 mm/s RMS. "
                    "Critical threshold is 4.5 mm/s RMS. Any reading above 4.5 mm/s "
                    "indicates structural bearing degradation and mandates scheduled "
                    "bearing replacement within 72 operating hours to avoid catastrophic spindle seizure."
                ),
            },
            {
                "doc_id": "HIST-MTR-018",
                "filename": "Spindle Motor Failure History.pdf",
                "section": "Incident Case #18: Inner Bearing Race Micro-Spalling",
                "classification": "Confidential Logs",
                "content": (
                    "Case History Motor-014 (Past Incident): Spindle exhibited 4.8 mm/s vibration "
                    "at 1480 RPM with primary 120 Hz harmonic resonance and elevated temperature of 71.4 degrees C. "
                    "Post-mortem teardown confirmed inner race fatigue flaking and micro-pitting. "
                    "Recommended action: Swift bearing replacement prevents complete stator-rotor rub."
                ),
            },
            {
                "doc_id": "SPEC-BRG-6208",
                "filename": "Hydraulic & Bearing Tolerances.xlsx",
                "section": "Sheet 3: Deep Groove & Cylindrical Roller Bearings",
                "classification": "OEM Blueprint",
                "content": (
                    "Bearing Specification 6208-2RS / C3: Maximum permissible continuous temperature "
                    "is 75.0 degrees C. Temperature warning trigger at 70.0 degrees C. "
                    "Thermal gradient above 71.0 degrees C coupled with vibration > 4.5 mm/s "
                    "confirms lubricating grease dry-out and metal-to-metal race contact."
                ),
            },
            {
                "doc_id": "PID-LINE3-OPC",
                "filename": "Plant Line 3 P&ID Diagram.pptx",
                "section": "Slide 14: Sensor Instrumentation & OPC-UA Tag Mappings",
                "classification": "Restricted Layout",
                "content": (
                    "Line 3 CNC Milling Center Telemetry Map: Motor-014 vibration accelerometer "
                    "is registered on local OPC-UA server node: ns=2;s=Motor014.Vibration. "
                    "Thermal infrared sensor is mapped to ns=2;s=Motor014.Temperature. "
                    "Tachometer is mapped to ns=2;s=Motor014.RPM. Read operations classified as LOW RISK."
                ),
            },
        ]

        for d in seed_docs:
            text = d["content"]
            chunks = _chunk_text(text, chunk_size=200)
            sha = _sha256(text.encode())
            # Register document
            doc_entry = {
                "doc_id": d["doc_id"],
                "filename": d["filename"],
                "section": d["section"],
                "classification": d["classification"],
                "sha256": sha[:16] + "...",
                "total_chunks": len(chunks),
                "ingested_at": "pre-seeded",
                "file_size_kb": round(len(text) / 1024, 2),
            }
            self.documents.append(doc_entry)
            
            # Index each chunk
            for ci, chunk in enumerate(chunks):
                self.chunks.append({
                    "doc_id": d["doc_id"],
                    "filename": d["filename"],
                    "section": d["section"],
                    "classification": d["classification"],
                    "content": chunk,
                    "chunk_index": ci,
                    "sha256": sha[:16] + "...",
                })
        
        # Embed and add all seed chunks at once
        if self.chunks:
            texts = [c["content"] for c in self.chunks]
            try:
                vecs = self._embed(texts)
                self.index.add(vecs)
                logger.info(f"Seed knowledge indexed: {len(self.chunks)} chunks, "
                            f"{len(self.documents)} documents in FAISS.")
            except Exception as e:
                logger.error(f"Failed to index seed knowledge: {e}")

    # ── Dynamic document ingest ───────────────────────────────────────────────

    def ingest_file(
        self,
        filename: str,
        file_bytes: bytes,
        classification: str = "User Ingested",
    ) -> Dict[str, Any]:
        """Parse any file type, chunk it, embed it, and add it to FAISS."""
        sha = _sha256(file_bytes)
        
        # Check if already ingested (by sha)
        for doc in self.documents:
            if doc.get("sha256_full") == sha:
                return {"status": "already_indexed", "doc": doc}
        
        # Parse text content
        raw_text = parse_document(filename, file_bytes)
        if not raw_text.strip():
            return {"status": "error", "detail": "Could not extract text from file."}
        
        # Generate doc_id
        doc_id = f"INGEST-{len(self.documents)+1:03d}"
        
        # Chunk the text
        chunks = _chunk_text(raw_text, chunk_size=400, overlap=60)
        if not chunks:
            return {"status": "error", "detail": "No usable text chunks found."}
        
        # Embed all chunks
        try:
            vecs = self._embed(chunks)
        except Exception as e:
            return {"status": "error", "detail": f"Embedding failed: {e}"}
        
        # Register document metadata
        doc_entry = {
            "doc_id": doc_id,
            "filename": filename,
            "section": "User Uploaded Document",
            "classification": classification,
            "sha256": sha[:16] + "...",
            "sha256_full": sha,
            "total_chunks": len(chunks),
            "ingested_at": datetime.now().isoformat(),
            "file_size_kb": round(len(file_bytes) / 1024, 2),
            "char_count": len(raw_text),
        }
        self.documents.append(doc_entry)
        
        # Register and index chunks
        for ci, (chunk, vec) in enumerate(zip(chunks, vecs)):
            self.chunks.append({
                "doc_id": doc_id,
                "filename": filename,
                "section": f"Chunk {ci+1}/{len(chunks)}",
                "classification": classification,
                "content": chunk,
                "chunk_index": ci,
                "sha256": sha[:16] + "...",
            })
        
        self.index.add(vecs)
        logger.info(f"Ingested '{filename}': {len(chunks)} chunks added to FAISS. "
                    f"Total vectors: {self.index.ntotal}")
        
        return {
            "status": "success",
            "doc_id": doc_id,
            "filename": filename,
            "total_chunks": len(chunks),
            "sha256": sha[:16] + "...",
            "doc": doc_entry,
        }

    # ── Semantic search ───────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 3,
        doc_filter: str = None,  # filter by doc_id or filename substring
    ) -> List[Dict[str, Any]]:
        """Semantic FAISS search. Returns top_k most relevant chunks."""
        if self.index.ntotal == 0:
            return []
        
        q_vec = self._embed_one(query).reshape(1, -1).astype(np.float32)
        k = min(top_k * 3, self.index.ntotal)  # retrieve more then filter
        scores, indices = self.index.search(q_vec, k)
        
        results = []
        seen_content = set()
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = dict(self.chunks[idx])
            
            # Optional document filter
            if doc_filter:
                if (doc_filter.lower() not in chunk.get("filename", "").lower() and
                        doc_filter.lower() not in chunk.get("doc_id", "").lower()):
                    continue
            
            # Deduplicate very similar content
            key = chunk["content"][:80]
            if key in seen_content:
                continue
            seen_content.add(key)
            
            chunk["similarity_score"] = round(float(score), 4)
            results.append(chunk)
            
            if len(results) >= top_k:
                break
        
        return results

    def search_all_docs(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search across all documents — convenience wrapper."""
        return self.search(query, top_k=top_k)

    def get_document_context(self, doc_id_or_name: str, max_chunks: int = 8) -> str:
        """
        Retrieve the most relevant chunks for a document as a single
        concatenated context string. Useful for feeding to the LLM.
        """
        chunks = [
            c for c in self.chunks
            if (c.get("doc_id") == doc_id_or_name or
                doc_id_or_name.lower() in c.get("filename", "").lower())
        ]
        if not chunks:
            return ""
        # Return first max_chunks by order
        return "\n\n---\n\n".join(c["content"] for c in chunks[:max_chunks])

    def list_documents(self) -> List[Dict[str, Any]]:
        return list(self.documents)


# Global Singleton instance
rag_service = LocalFaissRag()


if __name__ == "__main__":
    res = rag_service.search("safe vibration limit for Motor-014", top_k=2)
    print("FAISS Search Results:")
    for r in res:
        print(f"[{r['similarity_score']:.4f}] {r['filename']} — {r['section']}")
        print(f"  {r['content'][:120]}...")
