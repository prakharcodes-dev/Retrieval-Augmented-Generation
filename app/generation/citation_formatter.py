from typing import List, Dict, Any


class CitationFormatter:
    """
    Extracts and formats verified source citations strictly from retrieved chunk metadata.
    Prevents fake citations, fake filenames, or fabricated page numbers.
    """

    @staticmethod
    def extract_citations(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extracts unique citation metadata from retrieved context chunks."""
        citations = []
        seen = set()

        for chunk in chunks:
            meta = chunk.get("metadata", {})
            source = meta.get("source", meta.get("filename", "Unknown Document"))
            page = meta.get("page_number", meta.get("page", None))

            # Normalize page value
            if page == "" or page == "None" or page is None:
                page = None
            else:
                try:
                    page = int(page)
                except (ValueError, TypeError):
                    pass

            key = (str(source), page)
            if key not in seen and source != "Unknown Document":
                seen.add(key)
                citations.append({
                    "source": str(source),
                    "page_number": page,
                    "chunk_id": chunk.get("chunk_id", "")
                })

        return citations

    @classmethod
    def format_citations_block(cls, citations: List[Dict[str, Any]]) -> str:
        """Formats extracted citations into a clean numbered markdown list block."""
        if not citations:
            return ""

        formatted_lines = []
        for idx, cite in enumerate(citations, 1):
            source = cite.get("source", "Document")
            page = cite.get("page_number")

            if page is not None and str(page).isdigit():
                formatted_lines.append(f"[{idx}] {source} — Page {page}")
            else:
                formatted_lines.append(f"[{idx}] {source}")

        return "\n".join(formatted_lines)

    @classmethod
    def format_citations_badges(cls, citations: List[Dict[str, Any]]) -> str:
        """Formats extracted citations into clean styled HTML badge pills for modern UI rendering."""
        if not citations:
            return ""

        badges = []
        for cite in citations:
            source = cite.get("source", "Document")
            page = cite.get("page_number")
            if page is not None and str(page).isdigit():
                label = f"📄 {source} · Page {page}"
            else:
                label = f"📄 {source}"
            badge_html = f'<span class="citation-pill">{label}</span>'
            badges.append(badge_html)

        return "".join(badges)
