"""
Fixture generator: creates sample test files for ingestion tests.

Run this script once to generate sample DOCX, TXT, and PDF files
in the tests/fixtures/ directory.
"""

from pathlib import Path


FIXTURES_DIR = Path(__file__).parent


def create_sample_txt():
    """Create a sample .txt file with varied content."""
    content = """The Evolution of Artificial Intelligence

Artificial intelligence (AI) has undergone remarkable transformations since its inception in the 1950s. What began as a theoretical exploration of machine thinking has evolved into one of the most transformative technologies of the 21st century.

Early Foundations (1950-1970)
The field was founded at the Dartmouth Conference in 1956, where researchers like John McCarthy, Marvin Minsky, and Claude Shannon laid the groundwork for AI research. Early programs could play checkers and solve algebra problems, sparking tremendous optimism.

The AI Winters (1970-1990)
Progress stalled twice during periods known as "AI winters," when funding dried up due to unmet expectations. The limitations of early approaches became apparent as researchers struggled with the complexity of real-world problems.

Machine Learning Revolution (1990-2010)
The shift from rule-based systems to statistical machine learning marked a turning point. Algorithms like support vector machines, random forests, and neural networks began showing practical results in areas like spam detection and image classification.

Deep Learning Era (2010-Present)
The combination of large datasets, powerful GPUs, and deep neural network architectures led to breakthroughs in computer vision, natural language processing, and game playing. Landmark achievements include AlphaGo's victory over Go champion Lee Sedol and the development of large language models.

Current Trends
Today, AI research focuses on areas like:
- Retrieval-Augmented Generation (RAG)
- Multimodal AI systems
- AI safety and alignment
- Efficient inference and edge deployment
- Agentic AI systems

The future of AI promises even more transformative advances, with potential impacts across healthcare, education, scientific research, and creative industries.
"""
    filepath = FIXTURES_DIR / "sample.txt"
    filepath.write_text(content, encoding="utf-8")
    print(f"Created: {filepath}")


def create_sample_docx():
    """Create a sample .docx file with paragraphs and a table."""
    try:
        from docx import Document
        from docx.shared import Inches

        doc = Document()
        doc.add_heading("Document Processing Pipeline", level=1)

        doc.add_paragraph(
            "This document serves as a test fixture for the Intelligent Document Q&A Engine. "
            "It contains multiple paragraphs, headings, and a table to verify the DOCX "
            "ingestion pipeline."
        )

        doc.add_heading("Overview", level=2)
        doc.add_paragraph(
            "The pipeline supports three document formats: PDF, DOCX, and TXT. "
            "Each format has a dedicated reader that extracts text content and "
            "metadata from the document."
        )

        doc.add_heading("Chunking Strategies", level=2)
        doc.add_paragraph(
            "After ingestion, documents are split into smaller chunks using one of "
            "three strategies: recursive character splitting, semantic chunking, "
            "or sliding-window chunking. Each strategy has different trade-offs "
            "between speed, cost, and semantic coherence."
        )

        doc.add_heading("Comparison Table", level=3)
        table = doc.add_table(rows=4, cols=3)
        table.style = "Light List Accent 1"

        # Header row
        header = table.rows[0].cells
        header[0].text = "Strategy"
        header[1].text = "Speed"
        header[2].text = "Coherence"

        # Data rows
        data = [
            ("Recursive", "Fast", "Good"),
            ("Semantic", "Slow", "Excellent"),
            ("Sliding Window", "Fast", "Moderate"),
        ]
        for i, (strategy, speed, coherence) in enumerate(data, start=1):
            row = table.rows[i].cells
            row[0].text = strategy
            row[1].text = speed
            row[2].text = coherence

        doc.add_heading("Conclusion", level=2)
        doc.add_paragraph(
            "The choice of chunking strategy depends on the specific use case, "
            "the nature of the documents, and the available budget for API calls. "
            "Recursive chunking is recommended as the default strategy for most "
            "applications."
        )

        filepath = FIXTURES_DIR / "sample.docx"
        doc.save(str(filepath))
        print(f"Created: {filepath}")

    except ImportError:
        print("python-docx not installed, skipping DOCX fixture generation.")


def create_sample_pdf():
    """Create a sample PDF by writing a minimal PDF manually."""
    # Minimal valid PDF with text content
    filepath = FIXTURES_DIR / "sample.pdf"

    # We'll create a simple text-based PDF using reportlab if available,
    # otherwise create a note explaining manual creation is needed.
    try:
        from PyPDF2 import PdfWriter
        from io import BytesIO

        # Create a minimal PDF using PyPDF2 (limited text support)
        # For a proper test PDF, it's better to use a real PDF file.
        # We'll write a note file instead.
        note_path = FIXTURES_DIR / "README_fixtures.md"
        note_path.write_text(
            "# Test Fixtures\n\n"
            "- `sample.txt` — Auto-generated text file\n"
            "- `sample.docx` — Auto-generated Word document\n"
            "- `sample.pdf` — Place a real PDF file here for testing\n\n"
            "To generate fixtures, run:\n"
            "```bash\n"
            "python -m tests.fixtures.create_fixtures\n"
            "```\n",
            encoding="utf-8",
        )
        print(f"Created: {note_path}")
        print("Note: Place a real PDF file as 'sample.pdf' for PDF testing.")

    except ImportError:
        print("PyPDF2 not installed, skipping PDF fixture note.")


if __name__ == "__main__":
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    create_sample_txt()
    create_sample_docx()
    create_sample_pdf()
    print("\nFixture generation complete!")
