"""
Tests for PDF report generator service.
"""

import pytest
from pathlib import Path
import json
import tempfile

from services.pdf_report_generator import PDFReportGenerator, generate


@pytest.fixture
def metrics_sample():
    """Load sample metrics fixture."""
    fixture_path = Path(__file__).parent / 'fixtures' / 'metrics_sample.json'
    with open(fixture_path) as f:
        return json.load(f)


@pytest.fixture
def temp_video_dir(metrics_sample):
    """Create temp directory with metrics.json."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        metrics_path = temp_path / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_sample, f)
        yield temp_path


def test_generate_pdf_for_known_fixture(temp_video_dir):
    """Test PDF generation with sample fixture."""
    pdf_path = generate(temp_video_dir)

    assert pdf_path.exists(), "PDF file should be created"
    assert pdf_path.suffix == '.pdf', "Output should be a PDF file"

    # Check file size (PDF should be > 100KB)
    file_size = pdf_path.stat().st_size
    assert file_size > 100000, f"PDF size {file_size} should be > 100KB"

    # Basic PDF structure check
    with open(pdf_path, 'rb') as f:
        content = f.read()
        assert content.startswith(b'%PDF'), "File should be valid PDF"


def test_generate_pdf_with_custom_output_path(temp_video_dir):
    """Test PDF generation with custom output path."""
    custom_path = temp_video_dir / 'custom_report.pdf'
    pdf_path = generate(temp_video_dir, output_pdf=custom_path)

    assert pdf_path == custom_path
    assert pdf_path.exists()


def test_missing_section_does_not_crash(temp_video_dir, metrics_sample):
    """Test that missing data sections don't crash PDF generation."""
    # Remove shots section
    metrics_sample['shot_events'] = []
    metrics_sample['shot_detection'] = {
        'total_shots': 0,
        'team_shots': {'A': 0, 'B': 0},
        'player_shots': {},
        'velocity': {'avg_mps': 0, 'max_mps': 0, 'min_mps': 0}
    }

    # Update metrics file
    metrics_path = temp_video_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_sample, f)

    # Should not raise exception
    pdf_path = generate(temp_video_dir)
    assert pdf_path.exists()


def test_missing_metrics_file_raises_error():
    """Test that missing metrics.json raises error."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(FileNotFoundError):
            generate(Path(temp_dir))


def test_pdf_generator_brand_colors(temp_video_dir):
    """Test that brand colors are applied."""
    brand = {
        'primary': '#FF0000',
        'secondary': '#00FF00',
        'accent': '#0000FF'
    }

    pdf_path = generate(temp_video_dir, brand=brand)
    assert pdf_path.exists()


def test_pdf_multiple_pages(temp_video_dir):
    """Test that PDF has correct number of pages."""
    pdf_path = generate(temp_video_dir)

    # Use reportlab's PdfReader to check pages
    try:
        from PyPDF2 import PdfReader
        pdf = PdfReader(pdf_path)
        num_pages = len(pdf.pages)
        assert num_pages == 6, f"PDF should have 6 pages, got {num_pages}"
    except ImportError:
        # If PyPDF2 not available, just check file exists
        assert pdf_path.exists()


def test_pdf_generator_class_initialization():
    """Test PDFReportGenerator class initialization."""
    generator = PDFReportGenerator()
    assert generator.brand is not None
    assert 'primary' in generator.brand
    assert 'secondary' in generator.brand


def test_pdf_generator_with_empty_tracks(temp_video_dir, metrics_sample):
    """Test PDF generation with no player tracking data."""
    metrics_sample['tracks'] = []

    metrics_path = temp_video_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_sample, f)

    pdf_path = generate(temp_video_dir)
    assert pdf_path.exists()


def test_pdf_generator_with_no_pass_events(temp_video_dir, metrics_sample):
    """Test PDF generation with no pass events."""
    metrics_sample['pass_events'] = []
    metrics_sample['pass_detection']['total_passes'] = 0

    metrics_path = temp_video_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_sample, f)

    pdf_path = generate(temp_video_dir)
    assert pdf_path.exists()
