"""
Generate Report Dashboard Page - Create and download PDF match reports.

Provides UI for selecting matches, configuring report sections,
generating PDFs, and downloading recent reports.
"""

import streamlit as st
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.pdf_report_generator import generate


def get_processed_matches() -> Dict[str, Path]:
    """
    Find all processed matches with metrics.json.

    Returns:
        Dict mapping match name to metrics directory path
    """
    matches = {}
    base_dir = Path('demo/demo_outputs')

    if not base_dir.exists():
        return matches

    for match_dir in base_dir.iterdir():
        if not match_dir.is_dir():
            continue

        metrics_path = match_dir / 'metrics.json'
        if metrics_path.exists():
            matches[match_dir.name] = match_dir

    return matches


def get_recent_reports(video_dir: Path, limit: int = 10) -> List[Dict[str, any]]:
    """
    Get list of recently generated reports in a directory.

    Args:
        video_dir: Directory containing PDFs
        limit: Maximum number of reports to return

    Returns:
        List of dicts with 'path', 'name', 'size', 'date'
    """
    reports = []
    pattern = video_dir.glob('report_*.pdf')

    for pdf_path in sorted(pattern, key=lambda p: p.stat().st_mtime, reverse=True):
        if len(reports) >= limit:
            break

        stat = pdf_path.stat()
        reports.append({
            'path': pdf_path,
            'name': pdf_path.name,
            'size': stat.st_size,
            'date': datetime.fromtimestamp(stat.st_mtime)
        })

    return reports


def render_generate_report():
    """Main render function for Generate Report page."""
    st.title("Generate Match Report")
    st.markdown("Create comprehensive PDF reports from match metrics.")

    # Get processed matches
    matches = get_processed_matches()

    if not matches:
        st.warning("No processed matches found. Process a video first.")
        return

    # Match Selection
    st.header("1. Select Match")
    match_name = st.selectbox(
        "Choose a processed match:",
        options=sorted(matches.keys()),
        help="Select a match to generate a report for"
    )

    if not match_name:
        st.info("Please select a match")
        return

    match_dir = matches[match_name]

    # Report Sections Configuration
    st.header("2. Report Sections")
    st.markdown("Select which sections to include in the report:")

    col1, col2 = st.columns(2)

    with col1:
        include_executive = st.checkbox("Executive Summary", value=True)
        include_tactical = st.checkbox("Tactical Analysis", value=True)
        include_shooting = st.checkbox("Shooting & xG", value=True)

    with col2:
        include_physical = st.checkbox("Physical Metrics", value=True)
        include_highlights = st.checkbox("Highlights", value=True)

    sections_config = {
        'executive': include_executive,
        'tactical': include_tactical,
        'shooting': include_shooting,
        'physical': include_physical,
        'highlights': include_highlights
    }

    # Generate PDF Button
    st.header("3. Generate")

    pdf_bytes = None
    pdf_path = None

    if st.button("Generate PDF Report", key="generate_btn", use_container_width=True):
        with st.spinner("Generating PDF report..."):
            try:
                pdf_path = generate(match_dir)

                # Read PDF bytes for download
                with open(pdf_path, 'rb') as f:
                    pdf_bytes = f.read()

                st.success(f"PDF generated: {pdf_path.name}")

                # Show file info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("File Size", f"{len(pdf_bytes) / 1024:.1f} KB")
                with col2:
                    st.metric("Pages", "6")
                with col3:
                    st.metric("Generated", pdf_path.stat().st_mtime)

            except Exception as e:
                st.error(f"Error generating PDF: {e}")

    # Download Button
    if pdf_bytes:
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=pdf_path.name,
            mime="application/pdf",
            use_container_width=True
        )

    # Recent Reports Section
    st.header("4. Recent Reports")

    recent_reports = get_recent_reports(match_dir)

    if recent_reports:
        st.markdown(f"Recently generated PDFs ({len(recent_reports)}):")

        for report in recent_reports:
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

            with col1:
                st.markdown(f"📄 **{report['name']}**")

            with col2:
                st.caption(f"{report['size'] / 1024:.1f} KB")

            with col3:
                st.caption(report['date'].strftime("%m/%d %H:%M"))

            with col4:
                with open(report['path'], 'rb') as f:
                    pdf_data = f.read()
                st.download_button(
                    label="Download",
                    data=pdf_data,
                    file_name=report['name'],
                    mime="application/pdf",
                    key=f"download_{report['path']}"
                )

    else:
        st.info("No previous reports. Generate one to see it here.")

    # Report Preview / Details
    if match_dir and (match_dir / 'metrics.json').exists():
        st.header("5. Match Details")

        with open(match_dir / 'metrics.json') as f:
            metrics = json.load(f)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            possession = metrics.get('possession', {})
            team_poss = possession.get('team_possession_percentage', {})
            st.metric("Team A Possession", f"{team_poss.get('A', 0):.1f}%")

        with col2:
            st.metric("Team B Possession", f"{team_poss.get('B', 0):.1f}%")

        with col3:
            pass_stats = metrics.get('pass_detection', {})
            st.metric("Total Passes", pass_stats.get('total_passes', 0))

        with col4:
            shot_stats = metrics.get('shot_detection', {})
            st.metric("Total Shots", shot_stats.get('total_shots', 0))


if __name__ == '__main__':
    render_generate_report()
