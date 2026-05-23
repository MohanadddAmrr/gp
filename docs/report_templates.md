# PDF Report Templates Documentation

## Overview

The TactiVision PDF Report Generator creates comprehensive, professional match analysis reports from processed video metrics. This document details the available report templates, sections, and customization options.

## Architecture

### Report Generation Pipeline

```
metrics.json → Report Template → PDF Output
     ↓
  Data Processing
     ↓
  Section Rendering
     ↓
  PDF Assembly
```

## Available Report Sections

### 1. Executive Summary

**Purpose:** High-level overview of match performance

**Contents:**
- Match metadata (teams, date, duration)
- Final score and result
- Key performance indicators (KPIs)
- Match timeline overview
- Top 3 pivotal moments

**Data Sources:**
```python
- possession: team_possession_percentage
- score: match_result
- duration_minutes: match_length
- key_events: event_detection
```

**Template Variables:**
```yaml
team_a_name: string
team_b_name: string
team_a_score: integer
team_b_score: integer
match_date: datetime
duration_minutes: float
possession_a: float  # percentage
possession_b: float  # percentage
```

---

### 2. Tactical Analysis

**Purpose:** Detailed tactical breakdown and strategic insights

**Contents:**
- Formation detection results
- Possession map visualization
- Pass network diagram
- Tactical strengths and weaknesses
- Set piece analysis
- Defensive actions summary

**Data Sources:**
```python
- formations_detected: formation_detector output
- possession: possession_tracker
- passes: pass_detection
- set_pieces: set_piece_detector
- defensive_actions: event_detector
```

**Key Metrics:**
```yaml
formation_a: string  # e.g., "4-3-3"
formation_b: string  # e.g., "4-2-3-1"
total_passes: integer
pass_accuracy: float  # percentage
average_pass_length: float  # meters
possession_percentage_a: float
possession_percentage_b: float
```

---

### 3. Shooting & xG Analysis

**Purpose:** Advanced shooting statistics and expected goals metrics

**Contents:**
- Shot map (location-based visualization)
- Shot type distribution (headed, feet, long-range)
- Expected goals (xG) analysis
- Conversion rate analysis
- Key shooter statistics
- Goalkeeper performance metrics

**Data Sources:**
```python
- shots: shot_detection
- xg_analysis: xg_calculator
- shot_quality: shot_quality_model
- goalkeeper_stats: goalkeeper_analyzer
```

**Key Metrics:**
```yaml
total_shots_a: integer
total_shots_b: integer
shots_on_target_a: integer
shots_on_target_b: integer
xg_a: float  # Expected goals team A
xg_b: float  # Expected goals team B
conversion_rate_a: float  # percentage
conversion_rate_b: float  # percentage
top_shooter: string
top_shooter_shots: integer
```

---

### 4. Physical Metrics

**Purpose:** Player physical performance and activity data

**Contents:**
- Sprint analysis (total sprints, sprint distance)
- High-intensity run tracking
- Distance covered by player/team
- Fatigue index analysis
- Physical intensity heatmaps
- Injury risk assessment

**Data Sources:**
```python
- sprints: sprint_detector
- distance_covered: tracking_system
- intensity: physical_analyzer
- fatigue: fatigue_calculator
```

**Key Metrics:**
```yaml
total_sprints_a: integer
total_sprints_b: integer
total_distance_a: float  # kilometers
total_distance_b: float  # kilometers
average_speed_a: float  # km/h
average_speed_b: float  # km/h
high_intensity_runs_a: integer
high_intensity_runs_b: integer
fatigue_index_a: float  # 0-100
fatigue_index_b: float  # 0-100
```

---

### 5. Highlights

**Purpose:** Key moments and highlights from the match

**Contents:**
- Extracted highlight clips
- Goal highlights (if any)
- Near-miss moments
- Controversial decisions
- Exceptional saves/plays
- Key turnovers

**Data Sources:**
```python
- highlights: highlights_generator
- key_events: event_detector
- video_segments: video_exporter
```

**Key Metrics:**
```yaml
total_highlights: integer
goal_count: integer
near_misses: integer
saves_highlighted: integer
controversial_moments: integer
highlight_duration_total: float  # seconds
```

---

## Report Templates

### Standard Template

**File:** `templates/standard_report.html`

**Sections Included:**
1. Cover Page
2. Executive Summary
3. Tactical Analysis
4. Shooting & xG
5. Physical Metrics
6. Highlights
7. Statistics Table
8. Conclusion

**Page Layout:**
- Page Size: A4 (210mm × 297mm)
- Margins: 20mm (all sides)
- Font: Arial, 11pt (body), 16pt (headings)
- Color Scheme: Team colors + neutral grays

**Example Usage:**
```python
from services.pdf_report_generator import generate

pdf_path = generate(
    video_dir=Path("demo/demo_outputs/liverpool_vs_city"),
    template="standard",
    sections=['executive', 'tactical', 'shooting', 'physical', 'highlights']
)
```

---

### Executive Template

**File:** `templates/executive_report.html`

**Sections Included:**
1. Cover Page
2. Executive Summary (expanded)
3. Key Statistics
4. Highlights

**Page Layout:**
- Condensed version for quick review
- Page Size: A4
- Focus on high-level insights
- Estimated Pages: 4-6

**Example Usage:**
```python
pdf_path = generate(
    video_dir=Path("demo/demo_outputs/match"),
    template="executive",
    sections=['executive']
)
```

---

### Detailed Template

**File:** `templates/detailed_report.html`

**Sections Included:**
1. Cover Page
2. Match Overview
3. Executive Summary
4. Tactical Deep Dive
5. Shooting Analysis (detailed)
6. Pass Network Analysis
7. Physical Performance
8. Player Performance Rankings
9. Highlights with Commentary
10. Appendix (raw statistics)

**Page Layout:**
- Comprehensive analysis
- Page Size: A4
- Multiple data visualizations per page
- Estimated Pages: 15-20

**Example Usage:**
```python
pdf_path = generate(
    video_dir=Path("demo/demo_outputs/match"),
    template="detailed",
    sections=['executive', 'tactical', 'shooting', 'physical', 'highlights']
)
```

---

## Data Processing Pipeline

### 1. Metrics Loading

```python
import json
from pathlib import Path

metrics_path = Path("demo/demo_outputs/match/metrics.json")
with open(metrics_path) as f:
    metrics = json.load(f)
```

**Expected metrics.json structure:**
```json
{
  "match_id": "match_20260521_liverpool_vs_city",
  "duration_minutes": 90,
  "team_names": {"A": "Liverpool", "B": "Manchester City"},
  "score": {"A": 2, "B": 1},
  "possession": {
    "team_possession_percentage": {"A": 45.5, "B": 54.5}
  },
  "shot_detection": {
    "total_shots": 25,
    "team_shots": {"A": 10, "B": 15},
    "shots_on_target": {"A": 5, "B": 7}
  },
  "pass_detection": {
    "total_passes": 850,
    "pass_accuracy": 78.5
  },
  "sprint_detection": {
    "total_sprints": 450,
    "team_sprints": {"A": 220, "B": 230}
  },
  "xg_analysis": {
    "xg_a": 1.8,
    "xg_b": 2.1,
    "conversion_rate": 12.5
  }
}
```

### 2. Section Rendering

Each section follows this rendering pipeline:

```python
def render_section(section_type, metrics, template):
    # 1. Extract relevant data
    section_data = extract_section_data(section_type, metrics)
    
    # 2. Process and aggregate
    processed_data = process_data(section_data)
    
    # 3. Generate visualizations
    charts = generate_charts(processed_data)
    
    # 4. Render HTML/PDF
    html = render_html(template, processed_data, charts)
    
    return html
```

### 3. PDF Assembly

```python
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# Create PDF
pdf_path = Path("output/report.pdf")
pages = []

# Add each section as pages
for section in sections:
    pages.extend(render_section(section))

# Combine pages into PDF
create_pdf(pages, pdf_path)
```

---

## Customization Guide

### Adding Custom Sections

**Step 1:** Create section template

```html
<!-- templates/sections/custom_section.html -->
<div class="section">
    <h2>{{ section_title }}</h2>
    <div class="content">
        <!-- Custom HTML here -->
    </div>
</div>
```

**Step 2:** Register section processor

```python
# services/pdf_report_generator.py

SECTION_PROCESSORS = {
    'custom': render_custom_section
}

def render_custom_section(metrics):
    # Extract and process custom data
    custom_data = metrics.get('custom_metrics', {})
    return {
        'title': 'Custom Analysis',
        'content': process_custom_data(custom_data)
    }
```

**Step 3:** Use in report generation

```python
pdf_path = generate(
    video_dir=Path("demo/demo_outputs/match"),
    sections=['executive', 'custom', 'tactical']
)
```

### Color Customization

```python
# templates/styles.css
:root {
    --team-a-color: #e74c3c;      /* Liverpool Red */
    --team-b-color: #3498db;      /* Man City Blue */
    --accent-color: #2c3e50;
    --text-color: #2c3e50;
    --background-color: #ecf0f1;
}
```

### Font Customization

```python
# config.yaml
report:
  fonts:
    body: "Arial"
    heading: "Arial Bold"
    code: "Courier New"
  sizes:
    body: 11
    heading: 16
    subheading: 14
```

---

## Output Examples

### Report Filename Convention

```
report_[team-a]_vs_[team-b]_[date]_[template-type].pdf

Examples:
- report_Liverpool_vs_ManCity_2026-05-21_standard.pdf
- report_Arsenal_vs_Chelsea_2026-05-20_executive.pdf
- report_Liverpool_vs_ManCity_2026-05-21_detailed.pdf
```

### File Size Reference

| Template | Pages | Size |
|----------|-------|------|
| Executive | 4-6 | 2-3 MB |
| Standard | 8-10 | 4-5 MB |
| Detailed | 15-20 | 8-10 MB |

---

## API Reference

### Main Function

```python
def generate(
    video_dir: Path,
    template: str = "standard",
    sections: List[str] = None,
    output_dir: Path = None,
    config: Dict = None
) -> Path:
    """
    Generate a PDF report from match metrics.
    
    Args:
        video_dir: Directory containing metrics.json
        template: Report template type (standard/executive/detailed)
        sections: List of sections to include
        output_dir: Output directory (default: video_dir)
        config: Custom configuration dict
    
    Returns:
        Path to generated PDF file
    
    Raises:
        FileNotFoundError: If metrics.json not found
        ValueError: If invalid template or sections
    """
```

### Section Types

```python
AVAILABLE_SECTIONS = [
    'executive',      # Executive Summary
    'tactical',       # Tactical Analysis
    'shooting',       # Shooting & xG
    'physical',       # Physical Metrics
    'highlights'      # Highlights
]
```

---

## Error Handling

### Common Issues

**Missing metrics.json**
```python
FileNotFoundError: metrics.json not found in video_dir
Solution: Run batch_processor first to generate metrics
```

**Invalid section**
```python
ValueError: Unknown section type 'invalid_section'
Solution: Use sections from AVAILABLE_SECTIONS list
```

**Missing dependencies**
```python
ImportError: reportlab not installed
Solution: pip install reportlab
```

---

## Performance Metrics

### Generation Time

- Executive Template: ~5-10 seconds
- Standard Template: ~15-20 seconds
- Detailed Template: ~30-45 seconds

### Memory Usage

- Executive: ~50-100 MB
- Standard: ~150-250 MB
- Detailed: ~300-500 MB

---

## Best Practices

1. **Always validate metrics.json** before report generation
2. **Choose appropriate template** based on use case
3. **Include relevant sections** to avoid cluttered reports
4. **Use custom branding** for team/organization identity
5. **Test reports** with sample matches before production use
6. **Archive generated reports** for record-keeping
7. **Monitor generation time** for performance optimization

---

## Future Enhancements

- [ ] Interactive PDF with embedded video clips
- [ ] Multi-match comparison reports
- [ ] Player performance tracking across seasons
- [ ] Automated report scheduling
- [ ] Email delivery of reports
- [ ] Report version control and history
- [ ] Real-time report generation during broadcasts
