"""
PDF Report Generator for post-match analysis.

Generates comprehensive 6-page PDF reports with match statistics, tactical diagrams,
shooting analytics, physical metrics, and highlights.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from io import BytesIO
from typing import Dict, Optional, Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle
import numpy as np
import yaml

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas


class PDFReportGenerator:
    """Generates professional PDF match reports."""

    def __init__(self, brand: Optional[Dict[str, Any]] = None):
        """
        Initialize PDF generator.

        Args:
            brand: Optional dict with branding colors/config
        """
        self.brand = brand or self._default_brand()
        self.temp_dir = None
        self.temp_files = []

    @staticmethod
    def _default_brand() -> Dict[str, Any]:
        """Load default brand colors from config.yaml."""
        try:
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
                return {
                    'primary': config.get('dashboard', {}).get('primary_color', '#e63946'),
                    'secondary': config.get('dashboard', {}).get('secondary_color', '#4361ee'),
                    'accent': config.get('dashboard', {}).get('accent_color', '#8b5cf6'),
                }
        except (FileNotFoundError, TypeError):
            return {
                'primary': '#e63946',
                'secondary': '#4361ee',
                'accent': '#8b5cf6',
            }

    def _hex_to_rgb(self, hex_color: str) -> tuple:
        """Convert hex color to RGB tuple (0-1 scale)."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    def _create_temp_image(self, fig) -> str:
        """Save matplotlib figure to temp file and return path."""
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()

        temp_file = Path(self.temp_dir) / f"chart_{len(self.temp_files)}.png"
        fig.savefig(str(temp_file), dpi=100, bbox_inches='tight')
        plt.close(fig)
        self.temp_files.append(str(temp_file))
        return str(temp_file)

    def _create_possession_chart(self, metrics: Dict) -> str:
        """Create possession percentage pie chart."""
        possession = metrics.get('possession', {})
        team_poss = possession.get('team_possession_percentage', {'A': 50, 'B': 50})

        fig, ax = plt.subplots(figsize=(6, 4))
        teams = ['Team A', 'Team B']
        values = [team_poss.get('A', 50), team_poss.get('B', 50)]
        colors_list = [self._hex_to_rgb(self.brand['primary']),
                      self._hex_to_rgb(self.brand['secondary'])]

        wedges, texts, autotexts = ax.pie(values, labels=teams, autopct='%1.1f%%',
                                           colors=colors_list, startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
            autotext.set_weight('bold')

        ax.set_title('Possession %', fontsize=14, weight='bold')
        return self._create_temp_image(fig)

    def _create_shot_map(self, metrics: Dict) -> str:
        """Create shot map on pitch."""
        shots = metrics.get('shot_events', [])

        fig, ax = plt.subplots(figsize=(10, 6))

        # Draw pitch
        pitch_length, pitch_width = 1920, 1080
        ax.set_xlim(0, pitch_length)
        ax.set_ylim(0, pitch_width)

        # Pitch background
        ax.add_patch(Rectangle((0, 0), pitch_length, pitch_width,
                              linewidth=2, edgecolor='white', facecolor='#2d5016'))

        # Halfway line
        ax.plot([pitch_length/2, pitch_length/2], [0, pitch_width], 'w-', linewidth=2)

        # Center circle
        circle = Circle((pitch_length/2, pitch_width/2), 100, fill=False,
                       edgecolor='white', linewidth=1)
        ax.add_patch(circle)

        # Goal areas
        goal_height = pitch_width * 0.3
        ax.add_patch(Rectangle((0, (pitch_width-goal_height)/2), 100, goal_height,
                              linewidth=1, edgecolor='white', fill=False))
        ax.add_patch(Rectangle((pitch_length-100, (pitch_width-goal_height)/2), 100, goal_height,
                              linewidth=1, edgecolor='white', fill=False))

        # Plot shots
        for shot in shots:
            if shot.get('shooter_team') == 'A':
                color = self.brand['primary']
                marker = 'o'
            else:
                color = self.brand['secondary']
                marker = 's'

            x, y = shot.get('ball_position', [pitch_length/2, pitch_width/2])
            ax.scatter(x, y, s=200, c=color, marker=marker, edgecolors='white', linewidth=1.5)

        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Shot Map', fontsize=14, weight='bold', color='white', pad=20)
        fig.patch.set_facecolor('#2d5016')

        return self._create_temp_image(fig)

    def _create_passes_chart(self, metrics: Dict) -> str:
        """Create pass statistics chart."""
        pass_stats = metrics.get('pass_detection', {})

        fig, ax = plt.subplots(figsize=(8, 5))

        categories = ['Forward', 'Backward', 'Lateral']
        direction = pass_stats.get('direction', {})
        values = [
            direction.get('forward', 0),
            direction.get('backward', 0),
            direction.get('lateral', 0)
        ]

        if sum(values) == 0:
            ax.text(0.5, 0.5, 'No pass data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.axis('off')
            return self._create_temp_image(fig)

        bars = ax.bar(categories, values, color=[self.brand['primary'],
                                                   self.brand['secondary'],
                                                   self.brand['accent']])
        ax.set_ylabel('Number of Passes', fontsize=11)
        ax.set_title('Pass Direction Distribution', fontsize=12, weight='bold')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10)

        return self._create_temp_image(fig)

    def _create_distance_chart(self, metrics: Dict) -> str:
        """Create distance covered bar chart."""
        tracks = metrics.get('tracks', [])

        fig, ax = plt.subplots(figsize=(10, 5))

        # Get top 10 by distance
        sorted_players = sorted(tracks,
                               key=lambda x: x.get('total_distance_m', 0),
                               reverse=True)[:10]

        if not sorted_players:
            ax.text(0.5, 0.5, 'No player data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.axis('off')
            return self._create_temp_image(fig)

        names = [p.get('display_name', 'P' + str(p.get('player_id')))
                for p in sorted_players]
        distances = [p.get('total_distance_m', 0) for p in sorted_players]

        colors_list = [self.brand['primary'] if p.get('team') == 'A'
                      else self.brand['secondary'] for p in sorted_players]

        bars = ax.barh(names, distances, color=colors_list)
        ax.set_xlabel('Distance (meters)', fontsize=11)
        ax.set_title('Top 10 Players by Distance Covered', fontsize=12, weight='bold')
        ax.invert_yaxis()

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{width:.0f}m', ha='left', va='center', fontsize=9)

        return self._create_temp_image(fig)

    def generate(self, video_dir: Path, output_pdf: Path | None = None,
                brand: Dict | None = None) -> Path:
        """
        Generate comprehensive PDF report.

        Args:
            video_dir: Directory containing metrics.json
            output_pdf: Output PDF path (auto-generated if None)
            brand: Optional brand config

        Returns:
            Path to generated PDF
        """
        if brand:
            self.brand = brand

        video_dir = Path(video_dir)
        metrics_path = video_dir / 'metrics.json'

        if not metrics_path.exists():
            raise FileNotFoundError(f"metrics.json not found in {video_dir}")

        with open(metrics_path) as f:
            metrics = json.load(f)

        if output_pdf is None:
            output_pdf = video_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        output_pdf = Path(output_pdf)
        output_pdf.parent.mkdir(parents=True, exist_ok=True)

        # Create PDF
        doc = SimpleDocTemplate(str(output_pdf), pagesize=letter,
                              rightMargin=0.5*inch, leftMargin=0.5*inch,
                              topMargin=0.5*inch, bottomMargin=0.5*inch)

        story = []
        styles = getSampleStyleSheet()

        # Page 1: Cover
        story.extend(self._create_cover_page(metrics, styles))
        story.append(PageBreak())

        # Page 2: Executive Summary
        story.extend(self._create_executive_summary(metrics, styles))
        story.append(PageBreak())

        # Page 3: Tactical
        story.extend(self._create_tactical_page(metrics, styles))
        story.append(PageBreak())

        # Page 4: Shooting & xG
        story.extend(self._create_shooting_page(metrics, styles))
        story.append(PageBreak())

        # Page 5: Physical
        story.extend(self._create_physical_page(metrics, styles))
        story.append(PageBreak())

        # Page 6: Highlights
        story.extend(self._create_highlights_page(metrics, styles))

        doc.build(story)

        # Cleanup temp files
        self._cleanup_temp_files()

        return output_pdf

    def _create_cover_page(self, metrics: Dict, styles) -> list:
        """Create cover page."""
        elements = []

        elements.append(Spacer(1, 1.5*inch))

        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=28,
            textColor=colors.HexColor(self.brand['primary']),
            spaceAfter=30,
            alignment=TA_CENTER,
            weight='bold'
        )

        elements.append(Paragraph('MATCH REPORT', title_style))
        elements.append(Spacer(1, 0.3*inch))

        team_a = metrics.get('team_names', {}).get('A', 'Team A')
        team_b = metrics.get('team_names', {}).get('B', 'Team B')

        match_title = f"{team_a} vs {team_b}"
        elements.append(Paragraph(match_title, styles['Heading2']))
        elements.append(Spacer(1, 0.3*inch))

        date_str = datetime.now().strftime('%B %d, %Y')
        elements.append(Paragraph(f"Date: {date_str}", styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))

        time_str = datetime.now().strftime('%H:%M:%S')
        elements.append(Paragraph(f"Generated: {time_str}", styles['Normal']))

        return elements

    def _create_executive_summary(self, metrics: Dict, styles) -> list:
        """Create executive summary page."""
        elements = []

        elements.append(Paragraph('EXECUTIVE SUMMARY', styles['Heading1']))
        elements.append(Spacer(1, 0.2*inch))

        possession = metrics.get('possession', {})
        team_poss = possession.get('team_possession_percentage', {'A': 50, 'B': 50})

        pass_stats = metrics.get('pass_detection', {})
        shot_stats = metrics.get('shot_detection', {})
        sprint_stats = metrics.get('sprint_detection', {})

        summary_data = [
            ['Metric', 'Team A', 'Team B'],
            ['Possession %', f"{team_poss.get('A', 0):.1f}%", f"{team_poss.get('B', 0):.1f}%"],
            ['Total Passes', str(pass_stats.get('total_passes', 0)), 'N/A'],
            ['Pass Accuracy', f"{pass_stats.get('pass_accuracy', 0):.1f}%", 'N/A'],
            ['Total Shots', str(shot_stats.get('team_shots', {}).get('A', 0)),
             str(shot_stats.get('team_shots', {}).get('B', 0))],
            ['Total Sprints', str(sprint_stats.get('team_sprints', {}).get('A', 0)),
             str(sprint_stats.get('team_sprints', {}).get('B', 0))],
        ]

        table = Table(summary_data, colWidths=[2*inch, 2*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.brand['primary'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 0.3*inch))

        # Top sprinter
        top_sprinters = sorted(metrics.get('tracks', []),
                              key=lambda x: x.get('max_speed_mps', 0),
                              reverse=True)[:1]

        if top_sprinters:
            sprinter = top_sprinters[0]
            sprinter_name = sprinter.get('display_name', 'Player')
            sprinter_speed = sprinter.get('max_speed_mps', 0)
            sprinter_dist = sprinter.get('total_distance_m', 0)
            elements.append(Paragraph(
                f"<b>Top Sprinter:</b> {sprinter_name} - {sprinter_speed:.1f} m/s, {sprinter_dist:.0f}m distance",
                styles['Normal']
            ))

        return elements

    def _create_tactical_page(self, metrics: Dict, styles) -> list:
        """Create tactical analysis page."""
        elements = []

        elements.append(Paragraph('TACTICAL ANALYSIS', styles['Heading1']))
        elements.append(Spacer(1, 0.2*inch))

        elements.append(Paragraph('Formation Diagram', styles['Heading2']))
        elements.append(Spacer(1, 0.1*inch))

        # Create simple formation visualization
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect('equal')

        # Draw pitch
        ax.add_patch(Rectangle((0, 0), 100, 100, linewidth=2, edgecolor='white', facecolor='#2d5016'))
        ax.plot([50, 50], [0, 100], 'w-', linewidth=1)

        # Sample formation positions (4-3-3)
        team_a_positions = [(10, 50), (25, 25), (25, 50), (25, 75),
                            (50, 30), (50, 50), (50, 70),
                            (75, 20), (75, 50), (75, 80)]

        for pos in team_a_positions:
            circle = Circle(pos, 3, color=self.brand['primary'], alpha=0.7)
            ax.add_patch(circle)

        ax.set_title('Formation: 4-3-3', fontsize=12, weight='bold', color='white', pad=20)
        ax.axis('off')
        fig.patch.set_facecolor('#2d5016')

        chart_path = self._create_temp_image(fig)
        elements.append(Image(chart_path, width=5*inch, height=3*inch))

        return elements

    def _create_shooting_page(self, metrics: Dict, styles) -> list:
        """Create shooting and xG page."""
        elements = []

        elements.append(Paragraph('SHOOTING & EXPECTED GOALS', styles['Heading1']))
        elements.append(Spacer(1, 0.2*inch))

        shot_stats = metrics.get('shot_detection', {})

        # Add shot map
        shot_map_path = self._create_shot_map(metrics)
        elements.append(Image(shot_map_path, width=5*inch, height=3*inch))
        elements.append(Spacer(1, 0.1*inch))

        # Shot statistics
        elements.append(Paragraph('Shot Summary', styles['Heading3']))

        shot_data = [
            ['Metric', 'Value'],
            ['Total Shots', str(shot_stats.get('total_shots', 0))],
            ['Team A Shots', str(shot_stats.get('team_shots', {}).get('A', 0))],
            ['Team B Shots', str(shot_stats.get('team_shots', {}).get('B', 0))],
        ]

        table = Table(shot_data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.brand['primary'])),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(table)

        return elements

    def _create_physical_page(self, metrics: Dict, styles) -> list:
        """Create physical metrics page."""
        elements = []

        elements.append(Paragraph('PHYSICAL METRICS', styles['Heading1']))
        elements.append(Spacer(1, 0.2*inch))

        # Distance chart
        elements.append(Paragraph('Distance Covered', styles['Heading2']))
        distance_path = self._create_distance_chart(metrics)
        elements.append(Image(distance_path, width=5.5*inch, height=3*inch))

        return elements

    def _create_highlights_page(self, metrics: Dict, styles) -> list:
        """Create highlights page."""
        elements = []

        elements.append(Paragraph('DETECTED EVENTS & HIGHLIGHTS', styles['Heading1']))
        elements.append(Spacer(1, 0.2*inch))

        # Collect all events
        events = []

        for shot in metrics.get('shot_events', []):
            events.append({
                'time': shot.get('timestamp', 0),
                'type': 'Shot',
                'team': shot.get('shooter_team', 'N/A'),
                'velocity': shot.get('velocity_mps', 0)
            })

        for sprint in metrics.get('sprint_events', []):
            events.append({
                'time': sprint.get('start_time', 0),
                'type': 'Sprint',
                'team': sprint.get('team', 'N/A'),
                'speed': sprint.get('max_speed_mps', 0)
            })

        # Sort by time
        events.sort(key=lambda x: x['time'])

        if events:
            event_data = [['Time (s)', 'Event Type', 'Team', 'Details']]
            for event in events[:20]:  # Limit to 20 events
                details = f"{event.get('velocity', event.get('speed', 0)):.1f} m/s"
                event_data.append([
                    f"{event['time']:.1f}",
                    event['type'],
                    event['team'],
                    details
                ])

            table = Table(event_data, colWidths=[1*inch, 1.5*inch, 1*inch, 2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.brand['primary'])),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            elements.append(table)
        else:
            elements.append(Paragraph('No events detected', styles['Normal']))

        return elements

    def _cleanup_temp_files(self):
        """Clean up temporary image files."""
        import shutil
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
            self.temp_files = []


def generate(video_dir: Path, output_pdf: Path | None = None,
            brand: Dict | None = None) -> Path:
    """
    Public API to generate PDF report.

    Args:
        video_dir: Directory containing metrics.json
        output_pdf: Output PDF path (optional)
        brand: Brand colors config (optional)

    Returns:
        Path to generated PDF
    """
    generator = PDFReportGenerator(brand)
    return generator.generate(video_dir, output_pdf, brand)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m services.pdf_report_generator --video-dir PATH [--output PATH]")
        sys.exit(1)

    video_dir = None
    output_pdf = None

    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--video-dir' and i + 1 < len(sys.argv) - 1:
            video_dir = Path(sys.argv[i + 2])
        elif arg == '--output' and i + 1 < len(sys.argv) - 1:
            output_pdf = Path(sys.argv[i + 2])

    if not video_dir:
        print("Error: --video-dir required")
        sys.exit(1)

    pdf_path = generate(video_dir, output_pdf)
    print(f"PDF generated: {pdf_path}")
