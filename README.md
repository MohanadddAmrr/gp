# TactiVision Pro - Football Analytics Platform

<p align="center">
  <img src="https://via.placeholder.com/200x200/1a1a24/e63946?text=TV" alt="TactiVision Pro Logo" width="200"/>
</p>

<p align="center">
  <strong>Professional Football Analytics Platform</strong><br>
  Real-time match analysis, player tracking, and tactical insights powered by AI
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#api-documentation">API</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#troubleshooting">Troubleshooting</a>
</p>

---

## 🎯 Overview

TactiVision Pro is a comprehensive football analytics platform that combines computer vision, machine learning, and advanced statistical analysis to provide deep insights into football matches. The platform processes video footage to track players, analyze tactics, calculate expected goals (xG), and generate detailed match reports.

### Key Capabilities

- **Real-time Player Tracking**: Track all 22 players plus the ball using state-of-the-art object detection
- **Tactical Analysis**: Formation detection, pressing intensity, transition tracking
- **Expected Goals (xG)**: Advanced shot quality analysis and goal probability
- **Physical Performance**: Distance covered, sprints, speed analysis
- **Video Highlights**: Automatic generation of key moments and highlight reels
- **Database Integration**: Store and query match data, player profiles, and statistics
- **Broadcast Graphics**: Professional overlays for live streaming
- **Social Media Export**: Generate clips optimized for various platforms

---

## ✨ Features

### 📊 Analytics Dashboard
- **10 Interactive Tabs**: Overview, Shooting, Passing, Physical, Tactical, xG & Analytics, Heatmaps, Highlights, Database, Settings
- **Real-time Updates**: Live data refresh during match processing
- **Export Options**: PDF reports, Excel, JSON, CSV, and video exports
- **Player Comparison**: Side-by-side player statistics and radar charts

### 🎯 Shooting Analysis
- Shot map visualization on tactical pitch
- Shot velocity tracking
- Body part and shot type classification
- Top shooters leaderboard
- xG integration for shot quality

### 🔄 Passing Analysis
- Pass direction distribution (radar chart)
- Pass distance statistics
- Top passers leaderboard
- Passing network visualization
- Pass completion rates by zone

### 💪 Physical Performance
- Distance covered (total and by zone)
- Sprint detection and analysis
- Speed metrics (average, max, acceleration)
- Workload scores
- Team comparison charts

### 🗺️ Tactical Analysis
- Formation detection and tracking
- Offside detection and analysis
- Set piece recognition (corners, free kicks, throw-ins)
- Dribble and take-on detection
- Pressing intensity metrics
- Transition tracking (defense to attack)
- Tactical pitch visualization

### 📈 Expected Goals (xG)
- Shot quality calculation based on position and angle
- Body part tracking
- Shot type classification
- xG timeline visualization
- Team and player xG comparison
- Conversion rate analysis

### 🔥 Heatmaps
- Team heatmaps (attacking/defensive activity)
- Individual player heatmaps
- Ball movement heatmaps
- Zone control analysis

### ⚡ Highlights
- Automatic key moment detection
- Event timeline visualization
- Highlight clip generation
- Importance scoring
- Video export with timestamps

### 🗄️ Database
- SQLite database for all match data
- Player profile management with face recognition
- Cross-match player tracking
- Historical statistics
- Data export and backup

### 📺 Broadcast & Streaming
- Real-time scoreboard overlays
- RTMP streaming support
- WebRTC integration
- Custom graphics templates

### 📱 Social Media
- Platform-specific exports (Twitter, Instagram, TikTok, YouTube)
- Auto-generated captions
- Hashtag suggestions
- Optimized video formats

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for real-time processing)
- 8GB+ RAM (16GB recommended)
- 10GB free disk space

### Quick Install

```bash
# Clone the repository
git clone https://github.com/yourusername/tactivision-pro.git
cd tactivision-pro

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download YOLO model (automatic on first run)
# Or manually download from https://github.com/ultralytics/assets/releases
```

### Dependencies

The platform requires the following main dependencies:

**Core:**
- `ultralytics` - YOLO object detection
- `opencv-python` - Video processing
- `numpy` - Numerical computing
- `pandas` - Data manipulation

**Machine Learning:**
- `scikit-learn` - ML algorithms
- `torch` - Deep learning framework
- `tensorflow` (optional) - Additional ML models

**Visualization:**
- `streamlit` - Interactive dashboard
- `plotly` - Interactive charts
- `matplotlib` - Static visualizations
- `pillow` - Image processing

**Database:**
- `sqlalchemy` - Database ORM
- `sqlite3` - Local database (built-in)

**API & Export:**
- `requests` - HTTP requests
- `pyyaml` - Configuration files
- `python-dotenv` - Environment variables

See [`requirements.txt`](requirements.txt) for complete list.

---

## 📖 Usage

### Quick Start

```bash
# Launch the main application
python main.py

# Or launch with specific options
python main.py --mode dashboard --video input.mp4
```

### Processing a Match Video

```bash
# Using the main entry point
python main.py --mode process --video path/to/match.mp4 --roster rosters/team.json

# Or using the demo runner
python demo/run_demo.py --video path/to/match.mp4
```

### Launching the Dashboard

```bash
# Launch the comprehensive dashboard
streamlit run demo/dashboard_final.py

# Or use the main launcher
python main.py --mode dashboard
```

### Command-Line Interface

```bash
# Show all available options
python main.py --help

# Process video with custom settings
python main.py \
  --mode process \
  --video match.mp4 \
  --roster team.json \
  --output ./outputs \
  --enable-xg \
  --enable-highlights

# Export data
python main.py \
  --mode export \
  --match-id 1 \
  --format pdf

# Run tests
python main.py --mode test
```

### Using the Dashboard

1. **Select a Match**: Choose from processed videos in the sidebar
2. **Navigate Tabs**: Use the 10 tabs to explore different analytics
3. **Export Data**: Use export buttons to save reports
4. **Compare Players**: Use the player comparison tool in the Database tab
5. **Configure Settings**: Customize appearance and notifications in Settings

---

## 🔌 API Documentation

### Database Manager API

```python
from services.database_manager import DatabaseManager

# Initialize
db = DatabaseManager("matches.db")
db.initialize_database()

# Create a match
match_id = db.create_match(
    video_path="path/to/video.mp4",
    team_a="Liverpool",
    team_b="Manchester City",
    duration_seconds=5400
)

# Get match data
match = db.get_match(match_id)

# Get player stats
players = db.get_players_for_match(match_id)

# Get events
events = db.get_events_for_match(match_id, event_type="goal")

# Export data
db.export_match_to_json(match_id, "output.json")
```

### Tactical Analyzer API

```python
from services.tactical_analyzer import TacticalAnalyzer

# Initialize
analyzer = TacticalAnalyzer(
    frame_width=1920,
    frame_height=1080,
    enable_formation=True,
    enable_offside=True
)

# Process frame
report = analyzer.analyze_frame(
    player_positions=positions,
    ball_position=ball_pos,
    timestamp=frame_time
)

# Get comprehensive report
full_report = analyzer.get_full_report()
```

### xG Calculator API

```python
from services.xg_calculator import xGCalculator, ShotEvent

# Initialize
calculator = xGCalculator()

# Record a shot
event = ShotEvent(
    timestamp=120.5,
    frame=3012,
    shooter_id=10,
    shooter_team="A",
    x=0.85,  # Normalized position
    y=0.45,
    shot_type=ShotType.OPEN_PLAY
)

# Calculate xG
xg = calculator.calculate_xg(event)
event.xg_value = xg

# Get statistics
stats = calculator.get_stats()
```

### Highlights Generator API

```python
from services.highlights_generator import HighlightsGenerator

# Initialize
generator = HighlightsGenerator()

# Add events
generator.add_event(
    timestamp=120.5,
    event_type="goal",
    importance="CRITICAL",
    description="Brilliant strike from outside the box"
)

# Generate highlights
highlights = generator.generate_highlights(
    video_path="match.mp4",
    output_dir="highlights/"
)
```

### Player Profile System API

```python
from services.player_profile_system import PlayerProfileSystem

# Initialize
profile_system = PlayerProfileSystem(db_manager)

# Create profile
profile_id = profile_system.create_profile(
    name="Mohamed Salah",
    team_name="Liverpool",
    jersey_number=11,
    position="RW"
)

# Get career stats
career = profile_system.get_career_stats(profile_id)

# Calculate form
form = profile_system.calculate_form(profile_id)
```

---

## ⚙️ Configuration

### Configuration File (config.yaml)

```yaml
# TactiVision Pro Configuration

# Application Settings
app:
  name: "TactiVision Pro"
  version: "2.0.0"
  debug: false
  log_level: "INFO"

# Video Processing
video:
  default_fps: 25
  default_resolution: [1920, 1080]
  processing_threads: 4
  batch_size: 32
  
  # Object Detection
  detection:
    model: "yolov8n.pt"
    confidence_threshold: 0.5
    iou_threshold: 0.45
    classes: [0]  # Person class
    
  # Tracking
  tracking:
    max_age: 30
    min_hits: 3
    iou_threshold: 0.3

# Analytics
analytics:
  # xG Calculation
  xg:
    model: "default"
    shot_distance_max: 40  # meters
    
  # Tactical Analysis
  tactical:
    formation_detection_interval: 5  # seconds
    pressing_threshold: 2.5  # meters per second
    
  # Physical Metrics
  physical:
    sprint_threshold: 5.5  # m/s
    high_intensity_threshold: 7.0  # m/s
    distance_smoothing: 0.8

# Database
database:
  path: "matches.db"
  backup_interval: 3600  # seconds
  max_connections: 10

# Dashboard
dashboard:
  port: 8501
  theme: "dark"
  auto_refresh: false
  refresh_interval: 30
  
# Export Settings
export:
  default_format: "mp4"
  video_quality: "high"
  include_overlays: true
  
# API Keys (use environment variables in production)
api:
  statsbomb:
    enabled: true
  football_data:
    enabled: false
    api_key: ""  # Set via FOOTBALL_DATA_API_KEY env var
  
# Team Colors
team_colors:
  home: "#e63946"
  away: "#4361ee"
  ball: "#ffffff"
  pitch: "#0d5c28"
```

### Environment Variables

Create a `.env` file:

```bash
# API Keys
FOOTBALL_DATA_API_KEY=your_api_key_here
OPTA_API_KEY=your_api_key_here

# Database
DATABASE_URL=sqlite:///matches.db

# Paths
INPUT_VIDEOS_PATH=./input_videos
OUTPUT_PATH=./outputs
MODELS_PATH=./models

# Processing
CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=4
```

### Customizing Team Colors

Edit `config.yaml` or use the Settings tab in the dashboard:

```yaml
team_colors:
  Liverpool: "#c8102e"
  "Manchester City": "#6cabdd"
  "Manchester United": "#da291c"
  Chelsea: "#034694"
  Arsenal: "#ef0107"
  Tottenham: "#132257"
```

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_ball_tracker.py

# Run with coverage
python -m pytest tests/ --cov=services --cov-report=html

# Run performance benchmarks
python tests/test_suite.py --benchmark
```

### Test Structure

```
tests/
├── test_ball_tracker.py       # Ball tracking tests
├── test_possession.py         # Possession calculation tests
├── test_possession_validation.py  # Validation tests
├── test_suite.py              # Comprehensive test suite
└── conftest.py               # Pytest configuration
```

### Writing Tests

```python
# tests/test_example.py
import pytest
from services.ball_tracker import BallTracker

def test_ball_tracker_initialization():
    tracker = BallTracker()
    assert tracker is not None
    assert tracker.history.maxlen == 30

def test_ball_position_update():
    tracker = BallTracker()
    tracker.update((100, 200), 0.9)
    assert len(tracker.history) == 1
    assert tracker.get_smoothed_position() == (100, 200)
```

---

## 🐛 Troubleshooting

### Common Issues

#### Issue: "No module named 'ultralytics'"

**Solution:**
```bash
pip install ultralytics
# Or reinstall all dependencies
pip install -r requirements.txt
```

#### Issue: "CUDA out of memory"

**Solution:**
```python
# Reduce batch size in config.yaml
video:
  batch_size: 16  # Reduce from 32
  
# Or use CPU processing
export CUDA_VISIBLE_DEVICES=""
```

#### Issue: "Database is locked"

**Solution:**
```bash
# Close other applications using the database
# Or reset the database
python -c "from services.database_manager import DatabaseManager; db = DatabaseManager(); db.reset_database()"
```

#### Issue: "Video processing is slow"

**Solutions:**
1. Enable GPU acceleration
2. Reduce video resolution
3. Adjust detection interval
4. Use a smaller YOLO model (yolov8n instead of yolov8x)

```yaml
# config.yaml
video:
  detection:
    model: "yolov8n.pt"  # Fastest model
  processing_interval: 2  # Process every 2nd frame
```

#### Issue: "Streamlit dashboard won't load"

**Solutions:**
```bash
# Check if port is available
lsof -i :8501

# Use different port
streamlit run demo/dashboard_final.py --server.port 8502

# Clear cache
streamlit cache clear
```

### Performance Optimization

1. **Use GPU**: Ensure CUDA is properly installed
2. **Batch Processing**: Increase batch size if GPU memory allows
3. **Frame Skipping**: Process every Nth frame for analysis
4. **Database Indexing**: Enable indexes for faster queries
5. **Caching**: Use Streamlit's caching decorators

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or set in config:

```yaml
app:
  debug: true
  log_level: "DEBUG"
```

---

## 📚 Additional Resources

### Documentation

- [API Reference](docs/api.md)
- [Architecture Overview](docs/architecture.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

### External Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [StatsBomb Open Data](https://github.com/statsbomb/open-data)

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/tactivision-pro.git

# Install dev dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Create a branch
git checkout -b feature/my-feature

# Make changes and commit
git commit -m "Add my feature"

# Push and create PR
git push origin feature/my-feature
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLO models
- [StatsBomb](https://statsbomb.com/) for football analytics inspiration
- [Streamlit](https://streamlit.io/) for the amazing dashboard framework
- All contributors and supporters of the project

---

## 📞 Support

- **Documentation**: [https://tactivision.pro/docs](https://tactivision.pro/docs)
- **Issues**: [GitHub Issues](https://github.com/yourusername/tactivision-pro/issues)
- **Email**: support@tactivision.pro
- **Discord**: [Join our community](https://discord.gg/tactivision)

---

<p align="center">
  Made with ❤️ for the beautiful game
</p>
