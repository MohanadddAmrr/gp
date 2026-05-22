# Data Enrichment Guide
**Owner:** Ahmed Khaled (Member E)

## Overview
The Data Enrichment layer merges **external API data** (from football-data.org) with **local video-analysis data** (metrics.json from `demo/demo_outputs/`) into a single enriched match record used by the dashboard and PDF report generator.

## How It Works

```
football-data.org API          Local Video Analysis
        │                              │
        ▼                              ▼
  MatchDataSync              demo_outputs/<match>/
  (live fixtures,              metrics.json
   standings, squads)          (possession, xG,
        │                       tracking stats)
        └──────────┬────────────┘
                   ▼
            DataEnrichment
          (merged enriched dict)
                   │
                   ▼
            matches.db + Dashboard
```

## Usage

### Enrich a Single Match
```python
from services.api_connector import APIManager
from services.database_manager import DatabaseManager
from services.match_data_sync import MatchDataSync
from services.data_enrichment import DataEnrichment
import os

api = APIManager()
api.add_football_data(os.environ["FOOTBALL_DATA_API_KEY"])
db   = DatabaseManager()
sync = MatchDataSync(api, db)

enricher = DataEnrichment(sync, db)
result = enricher.enrich_match(
    home_team="Arsenal FC",
    away_team="Manchester City FC",
    competition_id="2021",
    metrics_path="demo/demo_outputs/arsenal_vs_city/metrics.json"
)
print(enricher.get_enriched_summary(result))
# → Arsenal FC 2 – 1 Manchester City FC (Premier League)
```

### Enrich All Matches in demo_outputs
```python
results = enricher.enrich_all_from_demo_outputs(
    demo_outputs_dir="demo/demo_outputs",
    competition_id="2021"
)
print(f"Enriched {len(results)} matches")
```

## Enriched Record Format

```json
{
  "api_data":    { ... raw API match dict ... },
  "video_data":  { ... raw metrics.json content ... },
  "enriched": {
    "home_team":        "Arsenal FC",
    "away_team":        "Manchester City FC",
    "competition":      "Premier League",
    "match_date":       "2024-09-15T14:00:00",
    "home_score":       2,
    "away_score":       1,
    "possession_home":  55,
    "possession_away":  45,
    "shots_home":       8,
    "shots_away":       5,
    "xg_home":          1.8,
    "xg_away":          0.9,
    "tracking_mota":    0.72,
    "tracking_idf1":    0.68,
    "id_switches":      12
  },
  "enriched_at": "2026-05-15T10:00:00"
}
```

## Video Downloader (`services/video_downloader.py`)

Downloads video clips safely — only use on clips you have rights to.

```python
from services.video_downloader import download

path = download(
    url="https://example.com/clip.mp4",
    dest_dir="input_videos/",
    max_size_mb=500
)
```

### Features
- **Chunked streaming** — downloads in 1 MB pieces, no RAM overflow
- **Resume support** — sends `Range` header if a partial file exists
- **Size cap** — rejects files over `max_size_mb`
- **Extension validation** — only `.mp4`, `.mkv`, `.avi`, `.mov` allowed

## Running the Tests
```bash
pytest tests/test_data_enrichment.py -v
```
