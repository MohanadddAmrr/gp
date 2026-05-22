# API Integration Guide
**Owner:** Ahmed Khaled (Member E)

## Overview
TactiVision Pro connects to [football-data.org](https://www.football-data.org) to pull live fixtures, standings, and squad data directly into the dashboard and database.

## Setup

### 1. Get a Free API Key
Register at https://www.football-data.org/client/register — it is free for the competitions we use.

### 2. Set the Environment Variable
Create a `.env` file in the project root:
```
FOOTBALL_DATA_API_KEY=your_key_here
```

### 3. Load It in Your Code
```python
import os
from services.api_connector import APIManager

api = APIManager()
api.add_football_data(os.environ["FOOTBALL_DATA_API_KEY"])
```

## Key Classes

### `FootballDataConnector` (`services/api_connector.py`)
The main connector. Uses the football-data.org **v4 API**.

| Method | Description |
|---|---|
| `get_competitions()` | List all available competitions |
| `get_matches(competition_id, date_from, date_to)` | Fetch matches for a competition |
| `get_standings(competition_id)` | Get league table |
| `get_team_matches(team_id)` | All matches for a specific team |
| `fetch_squad(team_name_or_id)` | Squad roster with jersey numbers |
| `fetch_and_save_squad(team_name, db)` | Fetch squad and persist to DB |

### `MatchDataSync` (`services/match_data_sync.py`)
High-level sync service that orchestrates connector calls and DB writes.

```python
from services.match_data_sync import MatchDataSync
from services.database_manager import DatabaseManager

sync = MatchDataSync(api, DatabaseManager())
sync.sync_competitions()
sync.sync_matches("2021")           # Premier League
sync.sync_team_squad("Arsenal FC")
standings = sync.get_standings("2021")
```

## Competition IDs (football-data.org)

| Competition | ID |
|---|---|
| Premier League | 2021 |
| Bundesliga | 2002 |
| La Liga | 2014 |
| Serie A | 2019 |
| Ligue 1 | 2015 |
| UEFA Champions League | 2001 |

## Rate Limiting
The free tier allows **10 requests per minute**. The connector enforces this automatically via `_rate_limit()`. Results are cached in memory (1 hour TTL) and on disk (`cache/api_responses/`) to minimize API calls.

## Live Dashboard Tab
The Live Match Data tab is in `demo/dashboard_pages/live_match_data.py`.
It shows standings, recent results, and squad lookup for 6 competitions.

To wire it into the main dashboard:
```python
from demo.dashboard_pages import live_match_data
# inside a st.tab() block:
live_match_data.render()
```

## Running the Tests
```bash
pytest tests/test_api_connector.py tests/test_match_data_sync.py -v
```
