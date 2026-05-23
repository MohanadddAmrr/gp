# Demo Script Documentation

## Overview

The TactiVision Demo Script system enables orchestrated, reproducible demonstrations of the platform without requiring live commands or improvisation. This document details how to create, execute, and manage demo scenarios.

## Quick Start

### Running the Prototype Review Demo

```bash
# Navigate to project directory
cd /path/to/gpp

# Run the demo scenario
python -m services.scenario_player demo/scenarios/prototype_review.yaml

# Or via Python code
from services.demo_runner import run_demo_scenario
from pathlib import Path

run_demo_scenario(Path("demo/scenarios/prototype_review.yaml"))
```

---

## Demo Scenario Structure

### YAML Format

Each demo scenario is a YAML file with the following structure:

```yaml
name: scenario_name
description: Brief description of what the scenario demonstrates

steps:
  - kind: step_type
    # Step-specific parameters
    param1: value1
    param2: value2
    note: "Optional note for operator"
```

### Minimal Example

```yaml
name: quick_demo
description: Quick 5-minute demo

steps:
  - kind: pause
    note: "Welcome to TactiVision!"
  
  - kind: open_dashboard
    tab: model_comparison
    note: "Check out our model comparison"
```

---

## Step Types

### 1. `ensure_processed`

**Purpose:** Verify video is processed; skip if metrics exist

**Parameters:**
```yaml
kind: ensure_processed
video: path/to/video.mp4
note: "Optional message"
```

**Behavior:**
- Checks for `metrics.json` in video directory
- If exists: Logs message and continues
- If missing: Runs batch processor to generate metrics
- Waits for processing to complete before next step

**Example:**
```yaml
- kind: ensure_processed
  video: demo/input_videos/liverpool_vs_city.mp4
  note: "Processing Liverpool vs Man City match"
```

**Output:**
```
Metrics already exist: demo/demo_outputs/liverpool_vs_city/metrics.json
Skipping processing.
```

---

### 2. `open_dashboard`

**Purpose:** Instruct operator to navigate dashboard to specific tab

**Parameters:**
```yaml
kind: open_dashboard
tab: tab_name
note: "Optional instructions"
```

**Available Tabs:**
```
- Overview
- Shooting
- Passing
- Physical
- Tactical
- xG & Analytics
- Heatmaps
- Highlights
- Database
- Settings
- AI Recommendations
- Player Performance
- Match Comparison
- Generate Report
```

**Behavior:**
- Prints dashboard URL
- Prints tab name to navigate to
- Prints optional note with context
- Waits for operator to press Enter
- Continues to next step

**Example:**
```yaml
- kind: open_dashboard
  tab: model_comparison
  note: "Walk through the 5-model performance comparison. Compare accuracy, speed, and detection quality."

- kind: open_dashboard
  tab: accuracy_report
  note: "Review precision/recall metrics for key detections"

- kind: open_dashboard
  tab: highlights
  note: "Check the extracted highlights - notice the key moments"
```

**Output:**
```
Open dashboard in your browser:
URL: http://localhost:8501
Tab: model_comparison
Note: Walk through the 5-model performance comparison...

Press Enter when ready...
```

---

### 3. `generate_pdf`

**Purpose:** Generate PDF report from processed match metrics

**Parameters:**
```yaml
kind: generate_pdf
video_dir: path/to/match/directory
note: "Optional message"
```

**Behavior:**
- Checks for `metrics.json` in directory
- Invokes PDF report generator
- Saves PDF to output directory
- Reports success/failure
- Continues to next step

**Example:**
```yaml
- kind: generate_pdf
  video_dir: demo/demo_outputs/liverpool_vs_city
  note: "Generate comprehensive PDF report with all sections"
```

**Output:**
```
Generating PDF report...
Input directory: demo/demo_outputs/liverpool_vs_city
PDF generated: report_Liverpool_vs_ManCity_2026-05-21_standard.pdf
Location: demo/demo_outputs/liverpool_vs_city/report_Liverpool_vs_ManCity_2026-05-21_standard.pdf
```

---

### 4. `pause`

**Purpose:** Pause demo for operator interaction, Q&A, or manual inspection

**Parameters:**
```yaml
kind: pause
seconds: 0          # Optional: auto-continue after N seconds (0 = manual)
note: "Optional message or instructions"
```

**Behavior:**
- Displays optional note to operator
- Waits for Enter key press (or auto-continues after specified seconds)
- Useful for Q&A sessions, manual inspection, or dramatic pauses
- Continues to next step when ready

**Example:**
```yaml
- kind: pause
  seconds: 0
  note: "Q&A Session - Ask questions about the model comparison"

- kind: pause
  seconds: 30
  note: "Taking a 30-second break..."

- kind: pause
  note: "Review the PDF. Notice the tactical insights."
```

**Output:**
```
Q&A Session - Ask questions about the model comparison

Press Enter to continue...
```

---

## Demo Scenario Examples

### Example 1: Prototype Review (Standard)

**File:** `demo/scenarios/prototype_review.yaml`

```yaml
name: prototype_review
description: Demo scenario for prototype review meeting

steps:
  - kind: ensure_processed
    video: demo/input_videos/sample_match.mp4
    note: "Ensure demo video is processed"

  - kind: pause
    seconds: 0
    note: "Take a moment to review the input video"

  - kind: open_dashboard
    tab: model_comparison
    note: "Walk through the 5-model performance comparison"

  - kind: open_dashboard
    tab: accuracy_report
    note: "Review accuracy metrics and precision/recall"

  - kind: generate_pdf
    video_dir: demo/demo_outputs/sample_match
    note: "Generate comprehensive PDF report"

  - kind: pause
    seconds: 0
    note: "Review the generated PDF report. Q&A here"

  - kind: open_dashboard
    tab: highlights
    note: "Final: Review extracted highlights and key moments"
```

**Expected Duration:** 15-20 minutes

---

### Example 2: Quick 5-Minute Demo

**File:** `demo/scenarios/quick_demo.yaml`

```yaml
name: quick_demo
description: Quick 5-minute overview for busy stakeholders

steps:
  - kind: pause
    note: "Welcome to TactiVision - Football Analytics Platform"

  - kind: open_dashboard
    tab: Overview
    note: "Quick match overview with key metrics"

  - kind: open_dashboard
    tab: Heatmaps
    note: "Player positioning and heat maps"

  - kind: open_dashboard
    tab: Highlights
    note: "Auto-extracted highlights from the match"

  - kind: pause
    note: "Questions? Any other areas you'd like to explore?"
```

**Expected Duration:** 5-7 minutes

---

### Example 3: Technical Deep Dive

**File:** `demo/scenarios/technical_deep_dive.yaml`

```yaml
name: technical_deep_dive
description: In-depth technical demonstration for technical stakeholders

steps:
  - kind: ensure_processed
    video: demo/input_videos/technical_demo.mp4
    note: "Process demo video with full pipeline"

  - kind: open_dashboard
    tab: Database
    note: "Review raw data collected from video processing"

  - kind: open_dashboard
    tab: Tactical
    note: "Examine tactical analysis and formation detection"

  - kind: open_dashboard
    tab: "xG & Analytics"
    note: "Deep dive into xG model, shot quality, and advanced metrics"

  - kind: open_dashboard
    tab: "Player Performance"
    note: "Individual player tracking and performance metrics"

  - kind: generate_pdf
    video_dir: demo/demo_outputs/technical_demo
    note: "Generate detailed PDF with all technical sections"

  - kind: pause
    note: "Technical Q&A - Discuss architecture, data pipeline, and models"
```

**Expected Duration:** 30-45 minutes

---

### Example 4: Sales Demo

**File:** `demo/scenarios/sales_demo.yaml`

```yaml
name: sales_demo
description: High-level demo for prospective clients

steps:
  - kind: pause
    note: "Welcome to TactiVision Pro - Professional Football Analytics"

  - kind: open_dashboard
    tab: Overview
    note: "Beautiful, intuitive dashboard with match overview"

  - kind: open_dashboard
    tab: Tactical
    note: "Tactical insights help coaches plan and analyze"

  - kind: open_dashboard
    tab: Highlights
    note: "Automatic highlights generation saves time on video editing"

  - kind: generate_pdf
    video_dir: demo/demo_outputs/sample_match
    note: "Generate professional PDF reports for presentations and reports"

  - kind: pause
    note: "Pricing discussion and contract details"
```

**Expected Duration:** 20-30 minutes

---

## Creating Custom Scenarios

### Step 1: Plan Your Demo

Define the flow:
- What story do you want to tell?
- Which features to showcase?
- Where should Q&A occur?
- Estimated duration?

### Step 2: Prepare Videos

```bash
# Ensure you have processed demo videos
python -m services.batch_processor demo/input_videos/
```

### Step 3: Create YAML File

```bash
# Create new scenario file
touch demo/scenarios/my_custom_demo.yaml
```

### Step 4: Write Scenario

```yaml
name: my_custom_demo
description: My custom demonstration

steps:
  - kind: pause
    note: "Introduction"
  
  - kind: ensure_processed
    video: demo/input_videos/my_video.mp4
  
  - kind: open_dashboard
    tab: Overview
    note: "Show the overview"
  
  - kind: pause
    note: "Questions?"
```

### Step 5: Test Scenario

```bash
python -m services.scenario_player demo/scenarios/my_custom_demo.yaml
```

### Step 6: Refine

- Test with target audience
- Adjust timing and notes
- Add pauses where needed
- Verify video processing

---

## Running Demos

### Command Line

```bash
# Run specific scenario
python -m services.scenario_player demo/scenarios/prototype_review.yaml

# Run with error handling
python -m services.scenario_player demo/scenarios/prototype_review.yaml 2>&1 | tee demo.log
```

### Python Code

```python
from services.demo_runner import run_demo_scenario
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

try:
    run_demo_scenario(Path("demo/scenarios/prototype_review.yaml"))
except FileNotFoundError as e:
    print(f"Scenario not found: {e}")
except ValueError as e:
    print(f"Invalid scenario: {e}")
```

### Streamlit Dashboard

1. Start dashboard: `streamlit run demo/dashboard_final.py`
2. Navigate to "Generate Report" tab
3. Configure and generate reports manually
4. Download PDFs for distribution

---

## Best Practices

### Pre-Demo Checklist

- [ ] All videos are processed and `metrics.json` files exist
- [ ] Dashboard is running on correct URL/port
- [ ] Test scenario completely before live demo
- [ ] Have backup scenario if one fails
- [ ] Network connection is stable
- [ ] Presentation display is set up
- [ ] Audio/video is working
- [ ] Have printed handouts ready

### During Demo

- [ ] Follow script but be flexible
- [ ] Make eye contact with audience
- [ ] Pause for questions
- [ ] Highlight key features
- [ ] Use presenter notes
- [ ] Have backup demos ready
- [ ] Time each segment
- [ ] Watch for engagement

### Demo Environment Setup

```bash
# Start dashboard
streamlit run demo/dashboard_final.py --server.port=8501

# In another terminal, run scenario
python -m services.scenario_player demo/scenarios/prototype_review.yaml

# Or manually demo - open browser to http://localhost:8501
```

---

## Troubleshooting

### Scenario Won't Start

```
Error: Scenario file not found
Solution: Check file path and ensure it exists
```

### Video Processing Fails

```
Error: Video processing failed
Solution: Verify video file exists and is in supported format
         Check disk space and memory availability
```

### Dashboard Won't Open

```
Error: Connection refused
Solution: Ensure Streamlit is running on correct port
          Check firewall/network settings
          Try http://localhost:8501 in browser
```

### PDF Generation Fails

```
Error: Metrics not found
Solution: Run ensure_processed step first
          Verify metrics.json exists in directory
```

### Scenario Step Hangs

```
Issue: Pause step waiting for input
Solution: Press Enter to continue
          Check terminal is focused
          May need to click terminal window
```

---

## Advanced Features

### Custom Step Types

Extend demo runner with custom steps:

```python
# In services/demo_runner.py
class DemoRunner:
    STEP_KINDS = {
        'ensure_processed',
        'open_dashboard',
        'generate_pdf',
        'pause',
        'custom_analysis'  # NEW
    }
    
    def _step_custom_analysis(self, step):
        """Custom step implementation"""
        analysis_type = step.get('type')
        print(f"Running custom analysis: {analysis_type}")
```

### Scenario Validation

```python
import yaml
from pathlib import Path

def validate_scenario(scenario_path):
    with open(scenario_path) as f:
        scenario = yaml.safe_load(f)
    
    # Check required fields
    assert 'name' in scenario, "Missing 'name' field"
    assert 'steps' in scenario, "Missing 'steps' field"
    
    # Validate each step
    for step in scenario['steps']:
        kind = step.get('kind')
        assert kind in ['ensure_processed', 'open_dashboard', 'generate_pdf', 'pause']
    
    return True
```

---

## Demo Metrics & Analytics

### Track Demo Performance

```yaml
# Track in scenario metadata
metadata:
  duration_minutes: 15
  target_audience: "Technical stakeholders"
  success_metrics:
    - "Model comparison understood"
    - "Tactical insights appreciated"
    - "PDF report quality validated"
```

### Collect Feedback

```
Post-demo survey:
1. Was the demo clear and engaging?
2. Which feature was most impressive?
3. What questions remain?
4. Would you recommend TactiVision?
5. Interest level (1-10)?
```

---

## Automation & Scheduling

### Run Demos on Schedule

```bash
# Run daily at 2 PM
0 14 * * * cd /path/to/gpp && python -m services.scenario_player demo/scenarios/daily_demo.yaml

# Run weekly on Monday at 10 AM
0 10 * * 1 cd /path/to/gpp && python -m services.scenario_player demo/scenarios/weekly_demo.yaml
```

### Batch Multiple Demos

```bash
#!/bin/bash
# run_all_demos.sh

for scenario in demo/scenarios/*.yaml; do
    echo "Running: $scenario"
    python -m services.scenario_player "$scenario"
    echo "---"
done
```

---

## Distribution & Sharing

### Export Scenario

```bash
# Copy scenario and related files
cp demo/scenarios/prototype_review.yaml ~/shared/
cp -r demo/demo_outputs/ ~/shared/outputs/
```

### Share via GitHub

1. Commit scenario to Git
2. Push to repository
3. Team members clone and run

```bash
git add demo/scenarios/my_demo.yaml
git commit -m "Add new demo scenario"
git push origin demo-scenarios
```

---

## Performance & Optimization

### Scenario Timing

- Average step duration: 2-5 minutes
- Video processing: 5-15 minutes (first time)
- PDF generation: 10-20 seconds
- Dashboard navigation: 1-2 minutes per tab

### Optimization Tips

- Pre-process videos before demo
- Use quick_demo for time-constrained situations
- Cache metrics.json for repeated demos
- Run dashboard on dedicated machine
- Use high-speed network connection

---

## Example Complete Workflow

```bash
# 1. Prepare videos
cd /path/to/gpp
python -m services.batch_processor demo/input_videos/

# 2. Start dashboard
streamlit run demo/dashboard_final.py &

# 3. Wait for dashboard to start
sleep 5

# 4. Run demo scenario
python -m services.scenario_player demo/scenarios/prototype_review.yaml

# 5. After demo, generate report
# (This is handled by generate_pdf step)

# 6. Archive results
mkdir -p demo_archive/$(date +%Y%m%d_%H%M%S)
cp demo/demo_outputs/*.pdf demo_archive/$(date +%Y%m%d_%H%M%S)/
```

---

## Future Enhancements

- [ ] Record demo screen and audio
- [ ] Real-time audience polling during demo
- [ ] Multi-match comparison demos
- [ ] Interactive Q&A system
- [ ] Automated demo scheduling
- [ ] Demo performance analytics
- [ ] Multi-language scenario support
- [ ] Video streaming integration
