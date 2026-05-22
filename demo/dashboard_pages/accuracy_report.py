"""
demo/dashboard_pages/accuracy_report.py — Task D4 (Seif, Day 14)

Accuracy Report dashboard tab for TactiVision Pro.

This is the centerpiece of the discussion answer to "tracking accuracy was
very low."  It shows MOTA, IDF1, ID-switches/min, Detection F1, and mAP@50
per annotated clip — backed by numbers the doctor can re-produce with one
CLI command.

How to wire into dashboard_final.py
------------------------------------
Add one import at the top of dashboard_final.py:

    from demo.dashboard_pages import accuracy_report

Then add a new tab:

    tab1, tab2, ..., tab_accuracy = st.tabs([
        ..., "Accuracy Report"
    ])
    with tab_accuracy:
        accuracy_report.render()

Dependencies (all already in requirements.txt)
----------------------------------------------
    streamlit, pandas, plotly, services.accuracy_evaluator,
    services.ground_truth_manager, services.event_validator
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── path bootstrap ─────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # demo/dashboard_pages/
_ROOT = _HERE.parent.parent                      # project root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.accuracy_evaluator import evaluate
from services.ground_truth_manager import GroundTruthManager
from services.event_validator import evaluate_events

# ── paths ──────────────────────────────────────────────────────────────────
_GT_DIR    = _ROOT / "tests" / "ground_truth"
_PRED_BASE = _ROOT / "demo"  / "demo_outputs"
_REP_DIR   = _ROOT / "tests" / "accuracy_reports"

# Targets from plan §10.4
_TARGET_IDF1     = 0.65
_TARGET_IDF1_MIN = 0.50
_TARGET_MOTA     = 0.70
_TARGET_MOTA_MIN = 0.55
_TARGET_F1       = 0.75
_TARGET_MAP      = 0.65


# ══════════════════════════════════════════════════════════════════════════
# Pure-Python evaluation runner  (no Streamlit — can be called from CLI)
# ══════════════════════════════════════════════════════════════════════════

def run_evaluation_all_clips(
    gt_dir: Path   = _GT_DIR,
    pred_base: Path = _PRED_BASE,
    reports_dir: Path = _REP_DIR,
    iou_thr: float = 0.50,
) -> Optional[dict]:
    """
    Evaluate every annotated GT clip that has a matching per_frame_tracks.json.

    Saves a timestamped JSON report and returns the payload.
    Returns None if no clips could be evaluated.
    """
    mgr = GroundTruthManager(gt_dir=gt_dir)
    summaries = mgr.list_clips()

    clips_results: list[dict] = []
    skipped: list[str] = []

    for s in summaries:
        pred_path = pred_base / s.stem / "per_frame_tracks.json"
        if not pred_path.exists():
            skipped.append(
                f"{s.stem}  (no per_frame_tracks.json — run run_demo.py first)"
            )
            continue
        try:
            r = evaluate(s.gt_path, pred_path, iou_thr=iou_thr)
            r["stem"]               = s.stem
            r["n_frames_annotated"] = s.n_frames_annotated
            r["class_counts"]       = s.class_counts
            clips_results.append(r)
        except Exception as exc:
            skipped.append(f"{s.stem}  (error: {exc})")

    if not clips_results:
        return None

    agg = _aggregate(clips_results)
    payload = {
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "iou_threshold": iou_thr,
        "clips":         clips_results,
        "skipped":       skipped,
        "aggregate":     agg,
    }

    reports_dir.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = reports_dir / f"accuracy_report_{ts}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _aggregate(clips: list[dict]) -> dict:
    """Weighted-average metrics (weight = frames_scored per clip)."""
    total_frames = sum(c.get("frames_scored", 0) for c in clips)
    if total_frames == 0:
        return {}

    def _wavg(*keys: str) -> float:
        total = 0.0
        for c in clips:
            v: object = c
            for k in keys:
                v = v.get(k, 0) if isinstance(v, dict) else 0
            w = c.get("frames_scored", 0) / total_frames
            total += float(v) * w
        return round(total, 4)

    total_idsw_mota = sum(c.get("mota", {}).get("id_switches", 0) for c in clips)
    total_idsw_idm  = sum(c.get("identity", {}).get("id_switches", 0) for c in clips)
    total_min = sum(
        c.get("frames_scored", 0) / (c.get("fps", 25.0) * 60.0 + 1e-9)
        for c in clips
    )

    return {
        "mota":                    _wavg("mota", "mota"),
        "motp":                    _wavg("mota", "motp"),
        "idf1":                    _wavg("identity", "idf1"),
        "detection_f1":            _wavg("detection", "f1"),
        "detection_precision":     _wavg("detection", "precision"),
        "detection_recall":        _wavg("detection", "recall"),
        "map":                     _wavg("map", "mAP"),
        "id_switches_total":       total_idsw_idm,
        "id_switches_per_minute":  round(total_idsw_mota / total_min, 2)
                                   if total_min > 0 else 0.0,
        "frames_scored_total":     total_frames,
        "clips_evaluated":         len(clips),
    }


# ══════════════════════════════════════════════════════════════════════════
# Report I/O helpers
# ══════════════════════════════════════════════════════════════════════════

def _list_report_files(reports_dir: Path = _REP_DIR) -> list[Path]:
    """Most-recent-first list of saved report JSONs."""
    if not reports_dir.exists():
        return []
    files = sorted(reports_dir.glob("accuracy_report_*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def _load_report(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_latest_report(reports_dir: Path = _REP_DIR) -> Optional[dict]:
    files = _list_report_files(reports_dir)
    return _load_report(files[0]) if files else None


# ══════════════════════════════════════════════════════════════════════════
# Streamlit render entry-point
# ══════════════════════════════════════════════════════════════════════════

def render() -> None:
    """
    Render the full Accuracy Report tab.
    Call inside a ``with tab_accuracy:`` block in dashboard_final.py.
    """
    import streamlit as st

    # ── header ─────────────────────────────────────────────────────────
    st.markdown(
        """
        <h2 style="color:#f1f5f9;margin-bottom:0.2rem;">📊 Accuracy Report</h2>
        <p style="color:#94a3b8;margin-top:0;margin-bottom:1.5rem;">
        Tracking &amp; detection metrics measured against human-annotated
        ground truth. Every number here is reproducible with one CLI command.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── action bar ──────────────────────────────────────────────────────
    col_btn, col_sel, _ = st.columns([2, 3, 5])

    with col_btn:
        run_clicked = st.button(
            "▶ Run evaluation now",
            type="primary",
            help=(
                "Re-evaluates all annotated clips against pipeline outputs.\n"
                "Requires demo_outputs/<stem>/per_frame_tracks.json to exist."
            ),
        )

    # Load report to display
    report_data = _load_latest_report()

    with col_sel:
        report_files = _list_report_files()
        if report_files:
            chosen_name = st.selectbox(
                "Load saved report",
                options=[p.name for p in report_files],
                index=0,
                label_visibility="collapsed",
            )
            chosen_path = _REP_DIR / chosen_name
            if chosen_path != report_files[0]:
                report_data = _load_report(chosen_path)

    # ── run evaluation ──────────────────────────────────────────────────
    if run_clicked:
        with st.spinner("Running evaluator on all annotated clips…"):
            report_data = run_evaluation_all_clips()

        if report_data:
            n = len(report_data.get("clips", []))
            skipped = report_data.get("skipped", [])
            st.success(f"✅ Evaluated {n} clip(s) successfully.")
            if skipped:
                st.warning(
                    f"⚠️ {len(skipped)} clip(s) skipped:\n"
                    + "\n".join(f"  • {s}" for s in skipped)
                )
        else:
            st.error(
                "❌ No clips could be evaluated.\n\n"
                "Make sure:\n"
                "1. `tests/ground_truth/` contains annotated `.json` clips.\n"
                "2. `demo/run_demo.py` has been run to produce "
                "`demo_outputs/<stem>/per_frame_tracks.json`."
            )
        st.rerun()

    # ── no data state ───────────────────────────────────────────────────
    if not report_data:
        _render_no_data()
        return

    # ── KPI cards ───────────────────────────────────────────────────────
    _render_kpi_cards(report_data)
    st.divider()

    # ── per-clip table ──────────────────────────────────────────────────
    _render_per_clip_table(report_data)
    st.divider()

    # ── per-class event chart ────────────────────────────────────────────
    _render_event_chart(report_data)
    st.divider()

    # ── report metadata + CLI reproduce ─────────────────────────────────
    _render_report_meta(report_data)


# ══════════════════════════════════════════════════════════════════════════
# UI sub-renderers
# ══════════════════════════════════════════════════════════════════════════

def _render_no_data() -> None:
    import streamlit as st
    st.info(
        "No accuracy report found yet.  "
        "Click **▶ Run evaluation now** to generate one.",
        icon="ℹ️",
    )
    with st.expander("📖 How to generate ground truth & run the evaluator"):
        st.code(
            "# Step 1 — annotate a clip\n"
            "python tools/annotation_helper.py \\\n"
            "    --video input_videos/match.mp4 \\\n"
            "    --stride 25\n\n"
            "# Step 2 — run the pipeline\n"
            "python demo/run_demo.py \\\n"
            "    --video input_videos/match.mp4\n\n"
            "# Step 3 — evaluate from CLI\n"
            "python -m services.accuracy_evaluator \\\n"
            "    --gt  tests/ground_truth/match.json \\\n"
            "    --pred demo/demo_outputs/match/per_frame_tracks.json\n\n"
            "# Or click ▶ Run evaluation now above.",
            language="bash",
        )


def _color(val: float, lo: float, hi: float) -> str:
    if val >= hi:  return "#34d399"   # green
    if val >= lo:  return "#fbbf24"   # amber
    return "#fb7185"                  # red


def _kpi(label: str, value: str, sub: str, color: str) -> None:
    import streamlit as st
    st.markdown(
        f"""
        <div style="background:#1e293b;border-radius:10px;padding:1rem 1.2rem;
                    border-left:4px solid {color};margin-bottom:0.5rem;">
          <div style="color:#94a3b8;font-size:0.75rem;text-transform:uppercase;
                      letter-spacing:0.06em;margin-bottom:0.2rem;">{label}</div>
          <div style="color:{color};font-size:2rem;font-weight:700;
                      line-height:1.1;">{value}</div>
          <div style="color:#64748b;font-size:0.72rem;margin-top:0.2rem;">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_kpi_cards(report: dict) -> None:
    import streamlit as st
    agg = report.get("aggregate", {})
    if not agg:
        return

    mota      = agg.get("mota",               0.0)
    idf1      = agg.get("idf1",               0.0)
    f1        = agg.get("detection_f1",       0.0)
    m_ap      = agg.get("map",                0.0)
    id_sw_min = agg.get("id_switches_per_minute", 0.0)
    n_clips   = agg.get("clips_evaluated",    0)
    frames    = agg.get("frames_scored_total",0)

    st.markdown("### 📈 Overall Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        _kpi("MOTA", f"{mota:.3f}",
             f"target ≥ {_TARGET_MOTA}",
             _color(mota, _TARGET_MOTA_MIN, _TARGET_MOTA))
    with c2:
        _kpi("IDF1", f"{idf1:.3f}",
             f"target ≥ {_TARGET_IDF1}",
             _color(idf1, _TARGET_IDF1_MIN, _TARGET_IDF1))
    with c3:
        _kpi("Detection F1", f"{f1:.3f}",
             f"target ≥ {_TARGET_F1}",
             _color(f1, 0.60, _TARGET_F1))
    with c4:
        _kpi("mAP@50", f"{m_ap:.3f}",
             f"target ≥ {_TARGET_MAP}",
             _color(m_ap, 0.55, _TARGET_MAP))
    with c5:
        sw_color = "#34d399" if id_sw_min < 3 else ("#fbbf24" if id_sw_min < 6 else "#fb7185")
        _kpi("ID-sw / min", f"{id_sw_min:.2f}",
             f"{n_clips} clip(s) · {frames} frames",
             sw_color)


def _render_per_clip_table(report: dict) -> None:
    import streamlit as st
    clips = report.get("clips", [])
    if not clips:
        return

    st.markdown("### 🎬 Per-Clip Results")

    try:
        import pandas as pd
    except ImportError:
        st.warning("pandas not installed — cannot render table.")
        return

    rows = []
    for c in clips:
        det = c.get("detection", {})
        idm = c.get("identity",  {})
        mot = c.get("mota",      {})
        m   = c.get("map",       {})
        rows.append({
            "Clip":          c.get("stem", "?"),
            "Frames":        c.get("frames_scored", 0),
            "MOTA":          mot.get("mota",     0.0),
            "MOTP":          mot.get("motp",     0.0),
            "IDF1":          idm.get("idf1",     0.0),
            "ID-sw":         idm.get("id_switches", 0),
            "ID-sw/100fr":   idm.get("id_switch_rate_per_100_frames", 0.0),
            "Det F1":        det.get("f1",        0.0),
            "Precision":     det.get("precision", 0.0),
            "Recall":        det.get("recall",    0.0),
            "mAP@50":        m.get("mAP",         0.0),
            "GT objects":    det.get("true_positives", 0)
                             + det.get("false_negatives", 0),
        })

    df = pd.DataFrame(rows)

    def _style_mota(v):
        c = _color(v, _TARGET_MOTA_MIN, _TARGET_MOTA)
        return f"color:{c};font-weight:bold"

    def _style_idf1(v):
        c = _color(v, _TARGET_IDF1_MIN, _TARGET_IDF1)
        return f"color:{c};font-weight:bold"

    def _style_f1(v):
        c = _color(v, 0.60, _TARGET_F1)
        return f"color:{c};font-weight:bold"

    styled = (
        df.style
        .applymap(_style_mota, subset=["MOTA"])
        .applymap(_style_idf1, subset=["IDF1"])
        .applymap(_style_f1,   subset=["Det F1"])
        .format({
            "MOTA": "{:.3f}", "MOTP": "{:.3f}",
            "IDF1": "{:.3f}", "ID-sw/100fr": "{:.2f}",
            "Det F1": "{:.3f}", "Precision": "{:.3f}",
            "Recall": "{:.3f}", "mAP@50": "{:.3f}",
        })
    )
    st.dataframe(styled, use_container_width=True)

    # Skipped clips
    skipped = report.get("skipped", [])
    if skipped:
        with st.expander(f"⚠️ {len(skipped)} clip(s) skipped"):
            for s in skipped:
                st.markdown(f"- `{s}`")


def _render_event_chart(report: dict) -> None:
    import streamlit as st
    st.markdown("### 🎯 Per-Class Event Accuracy")

    # Aggregate event metrics from all clips if stored
    # (event metrics are stored when clips have event_validator results)
    all_event_data: dict[str, dict] = {}
    for c in report.get("clips", []):
        ev = c.get("event_metrics", {}).get("per_type", {})
        for etype, m in ev.items():
            if etype not in all_event_data:
                all_event_data[etype] = {"precision": [], "recall": [], "f1": []}
            all_event_data[etype]["precision"].append(m.get("precision", 0))
            all_event_data[etype]["recall"].append(m.get("recall", 0))
            all_event_data[etype]["f1"].append(m.get("f1", 0))

    if not all_event_data:
        st.info(
            "Event-level accuracy metrics are not yet available for these clips.\n\n"
            "They appear here once ground-truth event files are added to "
            "`tests/ground_truth/events_<stem>.json`.",
            icon="ℹ️",
        )
        return

    try:
        import plotly.graph_objects as go
        import pandas as pd
    except ImportError:
        st.warning("plotly / pandas not installed.")
        return

    rows = []
    for etype, vals in all_event_data.items():
        rows.append({
            "Event Type": etype,
            "Precision":  round(sum(vals["precision"]) / len(vals["precision"]), 3),
            "Recall":     round(sum(vals["recall"])    / len(vals["recall"]),    3),
            "F1":         round(sum(vals["f1"])        / len(vals["f1"]),        3),
        })
    df = pd.DataFrame(rows).sort_values("F1", ascending=True)

    fig = go.Figure()
    for metric, color in [("Precision", "#60a5fa"),
                           ("Recall",    "#34d399"),
                           ("F1",        "#f59e0b")]:
        fig.add_trace(go.Bar(
            name=metric, x=df[metric], y=df["Event Type"],
            orientation="h", marker_color=color,
        ))

    fig.update_layout(
        barmode="group",
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(color="#f1f5f9"),
        xaxis=dict(range=[0, 1], title="Score", gridcolor="#1e293b"),
        yaxis=dict(title=""),
        legend=dict(bgcolor="#1e293b"),
        height=max(300, len(rows) * 60),
        margin=dict(l=10, r=10, t=30, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_report_meta(report: dict) -> None:
    import streamlit as st
    ts      = report.get("timestamp", "unknown")
    iou     = report.get("iou_threshold", 0.5)
    skipped = report.get("skipped", [])
    agg     = report.get("aggregate", {})

    with st.expander("🔁 Reproduce these numbers"):
        st.markdown(
            "Run the following command to regenerate this exact report:\n"
        )
        for c in report.get("clips", []):
            stem = c.get("stem", "clip")
            st.code(
                f"python -m services.accuracy_evaluator \\\n"
                f"    --gt  tests/ground_truth/{stem}.json \\\n"
                f"    --pred demo/demo_outputs/{stem}/per_frame_tracks.json \\\n"
                f"    --iou {iou}",
                language="bash",
            )

    col1, col2, col3 = st.columns(3)
    col1.caption(f"🕐 Report timestamp: `{ts}`")
    col2.caption(f"📐 IoU threshold: `{iou}`")
    col3.caption(f"🎬 Clips evaluated: `{agg.get('clips_evaluated', 0)}`")


# ══════════════════════════════════════════════════════════════════════════
# CLI helper — run without Streamlit
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run accuracy evaluation on all GT clips and print aggregate."
    )
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--gt-dir",  type=Path, default=_GT_DIR)
    parser.add_argument("--pred-dir", type=Path, default=_PRED_BASE)
    args = parser.parse_args()

    report = run_evaluation_all_clips(
        gt_dir=args.gt_dir,
        pred_base=args.pred_dir,
        iou_thr=args.iou,
    )
    if not report:
        print("No clips evaluated. Check GT dir and pipeline outputs.")
        raise SystemExit(1)

    agg = report["aggregate"]
    print(f"\n=== AGGREGATE ({agg['clips_evaluated']} clips, "
          f"{agg['frames_scored_total']} frames) ===")
    print(f"MOTA  = {agg['mota']:.3f}   (target ≥ {_TARGET_MOTA})")
    print(f"IDF1  = {agg['idf1']:.3f}   (target ≥ {_TARGET_IDF1})")
    print(f"Det F1= {agg['detection_f1']:.3f}   (target ≥ {_TARGET_F1})")
    print(f"mAP50 = {agg['map']:.3f}   (target ≥ {_TARGET_MAP})")
    print(f"ID-sw/min = {agg['id_switches_per_minute']:.2f}")
    if report.get("skipped"):
        print(f"\nSkipped: {report['skipped']}")
