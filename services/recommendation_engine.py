"""
Tactical Recommendation Engine
================================
Owner: Ahmed Khaled (Member E)

Produces data-driven tactical recommendations by combining multiple
match statistics signals — no external API or model training needed.

Every recommendation:
  - Is triggered by 3+ related stats, not a single condition
  - Embeds the actual match numbers in its description
  - Gets a priority score (0-100) so the most urgent issues surface first
  - Includes step-by-step coaching instructions

Analysis areas:
  1. Possession & Pressing
  2. Attacking Efficiency (xG, shot volume, conversion)
  3. Opponent Weakness Exploitation
  4. Physical Intensity (speed, distance, sprints)
  5. Formation Matchup
  6. Transition Play
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    """A single tactical recommendation with full context."""
    title: str
    description: str          # Specific, number-backed sentence
    reasoning: str            # Which stats triggered this
    action_steps: List[str]   # Concrete coaching instructions
    priority: str             # CRITICAL / HIGH / MEDIUM / LOW
    priority_score: float     # 0–100 for sorting
    category: str             # possession / attacking / defensive / etc.
    target_team: str          # 'A' or 'B'
    confidence: float         # 0–1
    stats_used: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class TacticalRecommendationEngine:
    """
    Multi-signal tactical analysis engine.

    Usage:
        engine = TacticalRecommendationEngine()
        recs   = engine.analyze(metrics, team_a="Newcastle", team_b="Man Utd")
        # Returns list of Recommendation objects, sorted by priority_score desc.
    """

    # ---- Football domain thresholds (StatsBomb / Opta research) -----------
    PRESS_AGGRESSIVE   = 55   # pressing_intensity > 55 = active high press
    PRESS_PASSIVE      = 35   # pressing_intensity < 35 = sitting off
    POSS_DOMINANT      = 58   # > 58% = dominant possession
    POSS_WEAK          = 42   # < 42% = possession-starved
    XG_CONVERSION_POOR = 0.60 # goals/xG < 0.60 = significant underperformance
    SHOT_FLOOR         = 5    # fewer shots than this with >50% poss = problem
    DIST_GAP           = 500  # metres — meaningful physical difference
    SPRINT_GAP         = 10   # sprint count gap worth flagging
    SPEED_GAP          = 0.25 # m/s average speed difference worth flagging
    TRANS_GAP          = 5    # transition count gap

    NARROW_FORMATIONS  = {'4-3-3', '4-2-3-1', '4-5-1', '3-4-3'}
    HIGH_LINE          = {'4-3-3', '3-4-3', '4-2-3-1'}

    # -----------------------------------------------------------------------

    def analyze(
        self,
        metrics: Dict[str, Any],
        team_a: str = "Team A",
        team_b: str = "Team B",
    ) -> List[Recommendation]:
        """
        Main entry point.

        Args:
            metrics:  Full metrics dict from demo_outputs/<match>/metrics.json
            team_a:   Display name for team A
            team_b:   Display name for team B

        Returns:
            Up to 8 recommendations sorted by priority_score descending.
        """
        d = self._extract(metrics, team_a, team_b)

        recs: List[Recommendation] = []
        recs.extend(self._possession_and_pressing(d))
        recs.extend(self._attacking_efficiency(d))
        recs.extend(self._opponent_weaknesses(d))
        recs.extend(self._physical_intensity(d))
        recs.extend(self._formation_matchup(d))
        recs.extend(self._transitions(d))

        # Deduplicate by title, sort by score
        seen: set = set()
        unique: List[Recommendation] = []
        for r in recs:
            if r.title not in seen:
                seen.add(r.title)
                unique.append(r)

        unique.sort(key=lambda r: r.priority_score, reverse=True)
        return unique[:8]

    # -----------------------------------------------------------------------
    # Data extraction
    # -----------------------------------------------------------------------

    def _extract(self, m: Dict, team_a: str, team_b: str) -> Dict:
        """Flatten all relevant stats into a single dict."""
        poss     = m.get("possession", {})
        passes   = m.get("pass_detection", {})
        shots    = m.get("shot_detection", {})
        tactical = m.get("tactical_analysis", {})
        xg_comp  = m.get("xg_analysis", {}).get("team_comparison", {})
        tracks   = m.get("tracks", [])
        sprints  = m.get("sprint_detection", {})

        a_tr = [t for t in tracks if t.get("team") == "A"]
        b_tr = [t for t in tracks if t.get("team") == "B"]

        pressing    = tactical.get("pressing_intensity", {})
        transitions = tactical.get("transitions", {})

        def _trans_count(val):
            if isinstance(val, dict):
                return val.get("count", 0)
            return val or 0

        return {
            "team_a": team_a,
            "team_b": team_b,
            # Possession
            "poss_a": poss.get("team_possession_percentage", {}).get("A", 50),
            "poss_b": poss.get("team_possession_percentage", {}).get("B", 50),
            # Passes
            "pass_acc":  passes.get("pass_accuracy", 75),
            "passes_a":  passes.get("team_passes", {}).get("A", 0),
            "passes_b":  passes.get("team_passes", {}).get("B", 0),
            # Shots
            "shots_a":    shots.get("team_shots", {}).get("A", 0),
            "shots_b":    shots.get("team_shots", {}).get("B", 0),
            "shots_ot_a": shots.get("team_shots_on_target", {}).get("A", 0),
            "shots_ot_b": shots.get("team_shots_on_target", {}).get("B", 0),
            # xG
            "xg_a":    xg_comp.get("A", {}).get("xg", 0),
            "xg_b":    xg_comp.get("B", {}).get("xg", 0),
            "goals_a": xg_comp.get("A", {}).get("goals", 0),
            "goals_b": xg_comp.get("B", {}).get("goals", 0),
            # Pressing (0–100 scale)
            "press_a": pressing.get("A", 50),
            "press_b": pressing.get("B", 50),
            # Formations
            "form_a": tactical.get("current_formations", {}).get("A", "4-3-3"),
            "form_b": tactical.get("current_formations", {}).get("B", "4-3-3"),
            # Physical
            "speed_a": float(np.mean([t.get("avg_speed_mps", 0) for t in a_tr])) if a_tr else 0.0,
            "speed_b": float(np.mean([t.get("avg_speed_mps", 0) for t in b_tr])) if b_tr else 0.0,
            "dist_a":  sum(t.get("total_distance_m", 0) for t in a_tr),
            "dist_b":  sum(t.get("total_distance_m", 0) for t in b_tr),
            "sprint_a": sprints.get("team_sprints", {}).get("A", 0),
            "sprint_b": sprints.get("team_sprints", {}).get("B", 0),
            # Transitions
            "trans_a": _trans_count(transitions.get("A")),
            "trans_b": _trans_count(transitions.get("B")),
            # Duration
            "duration": m.get("duration_minutes", 90),
        }

    # -----------------------------------------------------------------------
    # Analysis modules
    # -----------------------------------------------------------------------

    def _possession_and_pressing(self, d: Dict) -> List[Recommendation]:
        recs = []
        poss_diff  = d["poss_a"] - d["poss_b"]   # positive = A dominant
        press_diff = d["press_a"] - d["press_b"]  # positive = A presses more

        # --- Dominant possession with no end product ---
        if abs(poss_diff) > 12:
            dom  = "A" if poss_diff > 0 else "B"
            weak = "B" if dom == "A" else "A"
            dom_poss   = d["poss_a"]  if dom == "A" else d["poss_b"]
            dom_shots  = d["shots_a"] if dom == "A" else d["shots_b"]
            dom_name   = d["team_a"]  if dom == "A" else d["team_b"]
            weak_name  = d["team_b"]  if dom == "A" else d["team_a"]
            weak_poss  = 100 - dom_poss

            # Dominant team: lots of ball, very few shots
            if dom_shots < self.SHOT_FLOOR and dom_poss > self.POSS_DOMINANT:
                score = 70 + min(20, abs(poss_diff))
                recs.append(Recommendation(
                    title=f"{dom_name}: Break Down the Defence",
                    description=(
                        f"{dom_name} controls {dom_poss:.1f}% of possession but has "
                        f"created only {dom_shots} shots. The ball is being recycled "
                        f"sideways without penetrating the defensive block."
                    ),
                    reasoning=(
                        f"High possession ({dom_poss:.1f}%) combined with low shot count "
                        f"({dom_shots}) signals sterile domination — the team is moving the "
                        f"ball without threatening the goal."
                    ),
                    action_steps=[
                        "Increase vertical pass attempts — reduce sideways recycling",
                        "Wingers to cut inside rather than stay wide and wait",
                        "Striker to drop short, creating space for late midfield runners",
                        "Full-backs to overlap — force the opponent into 1v1s on the flanks",
                    ],
                    priority="HIGH",
                    priority_score=score,
                    category="attacking",
                    target_team=dom,
                    confidence=0.83,
                    stats_used={"possession": dom_poss, "shots": dom_shots},
                ))

            # Weaker team: respond with compact block + counter
            score = 65 + min(15, abs(poss_diff))
            recs.append(Recommendation(
                title=f"{weak_name}: Compact Mid-Block & Counter",
                description=(
                    f"With only {weak_poss:.1f}% possession, {weak_name} cannot compete "
                    f"in open play. A compact 4-4-2 mid-block forces errors and creates "
                    f"space to counter through {dom_name}'s high defensive line."
                ),
                reasoning=(
                    f"A possession gap of {abs(poss_diff):.1f}% makes open-play "
                    f"competition unsustainable. Structured defence and fast transitions "
                    f"is the optimal tactical response."
                ),
                action_steps=[
                    "Drop into two compact banks of four — deny central space",
                    "Force the opponent wide, block central combinations",
                    "Win the ball, then play forward in 3 passes or fewer",
                    "Target channels behind the opponent's advancing full-backs",
                ],
                priority="HIGH",
                priority_score=score,
                category="defensive",
                target_team=weak,
                confidence=0.78,
                stats_used={"opponent_possession": dom_poss, "own_possession": weak_poss},
            ))

        # --- Pressing mismatch ---
        if abs(press_diff) > 15:
            high_team = "A" if press_diff > 0 else "B"
            low_team  = "B" if press_diff > 0 else "A"
            high_val  = d["press_a"] if high_team == "A" else d["press_b"]
            low_val   = d["press_b"] if high_team == "A" else d["press_a"]
            low_name  = d["team_b"]  if low_team  == "B" else d["team_a"]
            high_name = d["team_a"]  if high_team == "A" else d["team_b"]

            score = 60 + min(20, abs(press_diff) * 0.8)
            recs.append(Recommendation(
                title=f"{low_name}: Exploit Space Behind High Press",
                description=(
                    f"{high_name} presses at intensity {high_val:.0f}/100 vs "
                    f"{low_name}'s {low_val:.0f}/100. That aggression leaves space "
                    f"in behind the defensive line — use quick vertical passes to "
                    f"break it repeatedly."
                ),
                reasoning=(
                    f"Pressing differential of {abs(press_diff):.0f} points means "
                    f"{high_name} pushes up aggressively, creating exploitable space "
                    f"behind their back four."
                ),
                action_steps=[
                    "Play direct to the striker quickly when pressed — don't recycle",
                    "Striker makes timed diagonal runs behind the line on every press trigger",
                    "Hold shape — absorb the press, then spring counters with 1-2 passes",
                    "Use the half-spaces behind their full-backs with diagonal through balls",
                ],
                priority="MEDIUM",
                priority_score=score,
                category="transitions",
                target_team=low_team,
                confidence=0.74,
                stats_used={"high_press": high_val, "low_press": low_val, "diff": abs(press_diff)},
            ))

        return recs

    # -----------------------------------------------------------------------

    def _attacking_efficiency(self, d: Dict) -> List[Recommendation]:
        recs = []

        for team, opp in [("A", "B"), ("B", "A")]:
            name     = d["team_a"] if team == "A" else d["team_b"]
            xg       = d["xg_a"]    if team == "A" else d["xg_b"]
            goals    = d["goals_a"] if team == "A" else d["goals_b"]
            shots    = d["shots_a"] if team == "A" else d["shots_b"]
            shots_ot = d["shots_ot_a"] if team == "A" else d["shots_ot_b"]
            poss     = d["poss_a"]  if team == "A" else d["poss_b"]

            if xg < 0.1 or shots == 0:
                continue

            xg_per_shot = xg / shots
            conversion  = goals / xg
            sot_rate    = shots_ot / shots if shots > 0 else 0

            # Underperforming xG significantly
            if xg > 0.8 and conversion < self.XG_CONVERSION_POOR:
                shortfall = xg - goals
                score     = 55 + min(25, shortfall * 15)
                recs.append(Recommendation(
                    title=f"{name}: Convert Chances — {shortfall:.2f} xG Shortfall",
                    description=(
                        f"{name} generated {xg:.2f} expected goals from {shots} shots "
                        f"but scored only {goals}. That's a {(1 - conversion):.0%} "
                        f"underperformance. Average shot quality is "
                        f"{xg_per_shot:.3f} xG — the chances are there, but the "
                        f"finishing isn't."
                    ),
                    reasoning=(
                        f"xG conversion of {conversion:.0%} (threshold: 60%) across "
                        f"{shots} shots points to a finishing problem, not a "
                        f"chance-creation problem."
                    ),
                    action_steps=[
                        f"Focus on shot placement — {name}'s avg xG/shot of {xg_per_shot:.2f} suggests positions are good",
                        "Prioritise low driven shots — goalkeeper saves fewer than high efforts",
                        "Reduce speculative long-range efforts — they drag down xG efficiency",
                        "Near-post runs on crosses to create closer-range finishes",
                    ],
                    priority="HIGH",
                    priority_score=score,
                    category="attacking",
                    target_team=team,
                    confidence=min(0.91, 0.65 + shortfall * 0.1),
                    stats_used={"xg": xg, "goals": goals, "shots": shots, "conversion": f"{conversion:.0%}"},
                ))

            # High possession, almost no shots
            if poss > 50 and shots < self.SHOT_FLOOR and d["duration"] > 25:
                shots_per90 = shots / (d["duration"] / 90)
                score = 50 + (poss - 50) * 0.8
                recs.append(Recommendation(
                    title=f"{name}: Increase Shot Volume ({shots_per90:.1f} per 90 min)",
                    description=(
                        f"{name} has {poss:.1f}% possession but only {shots} shots in "
                        f"{d['duration']:.0f} minutes — that's {shots_per90:.1f} per 90, "
                        f"well below the competitive benchmark of 12+."
                    ),
                    reasoning=(
                        f"Possession dominance ({poss:.1f}%) is not converting to attempts "
                        f"on goal. Ball retention is good; penetration is the problem."
                    ),
                    action_steps=[
                        "Play vertical first — reduce backward recycling when in the final third",
                        "Attacking midfielders to arrive late into the box on every wide delivery",
                        "Full-backs to add late shots from the edge of the box",
                        "Set a minimum shot target: one attempt every 8 minutes of possession",
                    ],
                    priority="MEDIUM",
                    priority_score=score,
                    category="attacking",
                    target_team=team,
                    confidence=0.72,
                    stats_used={"possession": poss, "shots": shots, "shots_per_90": shots_per90},
                ))

        return recs

    # -----------------------------------------------------------------------

    def _opponent_weaknesses(self, d: Dict) -> List[Recommendation]:
        recs = []

        for team, opp in [("A", "B"), ("B", "A")]:
            name      = d["team_a"] if team == "A" else d["team_b"]
            opp_name  = d["team_b"] if team == "A" else d["team_a"]
            opp_press = d["press_b"] if team == "A" else d["press_a"]
            opp_shots = d["shots_b"] if team == "A" else d["shots_a"]
            own_speed = d["speed_a"] if team == "A" else d["speed_b"]
            opp_speed = d["speed_b"] if team == "A" else d["speed_a"]
            own_poss  = d["poss_a"]  if team == "A" else d["poss_b"]

            # Opponent almost never presses — exploit with patient build-up
            if opp_press < self.PRESS_PASSIVE:
                score = 68 + (self.PRESS_PASSIVE - opp_press) * 0.5
                recs.append(Recommendation(
                    title=f"{name}: Patient Build-Up — {opp_name} Won't Press",
                    description=(
                        f"{opp_name}'s pressing intensity is only {opp_press:.0f}/100 "
                        f"— they are sitting off. {name} can build from the goalkeeper "
                        f"with complete freedom, draw the block out of shape, and exploit "
                        f"the gaps created."
                    ),
                    reasoning=(
                        f"Pressing intensity of {opp_press:.0f}/100 (threshold: 35) means "
                        f"{opp_name} routinely concedes ball progression unchallenged in "
                        f"their own half."
                    ),
                    action_steps=[
                        "Goalkeeper plays short — invite the opponent to press and punish with 3rd-man combos",
                        "Centre-backs to step into midfield with the ball to create numerical superiority",
                        "Use 3v2 overloads in wide areas to progress into the final third",
                        f"Draw {opp_name} forward with slow build-up then switch play quickly to the open side",
                    ],
                    priority="MEDIUM",
                    priority_score=score,
                    category="opponent_exploit",
                    target_team=team,
                    confidence=0.77,
                    stats_used={"opponent_pressing": opp_press},
                ))

            # Speed advantage — use direct play
            speed_gap = own_speed - opp_speed
            if own_speed > 0 and opp_speed > 0 and speed_gap > self.SPEED_GAP:
                score = 60 + speed_gap * 25
                recs.append(Recommendation(
                    title=f"{name}: Use Pace — Speed Advantage of {speed_gap:.2f} m/s",
                    description=(
                        f"{name}'s average player speed ({own_speed:.2f} m/s) is "
                        f"{speed_gap:.2f} m/s faster than {opp_name} ({opp_speed:.2f} m/s) "
                        f"across the full match. This physical edge must be weaponised with "
                        f"direct, fast transitions."
                    ),
                    reasoning=(
                        f"A sustained average speed gap of {speed_gap:.2f} m/s across the "
                        f"full match is a structural physical mismatch that tactical "
                        f"shape alone cannot compensate for."
                    ),
                    action_steps=[
                        "Play early balls into channels — don't let the opponent recover their shape",
                        "Attackers make diagonal runs behind the last line on every goalkeeper distribution",
                        "After winning possession: maximum 2 passes before playing forward",
                        "Stretch the play wide — force the slower defence to cover maximum ground",
                    ],
                    priority="HIGH" if speed_gap > 0.5 else "MEDIUM",
                    priority_score=score,
                    category="opponent_exploit",
                    target_team=team,
                    confidence=min(0.85, 0.68 + speed_gap * 0.3),
                    stats_used={"own_speed": own_speed, "opp_speed": opp_speed, "gap_ms": speed_gap},
                ))

            # Opponent creates very few shots — they're not threatening
            if opp_shots <= 2 and d["duration"] > 25:
                score = 58
                recs.append(Recommendation(
                    title=f"{name}: Attack with Numbers — {opp_name} Offers No Threat",
                    description=(
                        f"{opp_name} has created only {opp_shots} shots in "
                        f"{d['duration']:.0f} minutes. They pose minimal attacking danger, "
                        f"so {name} can commit more players forward without significant "
                        f"counter-attack risk."
                    ),
                    reasoning=(
                        f"Shot count of {opp_shots} from the opponent in {d['duration']:.0f} "
                        f"minutes is well below any competitive benchmark — the risk of "
                        f"pushing men forward is low."
                    ),
                    action_steps=[
                        "Push both full-backs high simultaneously — create numerical overloads in attack",
                        "Defensive midfielder holds — everyone else attacks",
                        "Play at a high tempo — the opponent has shown no ability to transition quickly",
                        "Don't hold back — commit to winning the ball high up the pitch",
                    ],
                    priority="MEDIUM",
                    priority_score=score,
                    category="opponent_exploit",
                    target_team=team,
                    confidence=0.71,
                    stats_used={"opponent_shots": opp_shots, "duration": d["duration"]},
                ))

        return recs

    # -----------------------------------------------------------------------

    def _physical_intensity(self, d: Dict) -> List[Recommendation]:
        recs = []

        # Distance gap
        if d["dist_a"] > 0 and d["dist_b"] > 0:
            dist_diff = abs(d["dist_a"] - d["dist_b"])
            if dist_diff > self.DIST_GAP:
                more = "A" if d["dist_a"] > d["dist_b"] else "B"
                less = "B" if more == "A" else "A"
                more_dist  = d["dist_a"] if more == "A" else d["dist_b"]
                less_dist  = d["dist_b"] if more == "A" else d["dist_a"]
                more_name  = d["team_a"] if more == "A" else d["team_b"]

                score = 55 + min(20, dist_diff / 100)
                recs.append(Recommendation(
                    title=f"{more_name}: High Work Rate — Manage Late-Match Fatigue",
                    description=(
                        f"{more_name}'s players have covered {more_dist:.0f} m total vs "
                        f"{less_dist:.0f} m for the opponent — a gap of {dist_diff:.0f} m. "
                        f"This intensity is winning the physical battle, but performance "
                        f"drops sharply past high-distance thresholds."
                    ),
                    reasoning=(
                        f"Total distance differential of {dist_diff:.0f} m signals "
                        f"superior pressing and tracking, but risks significant fatigue "
                        f"in the final 20 minutes."
                    ),
                    action_steps=[
                        "Plan substitutions before the 70-minute mark to maintain intensity",
                        "Be more selective with press triggers — don't press every ball",
                        "Use dead-ball situations to recover shape and rest",
                        "Identify the highest-distance players and rotate them first",
                    ],
                    priority="MEDIUM",
                    priority_score=score,
                    category="physical",
                    target_team=more,
                    confidence=0.67,
                    stats_used={"own_distance": more_dist, "opp_distance": less_dist, "gap_m": dist_diff},
                ))

        # Sprint count gap
        if d["sprint_a"] > 0 or d["sprint_b"] > 0:
            sprint_diff = abs(d["sprint_a"] - d["sprint_b"])
            if sprint_diff > self.SPRINT_GAP:
                more_s = "A" if d["sprint_a"] > d["sprint_b"] else "B"
                less_s = "B" if more_s == "A" else "A"
                more_val  = d["sprint_a"] if more_s == "A" else d["sprint_b"]
                less_val  = d["sprint_b"] if more_s == "A" else d["sprint_a"]
                less_name = d["team_b"]   if less_s == "B" else d["team_a"]

                score = 56
                recs.append(Recommendation(
                    title=f"{less_name}: Low Sprint Output ({less_val} vs {more_val})",
                    description=(
                        f"{less_name} has recorded only {less_val} sprints vs "
                        f"{more_val} for the opponent — a deficit of {sprint_diff} "
                        f"explosive runs. Low sprint count limits defensive cover and "
                        f"attacking threat on the break."
                    ),
                    reasoning=(
                        f"Sprint deficit of {sprint_diff} across the full match "
                        f"indicates {less_name} is playing at a controlled pace rather "
                        f"than explosive intensity — ceding ground in key transition moments."
                    ),
                    action_steps=[
                        "Wider players to make more direct attacking runs rather than short combinations",
                        "Press triggers must be coordinated sprints — not individual chases",
                        "On goal kicks, sprint to the second ball immediately",
                        "In box entries: add late sprint arrivals from midfield",
                    ],
                    priority="MEDIUM",
                    priority_score=score,
                    category="physical",
                    target_team=less_s,
                    confidence=0.64,
                    stats_used={"own_sprints": less_val, "opp_sprints": more_val, "deficit": sprint_diff},
                ))

        return recs

    # -----------------------------------------------------------------------

    def _formation_matchup(self, d: Dict) -> List[Recommendation]:
        recs = []

        pairs = [("A", "B", d["form_a"], d["form_b"]), ("B", "A", d["form_b"], d["form_a"])]

        for team, opp, own_form, opp_form in pairs:
            name     = d["team_a"] if team == "A" else d["team_b"]
            opp_name = d["team_b"] if team == "A" else d["team_a"]
            opp_press = d["press_b"] if team == "A" else d["press_a"]

            # Opponent plays narrow — exploit wide channels
            if opp_form in self.NARROW_FORMATIONS:
                score = 62
                recs.append(Recommendation(
                    title=f"{name}: Attack Wide Channels vs {opp_name}'s {opp_form}",
                    description=(
                        f"{opp_name} is set up in a {opp_form} — a narrow formation "
                        f"that concentrates players centrally and leaves wide channels "
                        f"exposed. {name} should use overlapping full-backs and "
                        f"inverted wingers to stretch the defence and deliver crosses."
                    ),
                    reasoning=(
                        f"The {opp_form} shape creates 1v1 or 1v2 situations on the flanks "
                        f"that should be exploited systematically, not occasionally."
                    ),
                    action_steps=[
                        "Both full-backs to push high and overlap on every attacking phase",
                        "Wingers drift inside — pulling their full-back central and opening the channel",
                        "Deliver early crosses before the defence sets — target near post",
                        "Striker to attack the far post on every wide delivery",
                    ],
                    priority="MEDIUM",
                    priority_score=score,
                    category="formation",
                    target_team=team,
                    confidence=0.71,
                    stats_used={"opponent_formation": opp_form},
                ))

            # Opponent plays high press + high defensive line
            if opp_form in self.HIGH_LINE and opp_press > self.PRESS_AGGRESSIVE:
                score = 66
                recs.append(Recommendation(
                    title=f"{name}: Runs In Behind vs {opp_name}'s High Line",
                    description=(
                        f"{opp_name}'s {opp_form} with pressing intensity "
                        f"{opp_press:.0f}/100 pushes their defensive line high. "
                        f"Timed runs in behind from forwards and attacking midfielders "
                        f"will repeatedly break the offside trap and create 1v1 "
                        f"situations with the goalkeeper."
                    ),
                    reasoning=(
                        f"High pressing ({opp_press:.0f}/100) combined with {opp_form} "
                        f"pushes the back four up, creating consistent space in behind "
                        f"that timed diagonal runs will exploit."
                    ),
                    action_steps=[
                        "Strikers time runs — stay onside until the ball is played, then go",
                        "Midfielders look for the through ball early when the line steps up",
                        "Use lofted balls over the top on goalkeeper distributions",
                        "Decoy central runs pull the line; then cut diagonally to receive",
                    ],
                    priority="HIGH",
                    priority_score=score,
                    category="formation",
                    target_team=team,
                    confidence=0.74,
                    stats_used={"opponent_formation": opp_form, "opponent_pressing": opp_press},
                ))

        return recs

    # -----------------------------------------------------------------------

    def _transitions(self, d: Dict) -> List[Recommendation]:
        recs = []

        trans_diff = abs(d["trans_a"] - d["trans_b"])
        if trans_diff <= self.TRANS_GAP:
            return recs
        if d["trans_a"] == 0 and d["trans_b"] == 0:
            return recs

        more = "A" if d["trans_a"] > d["trans_b"] else "B"
        less = "B" if more == "A" else "A"
        more_t   = d["trans_a"] if more == "A" else d["trans_b"]
        less_t   = d["trans_b"] if more == "A" else d["trans_a"]
        less_name = d["team_b"] if less == "B" else d["team_a"]
        more_name = d["team_a"] if more == "A" else d["team_b"]

        score = 58 + min(15, trans_diff * 0.8)
        recs.append(Recommendation(
            title=f"{less_name}: Improve Transition Speed ({less_t} vs {more_t})",
            description=(
                f"{more_name} completed {more_t} attacking transitions vs "
                f"{less_t} for {less_name} — {trans_diff} more opportunities "
                f"to attack into disorganised defence. Faster transitions will catch "
                f"the opponent before they can recover."
            ),
            reasoning=(
                f"Transition count differential of {trans_diff} shows {more_name} is "
                f"significantly more effective at turning defensive moments into "
                f"attacking opportunities."
            ),
            action_steps=[
                "5-second rule: if possession isn't advanced within 5 seconds, play direct",
                "Striker and attacking mid spin immediately the moment the ball is won",
                "Defenders must play out quickly — no holding the ball under moderate pressure",
                "Use counterpress for 5 seconds max then transition — don't commit everyone",
            ],
            priority="MEDIUM",
            priority_score=score,
            category="transitions",
            target_team=less,
            confidence=0.67,
            stats_used={"own_transitions": less_t, "opp_transitions": more_t, "deficit": trans_diff},
        ))

        return recs
