const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
        ShadingType, PageNumber, PageBreak, LevelFormat, TabStopType, TabStopPosition } = require("docx");

const ACCENT = "1B4F72";
const ACCENT2 = "2E86C1";
const LIGHT_BG = "EBF5FB";
const WHITE = "FFFFFF";
const DARK = "2C3E50";

const border = { style: BorderStyle.SINGLE, size: 1, color: "BDC3C7" };
const borders = { top: border, bottom: border, left: border, right: border };
const noBorder = { style: BorderStyle.NONE, size: 0 };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

function heading(text, level) {
  return new Paragraph({ heading: level, children: [new TextRun({ text, font: "Arial" })] });
}

function spacer(size = 100) {
  return new Paragraph({ spacing: { after: size }, children: [] });
}

function featureRow(num, feature, description, isHeader = false) {
  const fill = isHeader ? ACCENT : (num % 2 === 0 ? LIGHT_BG : WHITE);
  const fontColor = isHeader ? WHITE : DARK;
  const bold = isHeader;
  return new TableRow({
    children: [
      new TableCell({
        borders, width: { size: 600, type: WidthType.DXA }, shading: { fill, type: ShadingType.CLEAR },
        margins: cellMargins, verticalAlign: "center",
        children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: String(num), bold, color: fontColor, font: "Arial", size: 20 })] })]
      }),
      new TableCell({
        borders, width: { size: 3200, type: WidthType.DXA }, shading: { fill, type: ShadingType.CLEAR },
        margins: cellMargins,
        children: [new Paragraph({ children: [new TextRun({ text: feature, bold, color: fontColor, font: "Arial", size: 20 })] })]
      }),
      new TableCell({
        borders, width: { size: 5560, type: WidthType.DXA }, shading: { fill, type: ShadingType.CLEAR },
        margins: cellMargins,
        children: [new Paragraph({ children: [new TextRun({ text: description, bold, color: fontColor, font: "Arial", size: 19 })] })]
      }),
    ]
  });
}

function statusRow(num, feature, description, status, isHeader = false) {
  const fill = isHeader ? "7D3C98" : (num % 2 === 0 ? "F5EEF8" : WHITE);
  const fontColor = isHeader ? WHITE : DARK;
  const bold = isHeader;
  const statusColor = isHeader ? WHITE : (status === "Planned" ? "E74C3C" : status === "In Research" ? "F39C12" : "8E44AD");
  return new TableRow({
    children: [
      new TableCell({
        borders, width: { size: 600, type: WidthType.DXA }, shading: { fill, type: ShadingType.CLEAR },
        margins: cellMargins, verticalAlign: "center",
        children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: String(num), bold, color: fontColor, font: "Arial", size: 20 })] })]
      }),
      new TableCell({
        borders, width: { size: 2800, type: WidthType.DXA }, shading: { fill, type: ShadingType.CLEAR },
        margins: cellMargins,
        children: [new Paragraph({ children: [new TextRun({ text: feature, bold, color: fontColor, font: "Arial", size: 20 })] })]
      }),
      new TableCell({
        borders, width: { size: 4560, type: WidthType.DXA }, shading: { fill, type: ShadingType.CLEAR },
        margins: cellMargins,
        children: [new Paragraph({ children: [new TextRun({ text: description, bold, color: fontColor, font: "Arial", size: 19 })] })]
      }),
      new TableCell({
        borders, width: { size: 1400, type: WidthType.DXA }, shading: { fill, type: ShadingType.CLEAR },
        margins: cellMargins, verticalAlign: "center",
        children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: status, bold, color: statusColor, font: "Arial", size: 19 })] })]
      }),
    ]
  });
}

function sectionDivider(title) {
  return new Paragraph({
    spacing: { before: 300, after: 200 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 3, color: ACCENT2, space: 4 } },
    children: [new TextRun({ text: title, bold: true, color: ACCENT, font: "Arial", size: 26 })]
  });
}

// COMPLETED FEATURES
const completedFeatures = [
  // Core Detection & Tracking
  ["Real-Time Player Detection & Tracking", "YOLOv8-powered multi-object detection with persistent ID assignment, handling occlusions, re-identification, and tracking up to 22+ players simultaneously across frames."],
  ["Intelligent Ball Detection & Tracking", "Dual-mode detection combining deep learning (YOLO) with color-based fallback. Trajectory prediction fills gaps of up to 5 missed frames using velocity-based physics modeling."],
  ["Ball Trajectory Prediction Engine", "Physics-aware prediction system that estimates ball position during occlusions using historical velocity vectors, direction computation, and speed cap validation (5000 px/s)."],
  ["Tracking Filters & Deduplication", "Advanced post-processing pipeline with speed sanity capping (12 m/s for players), moving average smoothing, and DeduplicatorV2 for merging fragmented track IDs."],
  ["Jersey Color Classification System", "HSV-based clustering algorithm that classifies player jersey colors in real-time, supporting 8+ color families including referee detection (neon/yellow)."],
  ["Jersey Number OCR Recognition", "EasyOCR-powered optical character recognition for jersey numbers, with intelligent fallback and roster-based player name mapping from JSON configuration."],

  // Event Detection
  ["Automated Pass Detection & Analysis", "Detects completed and intercepted passes using velocity thresholds (>1 m/s), distance validation (3-60m range), and team classification with 5-frame cooldown deduplication."],
  ["Shot Detection & Goal Recognition", "Identifies shots on goal using directional analysis (45-degree cone toward goal), velocity thresholds (>3 m/s), attacking third validation, and automatic goal confirmation within 60 frames."],
  ["Sprint Detection & Speed Analytics", "Monitors player sprints (>5.5 m/s threshold) and high-speed runs (>7.0 m/s), with minimum 1-second duration validation and physics-based rejection of impossible speeds."],
  ["Possession Tracking with Tactical Context", "Ball possession detection via proximity analysis, enhanced with pressure metrics, tactical zone tracking (defensive/midfield/attacking thirds), and possession duration classification."],
  ["Dribble & Take-On Detection", "Identifies dribbling sequences through ball control radius monitoring (50px), detects successful take-ons against defenders, tracks direction changes, and calculates opponents beaten."],
  ["Offside Detection System", "Calculates offside lines from second-last defender positions, detects potential and confirmed offside events with distance measurements and visualization data output."],
  ["Set Piece Recognition Engine", "Classifies corners, free kicks, throw-ins, penalties, and goal kicks using positional analysis, player formation detection (wall formation), and outcome classification with confidence scoring."],

  // Tactical Analysis
  ["Formation Detection & Classification", "Recognizes 16+ tactical formations (4-3-3, 4-4-2, 3-5-2, 4-2-3-1, etc.) through player position clustering into defensive/midfield/attacking lines with confidence scoring."],
  ["Formation Change Tracking", "Monitors and logs formation transitions throughout the match, detecting tactical shifts with timestamp recording and change frequency analysis."],
  ["Pressing Intensity Analysis", "Quantifies team pressing behavior by measuring opponent proximity, high-press triggers, and defensive line height across different match phases."],
  ["Transition Tracking System", "Detects and analyzes attacking and defensive transitions, measuring speed of play changes and positional shifts during possession changes."],
  ["Comprehensive Tactical Report Generator", "Integration hub that combines formation, offside, set piece, dribble, pressing, and transition data into unified TacticalReport objects with full metric aggregation."],

  // Advanced Analytics
  ["Expected Goals (xG) Calculator", "Calculates shot quality using distance-to-goal, angle, shot type (open play/set piece/header/penalty/free kick), body part, pressure level, and context-aware xG values."],
  ["Passing Network Analysis", "Graph-based analysis computing degree centrality, betweenness centrality, closeness centrality, network density, clustering coefficients, and average path length."],
  ["Zone Control Analysis", "Divides the pitch into a 3x3 tactical grid (9 zones) and measures team dominance through passes, touches, and time spent in each zone."],
  ["Pressure Index Calculation", "Real-time pressure measurement combining opponent proximity, pressing intensity, and defensive positioning to quantify match control dynamics."],
  ["Ball Progression Tracking", "Measures forward ball movement patterns, identifying progressive passes and carries that advance play toward the attacking third."],

  // Player Performance
  ["Player Performance Analytics Suite", "Comprehensive player rating system covering physical metrics (distance, speed, sprints, fatigue), technical metrics (passes, shots, accuracy), and tactical metrics (positioning, pressing, recovery)."],
  ["Position-Specific Benchmarking", "Evaluates player performance against position-based benchmarks, comparing defenders, midfielders, and forwards against role-specific expectations."],
  ["Player Profile System with Face Recognition", "Persistent player identity management using face recognition embeddings for cross-match tracking, with career statistics aggregation and performance trend analysis."],
  ["Comparative Player Analysis", "Side-by-side player comparison across all tracked metrics with form trend analysis (5+ match rolling windows) and performance ratings (0-100 scale)."],

  // AI & Intelligence
  ["AI Tactical Recommendations Engine", "Context-aware suggestion system with priority levels (Critical/High/Medium/Low), 9 recommendation categories, confidence scoring, risk assessment, and implementation steps."],
  ["Tactical Insights & Pattern Recognition", "Identifies passing sequences, attacking patterns, and defensive actions, classifying outcomes and detecting tactical styles (Possession, Direct, Counter, High Press, Tiki Taka, etc.)."],
  ["Team Strength/Weakness Analysis", "Automated assessment across 10+ tactical areas including attacking third, wide play, transitions, set pieces, pressing, counter-attacks, and ball retention."],
  ["Opponent Analysis & Scouting System", "Historical formation tendency tracking, playing style detection, zone preference analysis, and comprehensive scout report generation with player-specific tendencies."],

  // Visualization & Dashboard
  ["Interactive Streamlit Dashboard", "Full-featured web interface combining real-time match analysis, player tracking, tactical heatmaps, xG analytics, video highlights, and database integration with Plotly visualizations."],
  ["Tactical Heatmap Generator", "Smooth gradient heatmaps (blue-green-yellow-red) with pitch background, field markings, Gaussian blur, and support for global, team, and individual player views."],
  ["Pitch Coordinate Transformation", "Homography-based mapping from video pixels to real-world pitch coordinates (105m x 68m standard), with automatic calibration from pitch lines and manual 4-point calibration."],
  ["Broadcast-Quality Graphics Overlay", "Professional scoreboard, player stats overlays, and event popups (goal, card, substitution, VAR, penalty) with team logos, custom branding, and watermark support."],
  ["Automated Highlights Generation", "Key moment detection with importance scoring across 12 event types (goals, big chances, saves, tackles, dribbles, etc.) with customizable clip timing and match summary."],

  // Data & Integration
  ["SQLite Database Management System", "Full CRUD database with comprehensive schema for matches, players, events, statistics, face encodings, and historical data with CSV/JSON export capabilities."],
  ["Multi-Source API Integration", "Connectors for StatsBomb, Opta Sports, Understat, and Football-Data.org with rate limiting, response caching, retry logic with exponential backoff, and standardized data structures."],
  ["Live Score Integration", "Real-time score feeds for Premier League, La Liga, Serie A, Bundesliga, Ligue 1, Champions League, Europa League, World Cup, and Euro competitions."],
  ["Wearable Device Data Fusion", "GPS vest data import supporting Catapult and Prozone formats with heart rate monitoring, accelerometer data, metabolic power estimation, and fatigue index calculation."],
  ["Video-to-Wearable Time Synchronization", "Temporal alignment system that maps wearable sensor timestamps to video frame numbers for unified physical and visual analytics."],

  // Export & Streaming
  ["Multi-Format Video Export", "Professional export pipeline supporting MP4, AVI, MOV, and MKV with overlay options (tactical drawings, arrows, circles), zoom effects, slow-motion replay, and watermarks."],
  ["Social Media Export Engine", "Platform-optimized exports for Twitter (16:9), Instagram Reels (9:16), TikTok, YouTube Shorts, and Facebook with auto-captions, hashtag suggestions, and transition effects."],
  ["Real-Time Streaming Pipeline", "Live video processing supporting RTMP, RTSP, HLS, WebRTC, and SRT protocols with low-latency overlay injection, frame queue management, and stream statistics monitoring."],

  // Infrastructure
  ["End-to-End Processing Pipeline", "Orchestrated 11-step video analysis pipeline: frame extraction, YOLO detection, ball/player tracking, possession detection, event detection, sprint analysis, tactical analysis, statistics aggregation, heatmap generation, database storage, and highlight extraction."],
  ["Interactive CLI with Setup Wizard", "7-option menu system with CLI modes (dashboard/process/test/status/setup/export), automated dependency checking, GPU detection, YOLO model download, and database initialization."],
  ["Multi-Format Data Export System", "Match data export supporting JSON, CSV, PDF, and XLSX formats with configurable data selection and customizable output templates."],
  ["Comprehensive Test Suite", "Full testing framework with unit tests, integration tests, performance benchmarks, accuracy measurement, and ground truth validation across all major service modules."],
  ["Memory-Optimized Processing", "Intelligent resource management with garbage collection hooks, GPU memory cleanup, and progress tracking for processing large match videos efficiently."],
  ["Roster Configuration System", "JSON-based team roster management with player names, jersey numbers, team colors, and multi-match support for persistent player identity linking."],
];

// FUTURE FEATURES
const futureFeatures = [
  // Phase 2 - Advanced AI
  ["Deep Learning Pose Estimation", "Full-body skeletal tracking using MediaPipe/OpenPose to analyze player biomechanics, body orientation, and movement quality in real-time.", "Planned"],
  ["Computer Vision VAR Replay System", "Automated multi-angle incident review with frame-by-frame analysis, offside line overlay calibration, and handball detection using pose estimation.", "Planned"],
  ["Neural Network xG Model", "Machine learning-trained expected goals model using thousands of historical shots with feature engineering for defender positions, goalkeeper positioning, and game state.", "In Research"],
  ["Predictive Match Outcome Engine", "Real-time win probability calculation using Bayesian inference, incorporating xG flow, possession patterns, momentum shifts, and historical match data.", "Planned"],
  ["Natural Language Match Commentary", "GPT-powered automatic match narration generating real-time text commentary from detected events, tactical changes, and key moments.", "In Research"],
  ["Automated Tactical Animation Generator", "Converts detected plays and formations into animated 2D/3D tactical diagrams for coaching presentations and media content.", "Planned"],

  // Phase 3 - Enhanced Tracking
  ["Multi-Camera Fusion System", "Triangulation-based 3D player positioning using synchronized multi-angle video feeds with automatic camera calibration and view synthesis.", "Planned"],
  ["Player Fatigue Prediction Model", "ML-based fatigue forecasting using sprint decay curves, distance accumulation, heart rate recovery, and historical workload data to recommend substitution timing.", "Planned"],
  ["Injury Risk Assessment Module", "Biomechanical risk scoring combining sprint load, direction changes, asymmetric movement patterns, and accumulated fatigue with historical injury correlation.", "In Research"],
  ["Referee Decision Analysis", "Automated referee positioning tracking, decision consistency scoring, and incident proximity analysis for referee performance evaluation.", "Planned"],
  ["Crowd Density & Atmosphere Analytics", "Audio-visual analysis of crowd reactions, noise levels, and atmosphere shifts correlated with on-pitch events for engagement metrics.", "In Research"],

  // Phase 4 - Platform & Scale
  ["Cloud-Native Distributed Processing", "Kubernetes-orchestrated microservices architecture with auto-scaling GPU workers for parallel multi-match processing across AWS/GCP/Azure.", "Planned"],
  ["Real-Time Mobile Companion App", "Native iOS/Android application with live match dashboards, push notifications for key events, and offline tactical review capabilities.", "Planned"],
  ["REST API & SDK for Third-Party Integration", "Documented RESTful API with authentication, rate limiting, webhooks, and client SDKs (Python, JavaScript, Java) for ecosystem integration.", "Planned"],
  ["Team Collaboration Portal", "Multi-user web platform with role-based access (coach, analyst, player, scout), shared annotations, discussion threads, and collaborative session planning.", "Planned"],
  ["Advanced Report Builder", "Drag-and-drop report designer with customizable templates, automated post-match PDF generation, and scheduled email distribution to coaching staff.", "Planned"],
  ["Historical Season Analytics Dashboard", "Season-long trend analysis with league-wide benchmarking, player development tracking, team evolution metrics, and regression analysis.", "Planned"],

  // Phase 5 - Next Generation
  ["3D Match Reconstruction Engine", "Volumetric scene reconstruction from 2D video feeds generating interactive 3D match replays with free-camera exploration and VR headset support.", "In Research"],
  ["Augmented Reality Coaching Overlay", "AR glasses integration projecting real-time tactical data, player statistics, and formation suggestions directly onto the coach's field of view.", "In Research"],
  ["Transfer Market Valuation Model", "Data-driven player valuation combining performance analytics, age curves, contract data, market trends, and comparable player transfer history.", "Planned"],
  ["Youth Academy Talent Identification", "Long-term player development tracking with age-adjusted performance benchmarks, potential ceiling estimation, and development pathway recommendations.", "Planned"],
  ["Broadcast Integration SDK", "Direct integration with broadcast production systems (OBS, vMix, Vizrt) for live on-air graphics injection with sub-frame latency.", "Planned"],
  ["Federated Learning for Privacy-Preserving Analytics", "Cross-club model training without sharing raw data, enabling collaborative AI improvement while maintaining strict data sovereignty and GDPR compliance.", "In Research"],
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: ACCENT },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: ACCENT2 },
        paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 1 } },
    ]
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ]
  },
  sections: [
    // COVER PAGE
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
        }
      },
      children: [
        spacer(600), spacer(600), spacer(600), spacer(600),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 100 },
          children: [new TextRun({ text: "TACTIVISION PRO", font: "Arial", size: 56, bold: true, color: ACCENT })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 60 },
          children: [new TextRun({ text: "Advanced Football Video Analysis Platform", font: "Arial", size: 28, color: ACCENT2 })]
        }),
        spacer(200),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          border: { top: { style: BorderStyle.SINGLE, size: 3, color: ACCENT2, space: 8 }, bottom: { style: BorderStyle.SINGLE, size: 3, color: ACCENT2, space: 8 } },
          spacing: { before: 200, after: 200 },
          children: [new TextRun({ text: "Feature Portfolio & Development Roadmap", font: "Arial", size: 32, bold: true, color: DARK })]
        }),
        spacer(200),
        new Paragraph({
          alignment: AlignmentType.CENTER, spacing: { after: 80 },
          children: [new TextRun({ text: "Version 2.0.0", font: "Arial", size: 24, color: ACCENT2 })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER, spacing: { after: 80 },
          children: [new TextRun({ text: "March 2026", font: "Arial", size: 24, color: ACCENT2 })]
        }),
        spacer(600), spacer(600), spacer(600),
        new Paragraph({
          alignment: AlignmentType.CENTER, spacing: { after: 40 },
          children: [new TextRun({ text: "CONFIDENTIAL", font: "Arial", size: 20, bold: true, color: "E74C3C" })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "For Internal Use Only", font: "Arial", size: 18, color: "95A5A6", italics: true })]
        }),
      ]
    },
    // MAIN CONTENT
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1200, bottom: 1440, left: 1200 }
        }
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: ACCENT2, space: 4 } },
            children: [
              new TextRun({ text: "TactiVision Pro", font: "Arial", size: 18, bold: true, color: ACCENT }),
              new TextRun({ text: "\tFeature Portfolio & Roadmap", font: "Arial", size: 16, color: "95A5A6", italics: true }),
            ],
            tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
          })]
        })
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            border: { top: { style: BorderStyle.SINGLE, size: 1, color: "BDC3C7", space: 4 } },
            children: [
              new TextRun({ text: "Page ", font: "Arial", size: 16, color: "95A5A6" }),
              new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 16, color: "95A5A6" }),
              new TextRun({ text: "  |  Confidential", font: "Arial", size: 16, color: "95A5A6" }),
            ]
          })]
        })
      },
      children: [
        // EXECUTIVE SUMMARY
        heading("Executive Summary", HeadingLevel.HEADING_1),
        new Paragraph({
          spacing: { after: 200 },
          children: [new TextRun({ text: "TactiVision Pro is a state-of-the-art football video analysis platform that leverages deep learning, computer vision, and advanced analytics to deliver comprehensive match intelligence. The platform currently encompasses ", font: "Arial", size: 22, color: DARK }),
          new TextRun({ text: "48 fully implemented features", font: "Arial", size: 22, color: ACCENT, bold: true }),
          new TextRun({ text: " across 32 specialized service modules, with ", font: "Arial", size: 22, color: DARK }),
          new TextRun({ text: "22 additional features", font: "Arial", size: 22, color: "7D3C98", bold: true }),
          new TextRun({ text: " planned across 4 upcoming development phases.", font: "Arial", size: 22, color: DARK })]
        }),

        // STATS SUMMARY TABLE
        new Table({
          width: { size: 9840, type: WidthType.DXA },
          columnWidths: [2460, 2460, 2460, 2460],
          rows: [
            new TableRow({
              children: [
                ...[["48", "Completed Features"], ["32", "Service Modules"], ["22", "Planned Features"], ["4", "Upcoming Phases"]].map(([num, label]) =>
                  new TableCell({
                    borders: noBorders, width: { size: 2460, type: WidthType.DXA },
                    shading: { fill: ACCENT, type: ShadingType.CLEAR }, margins: { top: 120, bottom: 120, left: 80, right: 80 },
                    children: [
                      new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: num, font: "Arial", size: 40, bold: true, color: WHITE })] }),
                      new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: label, font: "Arial", size: 18, color: LIGHT_BG })] }),
                    ]
                  })
                )
              ]
            })
          ]
        }),

        spacer(200),

        // ============ COMPLETED FEATURES ============
        heading("Phase 1: Completed Features (v2.0.0)", HeadingLevel.HEADING_1),
        new Paragraph({
          spacing: { after: 200 },
          children: [new TextRun({ text: "The following 48 features have been fully developed, tested, and integrated into the production platform.", font: "Arial", size: 22, color: DARK })]
        }),

        // Category 1
        sectionDivider("1. Core Detection & Tracking Engine"),
        new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "The foundation of the platform, providing real-time multi-object detection and tracking capabilities powered by YOLOv8 and custom computer vision algorithms.", font: "Arial", size: 20, color: "7F8C8D", italics: true })] }),
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [600, 3200, 5560],
          rows: [
            featureRow("#", "Feature", "Description", true),
            ...completedFeatures.slice(0, 6).map((f, i) => featureRow(i + 1, f[0], f[1]))
          ]
        }),

        spacer(200),
        sectionDivider("2. Intelligent Event Detection System"),
        new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "Automated recognition and classification of match events using physics-based validation, spatial analysis, and temporal reasoning.", font: "Arial", size: 20, color: "7F8C8D", italics: true })] }),
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [600, 3200, 5560],
          rows: [
            featureRow("#", "Feature", "Description", true),
            ...completedFeatures.slice(6, 13).map((f, i) => featureRow(i + 7, f[0], f[1]))
          ]
        }),

        new Paragraph({ children: [new PageBreak()] }),

        sectionDivider("3. Tactical Analysis & Intelligence"),
        new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "Deep tactical understanding of team strategies, formations, and playing patterns with real-time analysis capabilities.", font: "Arial", size: 20, color: "7F8C8D", italics: true })] }),
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [600, 3200, 5560],
          rows: [
            featureRow("#", "Feature", "Description", true),
            ...completedFeatures.slice(13, 18).map((f, i) => featureRow(i + 14, f[0], f[1]))
          ]
        }),

        spacer(200),
        sectionDivider("4. Advanced Statistical Analytics"),
        new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "Sophisticated statistical models and network analysis algorithms for quantifying match dynamics and player contributions.", font: "Arial", size: 20, color: "7F8C8D", italics: true })] }),
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [600, 3200, 5560],
          rows: [
            featureRow("#", "Feature", "Description", true),
            ...completedFeatures.slice(18, 23).map((f, i) => featureRow(i + 19, f[0], f[1]))
          ]
        }),

        spacer(200),
        sectionDivider("5. Player Performance & Profiling"),
        new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "Comprehensive player evaluation framework combining biometric data, performance metrics, and persistent identity management.", font: "Arial", size: 20, color: "7F8C8D", italics: true })] }),
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [600, 3200, 5560],
          rows: [
            featureRow("#", "Feature", "Description", true),
            ...completedFeatures.slice(23, 27).map((f, i) => featureRow(i + 24, f[0], f[1]))
          ]
        }),

        new Paragraph({ children: [new PageBreak()] }),

        sectionDivider("6. AI-Powered Intelligence & Scouting"),
        new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "Machine intelligence systems that transform raw data into actionable coaching insights and opponent intelligence.", font: "Arial", size: 20, color: "7F8C8D", italics: true })] }),
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [600, 3200, 5560],
          rows: [
            featureRow("#", "Feature", "Description", true),
            ...completedFeatures.slice(27, 31).map((f, i) => featureRow(i + 28, f[0], f[1]))
          ]
        }),

        spacer(200),
        sectionDivider("7. Visualization & Dashboard Platform"),
        new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "Professional-grade visualization tools and interactive dashboards for presenting complex analytics in intuitive formats.", font: "Arial", size: 20, color: "7F8C8D", italics: true })] }),
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [600, 3200, 5560],
          rows: [
            featureRow("#", "Feature", "Description", true),
            ...completedFeatures.slice(31, 36).map((f, i) => featureRow(i + 32, f[0], f[1]))
          ]
        }),

        spacer(200),
        sectionDivider("8. Data Management & External Integrations"),
        new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "Robust data infrastructure connecting the platform with external data sources, wearable devices, and industry-standard APIs.", font: "Arial", size: 20, color: "7F8C8D", italics: true })] }),
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [600, 3200, 5560],
          rows: [
            featureRow("#", "Feature", "Description", true),
            ...completedFeatures.slice(36, 41).map((f, i) => featureRow(i + 37, f[0], f[1]))
          ]
        }),

        new Paragraph({ children: [new PageBreak()] }),

        sectionDivider("9. Export, Streaming & Distribution"),
        new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "Multi-platform content distribution supporting professional broadcast, social media, and live streaming workflows.", font: "Arial", size: 20, color: "7F8C8D", italics: true })] }),
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [600, 3200, 5560],
          rows: [
            featureRow("#", "Feature", "Description", true),
            ...completedFeatures.slice(41, 44).map((f, i) => featureRow(i + 42, f[0], f[1]))
          ]
        }),

        spacer(200),
        sectionDivider("10. Platform Infrastructure & DevOps"),
        new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "Core platform architecture ensuring reliability, performance, and maintainability across all system components.", font: "Arial", size: 20, color: "7F8C8D", italics: true })] }),
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [600, 3200, 5560],
          rows: [
            featureRow("#", "Feature", "Description", true),
            ...completedFeatures.slice(44).map((f, i) => featureRow(i + 45, f[0], f[1]))
          ]
        }),

        // ============ FUTURE ROADMAP ============
        new Paragraph({ children: [new PageBreak()] }),
        heading("Development Roadmap: Upcoming Phases", HeadingLevel.HEADING_1),
        new Paragraph({
          spacing: { after: 200 },
          children: [new TextRun({ text: "The following features represent the next evolution of TactiVision Pro, organized across four strategic development phases designed to push the boundaries of football analytics technology.", font: "Arial", size: 22, color: DARK })]
        }),

        // Phase 2
        sectionDivider("Phase 2: Advanced AI & Predictive Intelligence"),
        new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "Elevating the platform with deep learning models, predictive analytics, and natural language processing for next-generation match intelligence.", font: "Arial", size: 20, color: "7F8C8D", italics: true })] }),
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [600, 2800, 4560, 1400],
          rows: [
            statusRow("#", "Feature", "Description", "Status", true),
            ...futureFeatures.slice(0, 6).map((f, i) => statusRow(i + 1, f[0], f[1], f[2]))
          ]
        }),

        spacer(200),
        sectionDivider("Phase 3: Enhanced Tracking & Biometric Intelligence"),
        new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "Next-level tracking capabilities combining multi-camera systems, biomechanical analysis, and predictive health monitoring.", font: "Arial", size: 20, color: "7F8C8D", italics: true })] }),
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [600, 2800, 4560, 1400],
          rows: [
            statusRow("#", "Feature", "Description", "Status", true),
            ...futureFeatures.slice(6, 11).map((f, i) => statusRow(i + 7, f[0], f[1], f[2]))
          ]
        }),

        new Paragraph({ children: [new PageBreak()] }),

        sectionDivider("Phase 4: Platform Scale & Collaboration"),
        new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "Transforming TactiVision Pro from a standalone tool into a scalable, cloud-native platform with multi-user collaboration and ecosystem integrations.", font: "Arial", size: 20, color: "7F8C8D", italics: true })] }),
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [600, 2800, 4560, 1400],
          rows: [
            statusRow("#", "Feature", "Description", "Status", true),
            ...futureFeatures.slice(11, 17).map((f, i) => statusRow(i + 12, f[0], f[1], f[2]))
          ]
        }),

        spacer(200),
        sectionDivider("Phase 5: Next-Generation Innovation"),
        new Paragraph({ spacing: { after: 100 }, children: [new TextRun({ text: "Breakthrough technologies that will define the future of football analytics, from 3D reconstruction to augmented reality coaching.", font: "Arial", size: 20, color: "7F8C8D", italics: true })] }),
        new Table({
          width: { size: 9360, type: WidthType.DXA }, columnWidths: [600, 2800, 4560, 1400],
          rows: [
            statusRow("#", "Feature", "Description", "Status", true),
            ...futureFeatures.slice(17).map((f, i) => statusRow(i + 18, f[0], f[1], f[2]))
          ]
        }),

        spacer(400),

        // CLOSING
        new Paragraph({
          border: { top: { style: BorderStyle.SINGLE, size: 3, color: ACCENT, space: 8 }, bottom: { style: BorderStyle.SINGLE, size: 3, color: ACCENT, space: 8 } },
          spacing: { before: 300, after: 300 },
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "Total Platform Scope: 70 Features", font: "Arial", size: 28, bold: true, color: ACCENT }),
            new TextRun({ text: "  |  ", font: "Arial", size: 28, color: "BDC3C7" }),
            new TextRun({ text: "48 Delivered", font: "Arial", size: 28, bold: true, color: "27AE60" }),
            new TextRun({ text: "  |  ", font: "Arial", size: 28, color: "BDC3C7" }),
            new TextRun({ text: "22 In Pipeline", font: "Arial", size: 28, bold: true, color: "7D3C98" }),
          ]
        }),
      ]
    }
  ]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("C:\\Users\\mohan\\OneDrive\\Desktop\\gpp\\TactiVision_Pro_Feature_Portfolio.docx", buffer);
  console.log("Document created successfully!");
});
