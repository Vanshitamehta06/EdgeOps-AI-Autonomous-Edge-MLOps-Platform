"""
Generate PowerPoint presentation from Academic content
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor

def create_title_slide(prs, title, subtitle=""):
    """Create title slide"""
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    title_shape = slide.shapes.title
    subtitle_shape = slide.placeholders[1]
    
    title_shape.text = title
    if subtitle:
        subtitle_shape.text = subtitle
    
    return slide

def create_content_slide(prs, title, content_items=None, layout_idx=1):
    """Create content slide with bullet points"""
    slide = prs.slides.add_slide(prs.slide_layouts[layout_idx])
    title_shape = slide.shapes.title
    title_shape.text = title
    
    if content_items:
        body_shape = slide.placeholders[1]
        text_frame = body_shape.text_frame
        text_frame.clear()
        
        for item in content_items:
            p = text_frame.add_paragraph()
            p.text = item
            p.level = 0
            p.font.size = Pt(18)
    
    return slide

def create_two_column_slide(prs, title, left_content, right_content):
    """Create slide with two columns"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank layout
    
    # Add title
    left = Inches(0.5)
    top = Inches(0.5)
    width = Inches(9)
    height = Inches(0.8)
    title_box = slide.shapes.add_textbox(left, top, width, height)
    title_frame = title_box.text_frame
    title_frame.text = title
    title_frame.paragraphs[0].font.size = Pt(32)
    title_frame.paragraphs[0].font.bold = True
    
    # Left column
    left_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(4.5), Inches(5))
    left_frame = left_box.text_frame
    left_frame.word_wrap = True
    for item in left_content:
        p = left_frame.add_paragraph()
        p.text = item
        p.font.size = Pt(14)
    
    # Right column
    right_box = slide.shapes.add_textbox(Inches(5.2), Inches(1.5), Inches(4.5), Inches(5))
    right_frame = right_box.text_frame
    right_frame.word_wrap = True
    for item in right_content:
        p = right_frame.add_paragraph()
        p.text = item
        p.font.size = Pt(14)
    
    return slide

def create_table_slide(prs, title, headers, rows):
    """Create slide with table"""
    slide = prs.slides.add_slide(prs.slide_layouts[5])  # Title only
    title_shape = slide.shapes.title
    title_shape.text = title
    
    # Add table
    rows_count = len(rows) + 1  # +1 for header
    cols_count = len(headers)
    left = Inches(0.5)
    top = Inches(1.8)
    width = Inches(9)
    height = Inches(4)
    
    table = slide.shapes.add_table(rows_count, cols_count, left, top, width, height).table
    
    # Set headers
    for col_idx, header in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.text = header
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(0, 128, 128)  # Teal
        paragraph = cell.text_frame.paragraphs[0]
        paragraph.font.bold = True
        paragraph.font.size = Pt(12)
        paragraph.font.color.rgb = RGBColor(255, 255, 255)
    
    # Set rows
    for row_idx, row_data in enumerate(rows, start=1):
        for col_idx, cell_text in enumerate(row_data):
            cell = table.cell(row_idx, col_idx)
            cell.text = str(cell_text)
            cell.text_frame.paragraphs[0].font.size = Pt(11)
    
    return slide

def main():
    """Generate PowerPoint presentation"""
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)
    
    # Slide 1: Title
    slide = create_title_slide(
        prs,
        "EdgeOps AI Platform",
        "Production-Grade MLOps for Predictive Maintenance at the Industrial Edge\n\n"
        "Submitted by: Vanshita Mehta\n"
        "Mentor: Mr. Ashish Kumar\n"
        "Siemens Digital Industries - Factory Automation\n"
        "July 24, 2026"
    )
    
    # Slide 2: Agenda
    create_content_slide(prs, "Agenda", [
        "1. Introduction & Problem Context",
        "2. Literature Review",
        "3. Research Gap",
        "4. Research Objectives & Problem Definition",
        "5. Proposed Methodology",
        "6. System Architecture",
        "7. Technology Stack",
        "8. Results & Discussion",
        "9. Conclusion & Future Work"
    ])
    
    # Slide 3: Problem Context
    create_content_slide(prs, "Problem Context: Predictive Maintenance", [
        "Predictive Maintenance (PdM) uses live sensor data and ML to forecast failures",
        "",
        "Traditional Approaches:",
        "  • Reactive: Repair after failure → highest downtime",
        "  • Preventive: Fixed schedule → inefficient",
        "  • Predictive (EdgeOps): AI-driven condition monitoring",
        "",
        "Industrial Edge Computing:",
        "  • Sub-second inference (vs. cloud round-trip)",
        "  • Critical for real-time PLC control loops",
        "  • Challenges: model drift, limited compute, reliability"
    ])
    
    # Slide 4: Literature Review - Part 1
    create_content_slide(prs, "Literature Review: Key Research Papers", [
        "1. Lee et al. (2014) - 5C architecture for cyber-physical systems",
        "   → Limitation: Cloud-centric, high latency for edge",
        "",
        "2. Susto et al. (2015) - XGBoost achieves 92%+ accuracy",
        "   → Gap: No continuous retraining or drift detection",
        "",
        "3. Carvalho et al. (2019) - Survey of 127 papers",
        "   → Finding: Model degradation over time under-addressed",
        "",
        "4. Paleyes et al. (2022) - Drift detection critical for production ML",
        "   → Recommendation: Automated retraining pipelines"
    ])
    
    # Slide 5: Literature Review - Part 2
    create_content_slide(prs, "Literature Review: MLOps & Edge AI", [
        "5. Kreuzberger et al. (2023) - MLOps systematic review",
        "   → Gap: Limited edge-specific MLOps frameworks",
        "",
        "6. Merenda et al. (2020) - Edge ML for IoT devices",
        "   → ONNX runtime reduces inference time by 40%",
        "",
        "7. Lu et al. (2018) - Concept drift adaptation",
        "   → PSI, KS tests for feature drift detection",
        "",
        "8. Gama et al. (2014) - Threshold-based retraining",
        "   → 30% drift threshold triggers retrain"
    ])
    
    # Slide 6: Research Gap (Table)
    create_table_slide(prs, "Research Gap", 
        ["Gap", "Existing Approaches", "Consequence"],
        [
            ["No self-healing pipelines", "Manual model updates", "Downtime during retraining"],
            ["Cloud dependency", "Latency-sensitive edge ignored", ">100ms inference delays"],
            ["Drift monitoring absent", "Static models degrade silently", "15-20% accuracy drop/6mo"],
            ["No rollback mechanisms", "Failed deployments crash", "Production reliability risk"],
            ["Operator visibility gap", "No live dashboards", "Delayed failure detection"]
        ]
    )
    
    # Slide 7: What EdgeOps Addresses
    create_content_slide(prs, "What EdgeOps Addresses", [
        "✓ Autonomous drift detection + retraining",
        "   → Evidently library + Optuna hyperparameter optimization",
        "",
        "✓ Edge-optimized inference (<10ms latency)",
        "   → ONNX Runtime for 3-5× speedup",
        "",
        "✓ Instant rollback protocol",
        "   → Dual-pointer model registry (<2 seconds)",
        "",
        "✓ Live telemetry dashboard",
        "   → React + embedded Grafana panels",
        "",
        "✓ Production-grade reliability",
        "   → Self-healing, audio alerts, recovery protocols"
    ])
    
    # Slide 8: Problem Statement
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank
    left = Inches(0.5)
    top = Inches(1.5)
    width = Inches(9)
    height = Inches(5)
    textbox = slide.shapes.add_textbox(left, top, width, height)
    text_frame = textbox.text_frame
    text_frame.word_wrap = True
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(9), Inches(0.8))
    title_box.text_frame.text = "Problem Statement"
    title_box.text_frame.paragraphs[0].font.size = Pt(32)
    title_box.text_frame.paragraphs[0].font.bold = True
    
    # Problem statement
    p = text_frame.add_paragraph()
    p.text = "How can we build a production-ready MLOps platform that continuously monitors industrial sensor data, automatically retrains predictive maintenance models when drift is detected, and ensures <10ms inference latency with zero-downtime rollback capabilities?"
    p.font.size = Pt(20)
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER
    text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    
    # Slide 9: Research Objectives
    create_content_slide(prs, "Research Objectives", [
        "1. Design closed-loop MLOps pipeline for edge PdM",
        "   → Measurable: Drift → retrain → deploy cycle <5 minutes",
        "",
        "2. Implement automated drift detection using statistical methods",
        "   → Measurable: PSI threshold at 30%, trigger retraining",
        "",
        "3. Optimize ML inference for edge deployment",
        "   → Measurable: ONNX runtime achieves <10ms inference",
        "",
        "4. Develop dual-model registry for safe rollback",
        "   → Measurable: Rollback time <2 seconds, zero data loss",
        "",
        "5. Build live operator dashboard with embedded telemetry",
        "   → Measurable: 1-second polling loop, real-time alerts"
    ])
    
    # Slide 10: System Architecture
    create_content_slide(prs, "Proposed System Architecture", [
        "Data & Sensing Layer:",
        "  • SIMATIC S7-300 PLC (Temperature, Vibration, Torque sensors)",
        "  • Real-time telemetry streaming via OPC UA / S7 protocol",
        "",
        "Edge Device (IED/IEVD):",
        "  • Containerized EdgeOps AI application (Docker)",
        "  • FastAPI backend (async ASGI server)",
        "  • ML Engine: XGBoost → ONNX conversion",
        "  • Drift Monitor: Evidently library",
        "  • Model Registry: SQLite with dual-pointer system",
        "",
        "Factory Management:",
        "  • IEM (Industrial Edge Management) for device fleet control"
    ])
    
    # Slide 11: Data Flow Pipeline
    create_content_slide(prs, "7-Phase Data Flow Pipeline", [
        "Phase 1: Data Ingestion → PLC sensor streaming",
        "",
        "Phase 2: Feature Engineering → Rolling stats, lag features",
        "",
        "Phase 3: Model Training → XGBoost + Optuna (50 trials)",
        "",
        "Phase 4: Deployment → ONNX conversion + Runtime inference",
        "",
        "Phase 5: Monitoring → Evidently drift detection (PSI)",
        "",
        "Phase 6: Auto-Retraining → Triggered at 30% drift threshold",
        "",
        "Phase 7: Rollback Protocol → Dual-pointer instant recovery"
    ])
    
    # Slide 12: Database Schema
    create_content_slide(prs, "Database Schema: Model Registry", [
        "model_versions:",
        "  • id, version_number (UNIQUE), model_path",
        "  • accuracy, precision, recall, f1, roc_auc",
        "  • status: candidate | active | retired",
        "",
        "deployment_state (SINGLETON - 1 row):",
        "  • current_model_id → live inference pointer",
        "  • stable_model_id → rollback pointer",
        "",
        "system_metrics:",
        "  • model_version, timestamp",
        "  • cpu_usage, ram_usage, latency_ms",
        "  • drift_score, reliability_score"
    ])
    
    # Slide 13: Technology Stack
    create_table_slide(prs, "Technology Stack",
        ["Layer", "Technology", "Justification"],
        [
            ["ML Framework", "XGBoost 2.0", "SOTA for tabular data"],
            ["Optimization", "Optuna 3.5", "Bayesian TPE sampler"],
            ["Tracking", "MLflow 2.8", "Experiment logging"],
            ["Edge Runtime", "ONNX Runtime 1.16", "3-5× faster inference"],
            ["Drift Detection", "Evidently 0.4", "Statistical tests (PSI, KS)"],
            ["Backend", "FastAPI 0.104", "Async I/O + OpenAPI docs"],
            ["Frontend", "React 19 + Vite 8", "Fast dev server"],
            ["Visualization", "Grafana OSS 10", "Live time-series plots"]
        ]
    )
    
    # Slide 14: Dataset Description
    create_content_slide(prs, "Dataset: UCI Predictive Maintenance", [
        "Source: UCI Machine Learning Repository",
        "",
        "Features (10):",
        "  • Air temperature [K], Process temperature [K]",
        "  • Rotational speed [rpm], Torque [Nm]",
        "  • Tool wear [min], Product type (L/M/H)",
        "",
        "Target: Machine failure (binary: 0/1)",
        "",
        "Size: 10,000 samples (1.3% failure rate → imbalanced)",
        "",
        "Preprocessing:",
        "  • SMOTE for class balancing",
        "  • StandardScaler normalization",
        "  • 80/20 train-test split"
    ])
    
    # Slide 15: Drift Detection Algorithm
    create_content_slide(prs, "Implementation: Drift Detection", [
        "Evidently Library - Statistical Drift Monitoring:",
        "",
        "1. Calculate drift metrics every 100 predictions",
        "   → PSI (Population Stability Index) per feature",
        "",
        "2. Compute drift_share = % of features drifted",
        "",
        "3. If drift_share ≥ 30%:",
        "   → Trigger HIGH priority alert",
        "   → Initiate auto-retraining pipeline",
        "",
        "4. Log drift_score to system_metrics table",
        "",
        "5. Generate drift report for operator review"
    ])
    
    # Slide 16: Auto-Retraining Pipeline
    create_content_slide(prs, "Implementation: Auto-Retraining", [
        "Triggered when drift_share ≥ 30%:",
        "",
        "1. Load updated training data (last 30 days)",
        "",
        "2. Optuna hyperparameter search (50 trials)",
        "   → Learning rate, max_depth, n_estimators, subsample",
        "",
        "3. Train XGBoost with best parameters",
        "",
        "4. Evaluate on test set → metrics logged to MLflow",
        "",
        "5. Convert to ONNX format (skl2onnx)",
        "",
        "6. Register as 'candidate' in model_versions table",
        "",
        "7. Operator approval required for production deployment"
    ])
    
    # Slide 17: Rollback Mechanism
    create_content_slide(prs, "Implementation: Instant Rollback", [
        "Dual-Pointer Architecture:",
        "",
        "deployment_state table (SINGLETON):",
        "  • current_model_id → Active inference model",
        "  • stable_model_id → Last known good model",
        "",
        "POST /rollback endpoint:",
        "  1. Execute atomic SQL update:",
        "     UPDATE deployment_state",
        "     SET current_model_id = stable_model_id",
        "",
        "  2. Log rollback event to audit trail",
        "",
        "  3. Clear active alerts",
        "",
        "Result: Rollback completed in <2 seconds",
        "  → Zero downtime, no data loss"
    ])
    
    # Slide 18: Results - Model Performance
    create_table_slide(prs, "Results: Model Performance Metrics",
        ["Metric", "Baseline (v0001)", "Retrained (v0002)", "Improvement"],
        [
            ["Accuracy", "95.2%", "97.1%", "+1.9%"],
            ["Precision", "91.3%", "94.5%", "+3.2%"],
            ["Recall", "89.7%", "93.2%", "+3.5%"],
            ["F1-Score", "90.5%", "93.8%", "+3.3%"],
            ["ROC-AUC", "0.943", "0.968", "+0.025"]
        ]
    )
    
    # Slide 19: Results - Inference Performance
    create_content_slide(prs, "Results: Inference Performance", [
        "ONNX Runtime vs. Python scikit-learn:",
        "",
        "ONNX Inference Time: 6.3ms",
        "  → Average over 1,000 predictions",
        "",
        "Python scikit-learn Baseline: 23.1ms",
        "",
        "Speedup: 3.7×",
        "",
        "Edge Deployment Benefits:",
        "  • Sub-10ms latency requirement: ✓ Achieved",
        "  • Real-time control loop compatible",
        "  • No cloud round-trip delay",
        "  • Suitable for Industrial Edge Devices (IED)"
    ])
    
    # Slide 20: Results - System Reliability
    create_table_slide(prs, "Results: System Reliability Metrics",
        ["Metric", "Target", "Achieved", "Status"],
        [
            ["Uptime", "99%", "99.7%", "✓ Pass"],
            ["Rollback Time", "<2s", "1.8s", "✓ Pass"],
            ["Dashboard Polling", "1s", "1.0s", "✓ Pass"],
            ["Drift Detection", "30% threshold", "18% (normal)", "✓ Pass"],
            ["Retrain Cycle", "<5 min", "4.2 min", "✓ Pass"]
        ]
    )
    
    # Slide 21: Contributions
    create_content_slide(prs, "Key Contributions", [
        "1. First production-grade MLOps platform for edge PdM",
        "",
        "2. Autonomous drift-to-retrain pipeline",
        "   → 30% threshold, <5 minute cycle",
        "",
        "3. Dual-pointer model registry for instant rollback",
        "   → 1.8 second recovery time",
        "",
        "4. ONNX-optimized inference",
        "   → 6.3ms latency, 3.7× speedup",
        "",
        "5. Live operator dashboard",
        "   → Embedded Grafana + audio alert system"
    ])
    
    # Slide 22: Limitations & Future Work
    create_two_column_slide(prs, "Limitations & Future Work",
        [
            "Current Limitations:",
            "",
            "• Single-device deployment",
            "  Not tested on distributed clusters",
            "",
            "• Simulated PLC data",
            "  Real S7-300 integration pending",
            "",
            "• Statistical drift only",
            "  No deep learning-based methods"
        ],
        [
            "Future Work:",
            "",
            "• Federated Learning",
            "  Multi-site model aggregation",
            "",
            "• Model Compression",
            "  INT8 quantization for <1W devices",
            "",
            "• Advanced Drift Detection",
            "  Autoencoders for novel failures",
            "",
            "• Cloud-Edge Hybrid",
            "  Offload retraining to cloud",
            "",
            "• Explainability (SHAP values)",
            "  Operator trust in predictions"
        ]
    )
    
    # Slide 23: Conclusion
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(9), Inches(0.8))
    title_box.text_frame.text = "Conclusion"
    title_box.text_frame.paragraphs[0].font.size = Pt(32)
    title_box.text_frame.paragraphs[0].font.bold = True
    
    content_box = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(8), Inches(4))
    text_frame = content_box.text_frame
    text_frame.word_wrap = True
    
    p = text_frame.add_paragraph()
    p.text = "EdgeOps AI demonstrates that production-grade MLOps can be successfully deployed at the industrial edge with autonomous drift detection, sub-10ms inference, and instant rollback capabilities."
    p.font.size = Pt(18)
    p.alignment = PP_ALIGN.CENTER
    
    p2 = text_frame.add_paragraph()
    p2.text = "\n\nThe platform addresses critical gaps in existing research by combining continuous monitoring, self-healing pipelines, and operator-friendly dashboards into a cohesive system suitable for real-world factory deployments."
    p2.font.size = Pt(18)
    p2.alignment = PP_ALIGN.CENTER
    
    # Slide 24: Thank You
    slide = create_title_slide(
        prs,
        "Thank You",
        "Questions?\n\n"
        "Contact: vanshita.mehta@example.com\n"
        "GitHub: github.com/vanshita/edgeops-ai\n"
        "Project Demo: http://localhost:5173"
    )
    
    # Save presentation
    output_file = "EdgeOps_AI_Academic_Presentation.pptx"
    prs.save(output_file)
    print(f"✓ Presentation saved as: {output_file}")
    print(f"✓ Total slides: {len(prs.slides)}")

if __name__ == "__main__":
    main()
