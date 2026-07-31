import os
import sys
from PIL import Image, ImageDraw
import reportlab
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage, KeepTogether
from reportlab.lib import colors
from reportlab.pdfgen import canvas

# ----------------- SECTION 1: PIL SCREENSHOT GENERATOR -----------------
def draw_mock_screenshot(filename, title, content_draw_func):
    # Base layout
    img = Image.new('RGB', (800, 500), color='#F8FAFC')
    draw = ImageDraw.Draw(img)
    
    # Left Sidebar (dark slate #0F172A)
    draw.rectangle([0, 0, 200, 500], fill='#0F172A')
    # Sidebar logo
    draw.rectangle([10, 15, 190, 50], fill='#1E293B')
    # Navigation items
    nav_items = ["Dashboard", "Predictive Analytics", "Reactive Inspection", "Root Cause Analytics", "AI Copilot", "Reports", "Datasets", "Model Performance", "Settings"]
    for i, item in enumerate(nav_items):
        y = 70 + i * 32
        if item == title:
            # Active item
            draw.rectangle([10, y, 190, y + 26], fill='#2563EB')
            
    # Top Bar (white #FFFFFF)
    draw.rectangle([200, 0, 800, 50], fill='#FFFFFF')
    draw.rectangle([220, 12, 450, 38], outline='#CBD5E1', width=1)
    draw.rectangle([470, 12, 650, 38], outline='#CBD5E1', width=1)
    draw.ellipse([750, 10, 780, 40], fill='#2563EB')
    
    # Call content specific draw
    content_draw_func(draw)
    
    img.save(filename)

def draw_dashboard(draw):
    # KPI Row (Quality, Health, Issues, Risk)
    draw.rectangle([220, 90, 340, 150], fill='#FFFFFF', outline='#E2E8F0')
    draw.rectangle([360, 90, 480, 150], fill='#FFFFFF', outline='#E2E8F0')
    draw.rectangle([500, 90, 620, 150], fill='#FFFFFF', outline='#E2E8F0')
    draw.rectangle([640, 90, 760, 150], fill='#FFFFFF', outline='#E2E8F0')
    
    # Left Column: Active Projects Risk Registry & Timeline
    draw.rectangle([220, 170, 480, 320], fill='#FFFFFF', outline='#E2E8F0')
    # Row list
    draw.rectangle([230, 210, 470, 250], fill='#F8FAFC', outline='#2563EB')
    draw.rectangle([230, 260, 470, 300], fill='#FFFFFF', outline='#E2E8F0')
    
    # Activity Timeline
    draw.rectangle([220, 335, 480, 485], fill='#FFFFFF', outline='#E2E8F0')
    draw.line([240, 380, 240, 460], fill='#E2E8F0', width=2)
    draw.ellipse([237, 390, 243, 396], fill='#2563EB')
    draw.ellipse([237, 430, 243, 436], fill='#2563EB')
    
    # Right Column: Context Scope & AI Quality Directives
    draw.rectangle([500, 170, 760, 260], fill='#FFFFFF', outline='#E2E8F0')
    draw.rectangle([500, 275, 760, 485], fill='#FFFFFF', outline='#E2E8F0')
    # Recommendation boxes
    draw.rectangle([515, 320, 745, 380], fill='#FEF2F2', outline='#EF4444')
    draw.rectangle([515, 395, 745, 455], fill='#FFFBEB', outline='#F59E0B')

def draw_predictor(draw):
    # Form input fields
    draw.rectangle([220, 90, 480, 380], fill='#FFFFFF', outline='#E2E8F0')
    draw.rectangle([500, 90, 760, 380], fill='#FFFFFF', outline='#E2E8F0')
    
    # Form fields represent
    for i in range(3):
        y = 140 + i * 70
        draw.line([240, y, 460, y], fill='#CBD5E1', width=3)
        draw.line([520, y, 740, y], fill='#CBD5E1', width=3)
        
    # Evaluate Button
    draw.rectangle([220, 400, 760, 440], fill='#2563EB')
    
    # Wizard action panel represent
    draw.rectangle([220, 450, 760, 490], fill='#F1F5F9', outline='#E2E8F0')

def draw_inspection(draw):
    # original image representation
    draw.rectangle([220, 90, 480, 330], fill='#CBD5E1', outline='#E2E8F0')
    # detection overlay
    draw.rectangle([500, 90, 760, 330], fill='#CBD5E1', outline='#E2E8F0')
    # draw defect box
    draw.rectangle([550, 130, 670, 240], outline='#EF4444', width=3)
    draw.rectangle([550, 110, 630, 130], fill='#EF4444')
    
    # defect registry grid
    draw.rectangle([220, 345, 760, 485], fill='#FFFFFF', outline='#E2E8F0')

def draw_root_cause(draw):
    # Sliders row
    draw.rectangle([220, 80, 760, 190], fill='#FFFFFF', outline='#E2E8F0')
    
    # Executive Briefing Card (What Happened?, Why?, Risk Reduction text)
    draw.rectangle([220, 205, 760, 370], fill='#FFFFFF', outline='#E2E8F0')
    draw.rectangle([540, 220, 740, 300], fill='#EFF6FF', outline='#3B82F6') # Reduction box
    
    # Recommended actions checklist
    draw.ellipse([240, 320, 248, 328], fill='#10B981')
    draw.ellipse([240, 345, 248, 353], fill='#10B981')
    
    # Advanced Diagnostics Expander (Collapsed represent)
    draw.rectangle([220, 385, 760, 485], fill='#F1F5F9', outline='#E2E8F0')
    draw.text((240, 400), "Show Secondary Technical ML Diagnostics (SHAP Graphs)", fill='#1E293B')

def draw_copilot(draw):
    # Context summary block
    draw.rectangle([220, 90, 760, 130], fill='#EFF6FF', outline='#3B82F6')
    
    # Prompt Chips
    draw.rectangle([220, 145, 380, 185], fill='#FFFFFF', outline='#CBD5E1')
    draw.rectangle([400, 145, 560, 185], fill='#FFFFFF', outline='#CBD5E1')
    draw.rectangle([580, 145, 760, 185], fill='#FFFFFF', outline='#CBD5E1')
    
    # Chat panel
    draw.rectangle([220, 200, 760, 430], fill='#FFFFFF', outline='#E2E8F0')
    # Bubble left
    draw.rectangle([240, 220, 540, 270], fill='#F1F5F9')
    # Bubble right
    draw.rectangle([440, 290, 740, 345], fill='#EFF6FF')
    
    # Input box
    draw.rectangle([220, 445, 760, 485], fill='#FFFFFF', outline='#CBD5E1')

def draw_reports(draw):
    draw.rectangle([220, 100, 760, 210], fill='#FFFFFF', outline='#E2E8F0')
    draw.rectangle([220, 230, 760, 480], fill='#FFFFFF', outline='#E2E8F0')

def draw_datasets(draw):
    draw.rectangle([220, 100, 760, 200], fill='#FFFFFF', outline='#E2E8F0')
    draw.rectangle([220, 220, 760, 480], fill='#FFFFFF', outline='#E2E8F0')

def draw_model_perf(draw):
    draw.rectangle([220, 90, 340, 140], fill='#FFFFFF', outline='#E2E8F0')
    draw.rectangle([360, 90, 480, 140], fill='#FFFFFF', outline='#E2E8F0')
    draw.rectangle([500, 90, 620, 140], fill='#FFFFFF', outline='#E2E8F0')
    draw.rectangle([640, 90, 760, 140], fill='#FFFFFF', outline='#E2E8F0')
    
    # Charts
    draw.rectangle([220, 160, 480, 480], fill='#FFFFFF', outline='#E2E8F0')
    draw.rectangle([500, 160, 760, 480], fill='#FFFFFF', outline='#E2E8F0')

def draw_settings(draw):
    draw.rectangle([220, 100, 760, 260], fill='#FFFFFF', outline='#E2E8F0')
    draw.rectangle([220, 280, 760, 480], fill='#FFFFFF', outline='#E2E8F0')

def generate_screenshots():
    print("Generating mock screenshot assets using PIL...")
    draw_mock_screenshot("shot_dashboard.png", "Dashboard", draw_dashboard)
    draw_mock_screenshot("shot_predictor.png", "Predictive Analytics", draw_predictor)
    draw_mock_screenshot("shot_inspection.png", "Reactive Inspection", draw_inspection)
    draw_mock_screenshot("shot_root_cause.png", "Root Cause Analytics", draw_root_cause)
    draw_mock_screenshot("shot_copilot.png", "AI Copilot", draw_copilot)
    draw_mock_screenshot("shot_reports.png", "Reports", draw_reports)
    draw_mock_screenshot("shot_datasets.png", "Datasets", draw_datasets)
    draw_mock_screenshot("shot_model_perf.png", "Model Performance", draw_model_perf)
    draw_mock_screenshot("shot_settings.png", "Settings", draw_settings)
    print("Screenshots generated successfully!")

# ----------------- SECTION 2: REPORTLAB PDF BUILDER -----------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        if self._pageNumber == 1:
            return  # Suppress page numbers on cover page
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor('#475569'))
        
        # Header
        self.drawString(54, 750, "FutureStrive Construction Intelligence Platform — Functional Documentation")
        self.setStrokeColor(colors.HexColor('#CBD5E1'))
        self.setLineWidth(0.5)
        self.line(54, 742, 558, 742)
        
        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 42, page_text)
        self.drawString(54, 42, "CONFIDENTIAL - FUTURESTRIVE SYSTEMS")
        self.line(54, 52, 558, 52)
        
        self.restoreState()

def build_pdf(filename="Construction Intelligence Platform – Functional Documentation.pdf"):
    print(f"Building functional PDF document: {filename}...")
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=28,
        leading=34,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=13,
        leading=18,
        textColor=colors.HexColor('#475569'),
        spaceAfter=40
    )
    
    meta_style = ParagraphStyle(
        'CoverMeta',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#2563EB'),
        spaceAfter=10
    )
    
    h1_style = ParagraphStyle(
        'DocH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=22,
        spaceAfter=12,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=15,
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=10
    )
    
    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155'),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=5
    )
    
    story = []
    
    # ----------------- COVER PAGE -----------------
    story.append(Spacer(1, 100))
    story.append(Paragraph("FUTURESTRIVE SYSTEMS", meta_style))
    story.append(Paragraph("Construction Intelligence Platform", title_style))
    story.append(Paragraph("Functional Specification & System Documentation", subtitle_style))
    story.append(Spacer(1, 200))
    
    meta_table_data = [
        [Paragraph("<b>Document Title:</b>", body_style), Paragraph("Construction Intelligence Platform – Functional Documentation", body_style)],
        [Paragraph("<b>Author:</b>", body_style), Paragraph("FutureStrive Lead Design Team", body_style)],
        [Paragraph("<b>Target Audience:</b>", body_style), Paragraph("Stakeholders, Quality Engineers, Development Team", body_style)],
        [Paragraph("<b>Version:</b>", body_style), Paragraph("v3.0 — Reactive Crack Detection Release", body_style)],
        [Paragraph("<b>Release Date:</b>", body_style), Paragraph("July 31, 2026", body_style)]
    ]
    t = Table(meta_table_data, colWidths=[120, 380])
    t.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(PageBreak())
    
    # ----------------- SECTION 1: PROJECT OVERVIEW -----------------
    story.append(Paragraph("1. Project Overview", h1_style))
    story.append(Paragraph(
        "The FutureStrive Construction Intelligence Platform is an enterprise-grade SaaS web application "
        "designed to optimize concrete quality control, risk prediction, and compliance workflows at construction sites. "
        "Historically, construction quality assurance has relied on manual, retrospective tests (e.g. concrete cube compression testing "
        "at 28 days), which identify defects only after significant structural elements are poured. FutureStrive addresses this "
        "problem by shifting quality control to the pre-pour phase, leveraging cascading machine learning models and local Small "
        "Language Models (SLMs). The platform enables site engineers, quality managers, and developers to predict curing defect risks, "
        "analyze surface cracks with computer vision, explain model outputs via SHAP (Shapley Additive exPlanations), "
        "and get corrective instructions through a RAG-enabled chatbot.",
        body_style
    ))
    
    # ----------------- SECTION 2: SYSTEM ARCHITECTURE -----------------
    story.append(Paragraph("2. System Architecture", h1_style))
    story.append(Paragraph(
        "The platform utilizes a modern 4-tier decoupled software architecture designed for offline-first resilience on remote construction sites:",
        body_style
    ))
    story.append(Paragraph("• <b>Ingestion Layer:</b> Manages site inputs from manuals forms, Excel/CSV files, and camera streams.", bullet_style))
    story.append(Paragraph("• <b>Inference Layer:</b> Executes lightweight local XGBoost cascading classifiers for occurrence, type, severity, and cause.", bullet_style))
    story.append(Paragraph("• <b>Explainability Layer:</b> Computes real-time SHAP values utilizing tree explainer background distributions to reveal risk factors.", bullet_style))
    story.append(Paragraph("• <b>Copilot Layer:</b> A local Qwen 0.5B parameter instruction model utilizing TF-IDF indexing for Retrieval-Augmented Generation (RAG).", bullet_style))
    
    # ----------------- SECTION 3: APPLICATION WORKFLOW -----------------
    story.append(Paragraph("3. Application Workflow", h1_style))
    story.append(Paragraph(
        "The platform's primary workflow follows a sequential, closed-loop engineering process:\n"
        "1. <b>Design & Input:</b> The engineer inputs mix design details and ambient environment parameters.\n"
        "2. <b>Prediction:</b> The XGBoost model calculates the crack probability, flagging risks using Green/Amber/Red indicators.\n"
        "3. <b>SHAP Analytics:</b> The explainer computes features' influence on the risk level, presenting plain-English cards.\n"
        "4. <b>What-If Simulator:</b> The user modifies parameters to simulate mitigation techniques.\n"
        "5. <b>SLM & Action:</b> The Copilot generates a step-by-step remediation plan based on specification codes.\n"
        "6. <b>Reporting:</b> PDF/Excel sheets are compiled for compliance sign-off.",
        body_style
    ))
    
    # ----------------- SECTION 4: PAGE-BY-PAGE EXPLANATION -----------------
    story.append(Paragraph("4. Page-by-Page Explanation", h1_style))
    pages = [
        ("Dashboard", "Provides project health scores, recent quality alerts, risk trends, and model execution performance summaries."),
        ("Predictive Analytics", "Supports single-pour simulators and batch file uploads to run predictions on dataset logs."),
        ("Reactive Inspection", "Applies AI bounding boxes on uploaded concrete photos to identify hairline/wide cracks and honeycombs."),
        ("Root Cause Analytics", "Explains prediction drivers using SHAP waterfall charts, summary plots, and a What-If sensitivity calculator."),
        ("AI Copilot", "Provides a ChatGPT-style conversational assistant loaded with concrete specifications manuals via RAG."),
        ("Reports", "Compiles and downloads Quality Executive Summaries, Root Cause analyses, and Inspection Logs in CSV format."),
        ("Datasets", "Searchable, filterable, and paginated registry for all pour records, logging counts, and validation checks."),
        ("Model Performance", "Displays evaluation matrices (Precision, Recall, ROC Curves, Confusion Matrices) for XGBoost models."),
        ("Settings", "Configures red/amber risk alert thresholds and toggles background calculations.")
    ]
    for p_title, p_desc in pages:
        story.append(Paragraph(f"<b>{p_title} Page:</b> {p_desc}", bullet_style))
        
    # ----------------- SECTION 5: USER JOURNEY -----------------
    story.append(Paragraph("5. User Journey", h1_style))
    story.append(Paragraph(
        "<b>Scenario: Curing quality warning on site</b><br/>"
        "1. Mithran (Quality Engineer) signs in and selects Project PJ-2026-MUM-01.<br/>"
        "2. He enters the **Predictive Analytics** page and simulates concrete column pour parameters. He clicks 'Evaluate'.<br/>"
        "3. The platform returns a **93.1% Crack Probability (Red Flag)** because curing duration is set to 8 days.<br/>"
        "4. He clicks 'View Root Cause' and is redirected to **Root Cause Analytics** page, seeing SHAP cards and a waterfall chart showing curing duration as the primary contributor (+0.24 risk factor).<br/>"
        "5. He uses the **What-If Simulator** slider to increase curing days to 14. The risk score drops to **4.5% (Green Flag)**.<br/>"
        "6. He opens the **AI Copilot** tab and asks for a remediation plan, which suggests wet burlap wrapping.<br/>"
        "7. He generates and downloads the **Defect Inspection Log** from the **Reports** tab.",
        body_style
    ))
    
    # ----------------- SECTION 6: FUNCTIONALITY OF EVERY FEATURE -----------------
    story.append(Paragraph("6. Functionality of Every Feature", h1_style))
    story.append(Paragraph(
        "The platform contains fully operational, interactive features including:\n"
        "• **Project Switcher:** Interconnects dashboards, alerts, and dataset tables dynamically.\n"
        "• **Interactive Sliders:** Triggers instant mathematical updates of probabilities and SHAP vectors in real-time.\n"
        "• **File Upload Parsing:** Accepts files, runs pandas validation checks, and calculates batch inference outputs.\n"
        "• **PIL Image Overlays:** Dynamically draws coordinates on uploaded concrete images to highlight cracks and honeycombs.",
        body_style
    ))
    
    # ----------------- SECTION 7: INPUT REQUIREMENTS -----------------
    story.append(Paragraph("7. Input Requirements", h1_style))
    st_table_data = [
        ["Variable Name", "Unit/Format", "Ranges / Options", "Validation Rule"],
        ["Concrete Grade", "Nominal", "M25, M30, M35, M40", "Required"],
        ["Designed W/C Ratio", "Decimal", "0.30 to 0.50", "Must be <= actual W/C"],
        ["Actual W/C Ratio", "Decimal", "0.30 to 0.60", "Exceeding 0.45 triggers warning"],
        ["Curing Duration", "Days", "1 to 28 days", "Fails spec if < 14 days"],
        ["Checklist Compliance", "Percentage", "0% to 100%", "Fails threshold if < 85%"],
        ["Wind Exposure", "km/h", "0 to 50 km/h", "Exceeding 15 km/h triggers warning"]
    ]
    st_t = Table(st_table_data, colWidths=[130, 90, 150, 130])
    st_t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
        ('FONTSIZE', (0,0), (-1,-1), 8),
    ]))
    story.append(st_t)
    story.append(Spacer(1, 10))
    
    # ----------------- SECTION 8: PROCESSING LOGIC -----------------
    story.append(Paragraph("8. Processing Logic", h1_style))
    story.append(Paragraph(
        "Probability calculations in the simulator and What-If panels are modeled using mathematical weights "
        "derived from local XGBoost training parameters. The core formula operates as follows:\n"
        "$$P(crack) = Baseline + \\sum_{i} W_i \\cdot \\Delta F_i$$\n"
        "Where W/C ratio deviations add 150% of the difference value, short curing duration adds 4% per day missing below the 14-day mark, "
        "and QC compliance failures below 85% scale up the probability by 35% of the shortfall. This ensures the output probability "
        "is consistent, realistic, and matches the behavior of the trained XGBoost classification models.",
        body_style
    ))
    
    # ----------------- SECTION 9: EXPECTED OUTPUTS -----------------
    story.append(Paragraph("9. Expected Outputs", h1_style))
    story.append(Paragraph(
        "• **Inference outputs:** Risk level text flags (Red, Amber, Green), crack types (Shrinkage, Shear, Flexural, Settlement), and likely root causes.\n"
        "• **SHAP outputs:** Waterfall and Feature Importance plots showing positive/negative direction values.\n"
        "• **RAG assistant outputs:** Text citations referencing specifications manuals.\n"
        "• **Report outputs:** CSV tables containing defect records, confidence rates, and timestamps.",
        body_style
    ))
    
    # ----------------- SECTION 10: DUMMY DATA USED -----------------
    story.append(Paragraph("10. Dummy Data Used", h1_style))
    story.append(Paragraph(
        "We utilize pre-populated, domain-specific lists representing standard project logs:\n"
        "1. **Datasets List:** Files such as `mumbai_tower_a_pour_logs.csv` and `delhi_metro_slab_curing.xlsx` containing record counts.\n"
        "2. **Defect Types:** Hairline cracks, Wide cracks, Honeycombing, Corrosion, Spalling.\n"
        "3. **Remediation Guides:** Injecting low-viscosity epoxy resin, removing concrete, burlap wetting.",
        body_style
    ))
    
    # ----------------- SECTION 11: AI WORKFLOW -----------------
    story.append(Paragraph("11. AI Workflow (Prediction → SHAP → Recommendations)", h1_style))
    story.append(Paragraph(
        "The model cascade starts with a binary prediction (Crack Occurrence) using XGBoost. "
        "If a crack is predicted (probability >= 25%), three downstream models execute in parallel: "
        "1) Crack Type Classifier, 2) Crack Severity Classifier, 3) Root Cause Classifier. "
        "Simultaneously, the SHAP tree explainer generates the waterfall contributions which are parsed into prioritized "
        "site recommendations (e.g. rejecting concrete batches, applying curing compounds) to optimize concrete performance.",
        body_style
    ))
    
    # ----------------- SECTION 12: REACTIVE INSPECTION WORKFLOW -----------------
    story.append(Paragraph("12. Reactive Crack Detection Workflow", h1_style))
    story.append(Paragraph(
        "The reactive inspection track allows site engineers to upload a photograph of a concrete surface directly "
        "inside the AI Assistant. The platform passes the image through a pluggable segmentation/detection model wrapper "
        "(crack_detector.py) that returns structured features: crack type, severity, area fraction, estimated width, "
        "bounding boxes, and an annotated overlay image. The AI assistant then builds an IS-code-referenced engineering "
        "report via Qwen LLM or a rule-based fallback."
        "<br/><br/>"
        "<b>Detection pipeline steps:</b><br/>"
        "1. <b>Image Upload:</b> User expands the Upload Crack Photo panel in the AI Assistant and uploads a JPG/PNG.<br/>"
        "2. <b>Model Inference:</b> crack_detector.run_crack_detection(image) is called. In production this runs the "
        "trained segmentation model. In demo mode (USE_STUB=True) synthetic outputs are seeded from image pixel data "
        "for repeatable UI testing.<br/>"
        "3. <b>Annotated Overlay:</b> Bounding boxes with severity-coded colours are drawn on the image "
        "(green=Minor, amber=Moderate, red=Severe, purple=Critical).<br/>"
        "4. <b>LLM Report:</b> qwen_crack_image_report() injects the detection dict into Qwen context and generates "
        "a structured report: Executive Summary, Root Cause Analysis, Risk to Structure, Remediation Recommendations "
        "(IS 456:2000, IS 13311, IS 516 referenced), and Urgency Level.<br/>"
        "5. <b>Rule-based Fallback:</b> _fallback_image_report() produces the same structured report without any LLM call.<br/>"
        "6. <b>Chat Integration:</b> Report appended to chat thread. Follow-up questions re-route to image_crack_analysis intent.",
        body_style
    ))
    
    # ----------------- SECTION 13: AI COPILOT WORKFLOW -----------------
    story.append(Paragraph("13. AI Copilot Workflow", h1_style))
    story.append(Paragraph(
        "The AI Assistant handles user natural language prompts through a 7-way intent router:"
        "<br/><b>1. greeting</b> — Returns a welcome message listing capabilities."
        "<br/><b>2. image_crack_analysis</b> — Reactive track: fires when a photo is pending, routes to crack_detector + Qwen report."
        "<br/><b>3. crack_prediction</b> — Rule-based IS 456 flagging on live pour parameters (no LLM call)."
        "<br/><b>4. defect_prediction</b> — Rule-based QC/SPI analysis on live defect parameters (no LLM call)."
        "<br/><b>5. analytical</b> — Pandas filter on active session parameters for numerical comparison queries."
        "<br/><b>6. knowledge</b> — FAISS retrieval (top-4 chunks) followed by Qwen 2.5-0.5B-Instruct generation."
        "<br/><b>7. off_topic</b> — Politely declines non-construction queries."
        "<br/><br/>For knowledge queries, Qwen receives: active module parameters, IS code deviation flags, "
        "last 3 prediction records, and FAISS knowledge chunks from the IS 456, CPWD, and IS 13311 knowledge base.",
        body_style
    ))
    
    # ----------------- SECTION 14: REPORTS WORKFLOW -----------------
    story.append(Paragraph("14. Reports Generation Workflow", h1_style))
    story.append(Paragraph(
        "The reports generation module compiles datasets (Executive Summaries, SHAP root cause rows) on demand. "
        "The compiled dataframe is converted into a binary CSV buffer and made available via a download button.",
        body_style
    ))
    
    # ----------------- SECTION 15: CURRENT IMPLEMENTED FEATURES -----------------
    story.append(Paragraph("15. Current Implemented Features", h1_style))
    story.append(Paragraph("• <b>Multi-page Navigation:</b> Functional left sidebar radiogroup.", bullet_style))
    story.append(Paragraph("• <b>Crack Intelligence (Predictive):</b> 21-input pour parameter form with XGBoost prediction + inline SHAP driver chart + IS 456 corrective recommendations.", bullet_style))
    story.append(Paragraph("• <b>Defect Volume Intelligence (Predictive):</b> 26-input form with XGBoost prediction + driver analysis + improvement recommendations.", bullet_style))
    story.append(Paragraph("• <b>Reactive Crack Detection:</b> Photo upload in AI Assistant → crack_detector.py segmentation wrapper → annotated overlay → IS-code-referenced engineering report in chat thread.", bullet_style))
    story.append(Paragraph("• <b>Prediction History:</b> Tabbed trend charts for both predictive modules with Altair line/bar visualisations.", bullet_style))
    story.append(Paragraph("• <b>Unified AI Assistant:</b> 7-way intent router covering greeting, image analysis, crack/defect prediction, analytical, knowledge (FAISS+Qwen), and off-topic.", bullet_style))
    story.append(Paragraph("• <b>Rule-based LLM Fallback:</b> Full IS-code-referenced engineering report generated without Qwen when the LLM is unavailable.", bullet_style))

    # ----------------- SECTION 16: REMAINING PLACEHOLDER FEATURES -----------------
    story.append(Paragraph("16. Remaining Placeholder / Planned Features", h1_style))
    story.append(Paragraph("• <b>Real-time databases:</b> Active project and prediction histories use RAM-based session state. PostgreSQL integration planned.", bullet_style))
    story.append(Paragraph("• <b>Production Crack Segmentation Model:</b> crack_detector.py currently runs in stub/demo mode (USE_STUB=True). The trained segmentation model will be connected by setting USE_STUB=False and implementing _run_real_model().", bullet_style))
    story.append(Paragraph("• <b>Crack Width Estimation:</b> Estimated width in mm is synthetic in demo mode. The real model will provide pixel-accurate width derived from mask geometry.", bullet_style))
    
    # ----------------- SECTION 17: FUTURE BACKEND PLAN -----------------
    story.append(Paragraph("17. Future Backend Integration Plan", h1_style))
    story.append(Paragraph(
        "The product roadmap outlines replacing session states with PostgreSQL database schemas, deploying RESTful APIs via FastAPI, "
        "and migrating computer vision models to GPU cloud nodes to automate live video streams.",
        body_style
    ))
    
    story.append(PageBreak())
    
    # ----------------- SECTION 18: SCREENSHOTS & ANNOTATIONS -----------------
    story.append(Paragraph("18. Page Screenshots & Annotations", h1_style))
    
    screen_pages = [
        ("Dashboard", "shot_dashboard.png", "Displays quality trends, active alerts, and model performance metrics."),
        ("Predictive Analytics", "shot_predictor.png", "Form widgets and batch file uploader for concrete pour simulations."),
        ("Reactive Inspection", "shot_inspection.png", "AI bounding box defect overlays on concrete photos."),
        ("Root Cause Analytics", "shot_root_cause.png", "SHAP waterfalls, importance plots, and What-If sliders."),
        ("AI Copilot", "shot_copilot.png", "RAG-powered conversational Small Language Model interface."),
        ("Reports", "shot_reports.png", "Compile and download project report sheets in CSV format."),
        ("Datasets", "shot_datasets.png", "Paginated, filterable database registry for uploaded files."),
        ("Model Performance", "shot_model_perf.png", "Evaluation metrics, ROC curves, and confusion matrices."),
        ("Settings", "shot_settings.png", "Alert threshold and background calculation preferences.")
    ]
    
    for label, file_path, desc in screen_pages:
        if os.path.exists(file_path):
            # Scale down image for ReportLab page fit
            story.append(Paragraph(f"<b>Page: {label}</b>", h2_style))
            story.append(Paragraph(desc, body_style))
            story.append(RLImage(file_path, width=400, height=250))
            story.append(Spacer(1, 15))
            
    # Compile
    doc.build(story, canvasmaker=NumberedCanvas)
    print("PDF documentation compiled successfully!")

if __name__ == "__main__":
    generate_screenshots()
    build_pdf()
