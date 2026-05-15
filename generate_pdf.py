from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

# ── PALETTE
BLACK   = colors.HexColor("#0D0D0D")
DARK    = colors.HexColor("#1A1A1A")
MED     = colors.HexColor("#4A4A4A")
LIGHT   = colors.HexColor("#888888")
XLIGHT  = colors.HexColor("#BBBBBB")
RULE    = colors.HexColor("#E0E0E0")
BG      = colors.HexColor("#F8F8F8")
BG2     = colors.HexColor("#F0F0F0")
WHITE   = colors.white
ACCENT  = colors.HexColor("#1A3C5E")   # deep navy — one accent only
ACCENT2 = colors.HexColor("#2E6CA4")   # mid blue

W, H   = A4
MARGIN = 19 * mm
AW     = W - 2 * MARGIN   # available width

# ── STYLE FACTORY
def mk(name, size=9, leading=14, font="Helvetica", color=DARK, align=TA_LEFT, **kw):
    return ParagraphStyle(name, fontSize=size, leading=leading,
                          fontName=font, textColor=color,
                          alignment=align, **kw)

S = {
    "h_title":   mk("h_title",  size=26, leading=32, font="Helvetica-Bold",
                    color=BLACK, align=TA_CENTER),
    "h_sub":     mk("h_sub",    size=11, leading=16, color=ACCENT2, align=TA_CENTER),
    "h_meta":    mk("h_meta",   size=8.5,leading=13, color=LIGHT,  align=TA_CENTER,
                    fontName="Helvetica-Oblique"),
    "h_pitch":   mk("h_pitch",  size=9.5,leading=15, color=MED,    align=TA_CENTER),
    "sec_lbl":   mk("sec_lbl",  size=7.5,leading=11, font="Helvetica-Bold",
                    color=LIGHT, spaceAfter=5),
    "phase_hd":  mk("phase_hd", size=9,  leading=13, font="Helvetica-Bold", color=ACCENT),
    "mod_num":   mk("mod_num",  size=11, leading=14, font="Helvetica-Bold",
                    color=ACCENT2, align=TA_CENTER),
    "mod_title": mk("mod_title",size=10, leading=14, font="Helvetica-Bold", color=BLACK),
    "mod_tag":   mk("mod_tag",  size=7,  leading=10, font="Helvetica-Bold",
                    color=LIGHT, align=TA_CENTER),
    "mod_desc":  mk("mod_desc", size=8.5,leading=13, color=MED),
    "mod_note":  mk("mod_note", size=7.5,leading=11, color=XLIGHT,
                    fontName="Helvetica-Oblique"),
    "bullet":    mk("bullet",   size=8.5,leading=13, color=MED, leftIndent=10),
    "role_hd":   mk("role_hd",  size=9,  leading=13, font="Helvetica-Bold", color=BLACK),
    "role_bd":   mk("role_bd",  size=8,  leading=12, color=MED),
    "contrib_hd":mk("contrib_hd",size=8.5,leading=13,font="Helvetica-Bold",color=ACCENT),
    "contrib_bd":mk("contrib_bd",size=8, leading=12, color=MED),
    "tk":        mk("tk",       size=8.5,leading=13, font="Helvetica-Bold", color=BLACK),
    "tv":        mk("tv",       size=8.5,leading=13, color=MED),
    "footer":    mk("footer",   size=7,  leading=10, color=XLIGHT, align=TA_CENTER,
                    fontName="Helvetica-Oblique"),
    "disc":      mk("disc",     size=7.5,leading=11, color=LIGHT, align=TA_CENTER,
                    fontName="Helvetica-Oblique"),
}

def hr(c=RULE, t=0.5, before=4, after=6):
    return HRFlowable(width="100%", thickness=t, color=c,
                      spaceBefore=before, spaceAfter=after)
def sp(h=6): return Spacer(1, h)
def B(txt, style="mod_desc"):
    return Paragraph(f"&#8226;&#160;&#160;{txt}", S[style])

# ── MODULE DATA  (15 modules total)
PHASES = [
    {
        "label": "PHASE 1 — CORE MVP",
        "note":  "Minimum viable product — must be delivered",
        "modules": [
            {
                "n": 1,
                "title": "User Authentication & Role Management",
                "tag": "CORE",
                "scope": "FYP Core",
                "desc": (
                    "Three distinct user roles — Patient, Doctor, and Admin — each with a completely "
                    "separate interface, dashboard, and permission set. Patients and doctors self-register; "
                    "admin accounts are provisioned internally by WebClinic. Role-based access control "
                    "is enforced at every API endpoint across both the mobile app and the web admin portal."
                ),
                "bullets": [
                    "Patients control which doctors can view their reports — sharing is always opt-in",
                    "Doctors cannot modify any medical data — read and annotate only",
                    "Admins manage operations but have no access to raw patient medical records",
                    "Secure session management with token-based authentication (JWT)",
                ],
                "note": None,
            },
            {
                "n": 2,
                "title": "Patient Profile & Onboarding",
                "tag": "CORE",
                "scope": "FYP Core",
                "desc": (
                    "A structured onboarding flow that captures the patient's full health context in one session. "
                    "This data is used to personalise ML model reference thresholds — ensuring risk predictions "
                    "are calibrated to the individual rather than a generic population average."
                ),
                "bullets": [
                    "Demographics: age, gender, height, weight, city, ethnicity",
                    "Pre-existing conditions: diabetes, hypertension, hepatitis, heart disease",
                    "Lifestyle context: sleep hours, physical activity level, diet type, smoking status, stress level",
                    "Profile can be updated at any time — changes trigger a re-evaluation of risk thresholds",
                ],
                "note": None,
            },
            {
                "n": 3,
                "title": "Blood Report Upload & Smart Extraction",
                "tag": "CORE",
                "scope": "FYP Core",
                "desc": (
                    "The patient uploads a photo or PDF of their lab report — from any Pakistani lab. "
                    "An automated extraction pipeline reads and parses every biomarker: test name, result "
                    "value, unit, and the lab's own reference range. The system handles varied and "
                    "non-standardised formats from different labs, standardises units, auto-calculates "
                    "FIB-4 and eGFR from extracted values, and clearly flags any out-of-range findings. "
                    "The patient reviews extracted data and confirms before it is saved. Manual entry "
                    "is always available as a fallback."
                ),
                "bullets": [
                    "Handles non-standardised Pakistani lab formats — each lab formats reports differently",
                    "Auto-calculation of derived clinical scores: FIB-4 (liver fibrosis), eGFR (kidney function)",
                    "Out-of-range biomarkers flagged immediately — Critical / Borderline / Normal",
                    "Patient confirmation step before data is finalised — human remains in the loop",
                ],
                "note": "Academic note: Pakistani laboratories show substantial variability in reference "
                        "intervals — the system stores each lab's own reference range alongside the extracted "
                        "value rather than applying a single global threshold.",
            },
            {
                "n": 4,
                "title": "ML Risk Prediction Engine",
                "tag": "AI / ML",
                "scope": "FYP Core",
                "desc": (
                    "The core intelligence of BloodMAPP. A Python-based machine learning model "
                    "takes the patient's blood biomarkers and lifestyle context as combined input and "
                    "outputs three simultaneous risk scores: Diabetes, Cardiovascular Disease, and NAFLD "
                    "(Non-Alcoholic Fatty Liver Disease). The model selection process evaluates multiple "
                    "algorithms (Logistic Regression, Random Forest, XGBoost, LightGBM) and selects the "
                    "best performer via cross-validated benchmarking. A supervisor layer identifies when "
                    "two or more scores are jointly elevated — flagging a Metabolic Syndrome pattern "
                    "that individual models would miss. Plain-language explanations of each key risk "
                    "driver are generated automatically in both Urdu and English."
                ),
                "bullets": [
                    "Simultaneous three-disease prediction from a single blood panel — novel at consumer app level",
                    "Model selection via performance benchmarking — XGBoost, Random Forest, LightGBM evaluated",
                    "Metabolic Syndrome supervisor layer — detects co-elevation patterns across diseases",
                    "SHAP-based feature importance — identifies which biomarkers are driving each risk score",
                    "Validated against clinical datasets; limitations transparently documented in methodology",
                ],
                "note": "Ethical note: Model is trained on available clinical datasets with Pakistani "
                        "population validation planned via WebClinic data. Known bias limitations are "
                        "documented; Module 15 (Feedback Loop) provides the mechanism for ongoing correction.",
            },
            {
                "n": 5,
                "title": "Health Dashboard & Biomarker Visualisation",
                "tag": "CORE",
                "scope": "FYP Core",
                "desc": (
                    "The patient's central home screen — the place they land every time they open the app. "
                    "Presents the full picture of their health in plain language, avoiding medical jargon. "
                    "Designed for low-health-literacy users in Pakistan."
                ),
                "bullets": [
                    "Traffic-light biomarker grid: every uploaded marker shown as Normal / Borderline / Critical",
                    "Overall Metabolic Health Score (0 to 100) — computed from all risk factors combined",
                    "Three disease risk cards: Diabetes, Cardiovascular Disease, NAFLD — each with score and plain explanation",
                    "Trend charts activate when the patient has uploaded two or more reports over time",
                    "Full bilingual support — every label, explanation, and status shown in Urdu and English",
                ],
                "note": None,
            },
            {
                "n": 6,
                "title": "Passive Health & Activity Monitoring",
                "tag": "SENSOR / ML",
                "scope": "FYP Core",
                "desc": (
                    "BloodMAPP replaces subjective self-reported lifestyle data with objective, "
                    "automatically collected sensor data — significantly improving ML prediction accuracy. "
                    "A three-tier architecture ensures the feature works for every user regardless of "
                    "what hardware they own, from a basic Android phone to a CGM device."
                ),
                "bullets": [
                    "Tier 1 — All users: Phone motion chip tracks daily steps and activity level passively. "
                    "Uses dedicated low-power hardware (not the microphone, not the main processor). "
                    "Zero battery impact. Implemented via Flutter pedometer package.",
                    "Tier 2 — Smartwatch users: If the patient has a Samsung, Apple Watch, or any "
                    "WearOS device, BloodMAPP reads heart rate, sleep cycles (light/deep/REM), and "
                    "activity minutes via Android Health Connect or Apple HealthKit.",
                    "Tier 3 — CGM users (Future Phase): Patients who own a FreeStyle Libre or "
                    "compatible continuous glucose monitor can sync real-time glucose data directly "
                    "into BloodMAPP via Bluetooth. CGM sensors are commercially available in Pakistan "
                    "(FreeStyle Libre 2 — Rs. 18,500 per 14-day sensor).",
                ],
                "note": "Privacy note: All sensor access requires explicit user permission via a single "
                        "on-screen consent popup (iOS HealthKit / Android Health Connect standard). "
                        "No data is collected silently. Users can revoke access at any time from device settings.",
            },
        ]
    },
    {
        "label": "PHASE 2 — INTELLIGENCE & COLLABORATION",
        "note":  "Core intelligence, doctor workflow, and communications",
        "modules": [
            {
                "n": 7,
                "title": "Personalised Health Recommendations Engine",
                "tag": "AI",
                "scope": "FYP Core",
                "desc": (
                    "Generates actionable, specific health guidance derived from the combination of "
                    "elevated biomarkers, lifestyle factors, and risk scores — not generic wellness tips. "
                    "Free and premium tiers provide different levels of depth."
                ),
                "bullets": [
                    "Free tier: General guidance linked to specific abnormal markers (e.g. high LDL → dietary fat reduction tips)",
                    "Premium tier: Detailed, cross-referenced action plan — e.g. high ALT + sedentary lifestyle "
                    "generates liver-specific dietary suggestions; elevated fasting glucose + poor sleep "
                    "generates targeted insulin-sensitivity interventions",
                    "Recommendations generated in Urdu and English — accessible to the full Pakistani user base",
                    "Sensor data from Module 6 feeds directly into recommendations — e.g. low step count "
                    "produces specific, quantified activity targets",
                ],
                "note": None,
            },
            {
                "n": 8,
                "title": "Alert & Retest Reminder System",
                "tag": "CORE",
                "scope": "FYP Core",
                "desc": (
                    "Proactive, automated communication that keeps patients engaged with their health "
                    "between report uploads. Abnormal findings trigger immediate alerts; risk-based "
                    "reminder schedules keep patients on track for appropriate retesting."
                ),
                "bullets": [
                    "Critical biomarkers: immediate in-app alert + push notification + email",
                    "Risk-based retest reminders: Critical findings → 4 weeks, Borderline → 3 months, Normal → 6 months",
                    "Doctor activity notifications: patient is notified when a doctor adds a note or sends a message",
                    "Push notifications via mobile (Android/iOS); email fallback for web users",
                ],
                "note": None,
            },
            {
                "n": 9,
                "title": "Doctor Portal & Patient Consultation",
                "tag": "COLLABORATION",
                "scope": "FYP Core",
                "desc": (
                    "Verified doctors receive a dedicated, separate dashboard distinct from the patient "
                    "interface. They can review any report a patient explicitly shares with them, add "
                    "clinical notes, override or annotate AI risk assessments, and communicate with "
                    "patients directly. All doctor accounts require admin approval before activation."
                ),
                "bullets": [
                    "Full biomarker summary, ML risk scores, lifestyle profile, and sensor data visible per shared report",
                    "Doctors can annotate, countermand, or endorse the AI risk assessment — human oversight by design",
                    "Asynchronous consultation messaging — secure, record-linked, audit-trailed",
                    "Doctor access is always patient-initiated — patients share; doctors cannot request access",
                    "Admin approval required for all doctor accounts before any patient data is accessible",
                ],
                "note": None,
            },
            {
                "n": 10,
                "title": "Data Privacy, Security & Consent Framework",
                "tag": "SECURITY",
                "scope": "FYP Core",
                "desc": (
                    "A standalone module — not an afterthought — that governs how all patient health data "
                    "is collected, stored, transmitted, and deleted. Designed to comply with the principles "
                    "of HIPAA, GDPR, and Pakistan's Personal Data Protection Bill framework. "
                    "Every patient sees a transparent consent screen on first use."
                ),
                "bullets": [
                    "End-to-end encryption: all API calls via HTTPS/TLS; no plain-text health data in transit",
                    "Encrypted storage at rest: database-level encryption for all patient medical records",
                    "Explicit informed consent on signup: what data is collected, how it is used, what is never done with it",
                    "Medical disclaimer on every risk result screen: BloodMAPP provides risk indicators — not diagnoses. "
                    "Consult a qualified doctor before any health decision. (Displayed in Urdu and English.)",
                    "Right to deletion: patients can permanently delete their account and all associated data",
                    "Sensor data permissions follow iOS HealthKit and Android Health Connect platform standards",
                ],
                "note": "This module is a direct response to medico-legal risk. AI-based health tools "
                        "carry liability exposure if they present outputs as clinical diagnoses. "
                        "The disclaimer and consent architecture protects users and WebClinic alike.",
            },
        ]
    },
    {
        "label": "PHASE 3 — PREMIUM & OPERATIONS",
        "note":  "Business model, advanced features, and platform management",
        "modules": [
            {
                "n": 11,
                "title": "Premium Subscription & Payments",
                "tag": "BUSINESS",
                "scope": "FYP — UI Complete, Payment Stubbed",
                "desc": (
                    "BloodMAPP operates on a freemium model. The free tier provides real, meaningful "
                    "value — not a crippled product. Premium unlocks deeper intelligence and direct "
                    "doctor access. Payment integration targets Pakistani mobile wallets — not credit "
                    "cards — making premium genuinely accessible to the target market."
                ),
                "bullets": [
                    "Free: Report upload, full dashboard, three risk scores, general recommendations, bilingual output",
                    "Premium: Personalised action plans, doctor consultation chat, future trajectory prediction",
                    "Payment via local Pakistani mobile wallets — to be integrated with available gateway",
                    "Subscription UI flow fully built in FYP; live payment processing to be finalised post-FYP",
                ],
                "note": "Scope note: Payment gateway merchant registration requires business documentation "
                        "and approval timelines outside the FYP window. The complete UI and business logic "
                        "are delivered; live payment processing is marked as a post-FYP integration task.",
            },
            {
                "n": 12,
                "title": "Future Trajectory Prediction",
                "tag": "PREMIUM AI",
                "scope": "FYP — Documented, Partially Implemented",
                "desc": (
                    "Activates once a patient has uploaded three or more blood reports over time. "
                    "A regression model analyses the trend in each key biomarker and projects its "
                    "trajectory over the next 12 to 18 months — displayed as a visual trend graph "
                    "with a plain-language clinical statement."
                ),
                "bullets": [
                    "Example projection: 'At the current rate, HbA1c is projected to enter the diabetic range in approximately 14 months.'",
                    "Biomarker-level trend lines for every tracked marker across all uploaded reports",
                    "Risk trajectory card: whether overall risk is improving, stable, or worsening",
                    "Available in premium tier only; requires a minimum of three uploads to activate",
                ],
                "note": "Scope note: Demonstration requires patients with three sequential uploads. "
                        "During FYP evaluation, synthetic multi-upload data will be used to demonstrate "
                        "the feature. Real-world activation depends on platform adoption post-launch.",
            },
            {
                "n": 13,
                "title": "Admin Panel & Platform Operations",
                "tag": "OPERATIONS",
                "scope": "FYP Core",
                "desc": (
                    "The operational backbone of WebClinic's platform. Without this module the "
                    "platform cannot be safely run, doctor quality cannot be assured, and the ML "
                    "model cannot be monitored or improved. Designed for WebClinic's internal team."
                ),
                "bullets": [
                    "Doctor account verification, approval, and revocation workflow",
                    "Platform-wide usage analytics: active patients, uploads per day, feature engagement",
                    "ML model performance monitoring: prediction distribution, flagged anomalies over time",
                    "Flagged content and user complaint management",
                    "Premium subscription status and management dashboard",
                ],
                "note": None,
            },
            {
                "n": 14,
                "title": "Multilingual Interface (Urdu / English)",
                "tag": "ACCESSIBILITY",
                "scope": "FYP Core",
                "desc": (
                    "System-wide bilingual support built into the architecture from day one — not added "
                    "as a translation layer on top. Every screen, result, explanation, risk score, "
                    "recommendation, alert, and notification is available in both Urdu and English. "
                    "Users toggle language in settings; the preference is saved to their profile."
                ),
                "bullets": [
                    "UI language toggle: Urdu and English on every screen",
                    "AI-generated explanations output in both languages simultaneously",
                    "Risk result cards, biomarker labels, and recommendations fully localised",
                    "Push notifications and email alerts sent in the patient's selected language",
                ],
                "note": "Differentiation note: No existing competitor — globally or in Pakistan — "
                        "provides medically accurate health risk analysis in Urdu. This is one of "
                        "BloodMAPP's strongest real-world differentiators.",
            },
            {
                "n": 15,
                "title": "Feedback Loop & Continuous Model Improvement",
                "tag": "ETHICAL AI",
                "scope": "FYP Core",
                "desc": (
                    "Closes the loop between the AI system and its real-world users. After every risk "
                    "prediction, both patients and doctors can flag whether the output seemed accurate "
                    "or misleading. This feedback is structured, stored, and used to identify systematic "
                    "model weaknesses — particularly biases relevant to the Pakistani population."
                ),
                "bullets": [
                    "Post-prediction feedback prompt: patient and doctor can flag accuracy in one tap",
                    "Structured feedback data stored for ML retraining cycles — not discarded",
                    "Population-specific bias tracking: feedback segmented by age, gender, city, and ethnicity",
                    "Model performance dashboard in Admin Panel (Module 13) surfaces feedback trends",
                    "Directly mitigates the known limitation of training on non-Pakistani datasets — "
                    "the feedback loop is the mechanism for localisation over time",
                ],
                "note": "Academic note: This module is the response to the well-documented problem of "
                        "algorithmic bias in underrepresented populations. Acknowledging and building "
                        "the correction mechanism into the architecture is the responsible approach.",
            },
        ]
    },
]

def row(item, aw):
    nw, tw, tagw = 26, aw - 26 - 60, 60

    num_t = Table([[Paragraph(str(item[\"n\"]), S[\"mod_num\"])]], colWidths=[nw])
    num_t.setStyle(TableStyle([
        (\"BACKGROUND\",    (0,0),(-1,-1), WHITE),
        (\"VALIGN\",        (0,0),(-1,-1), \"MIDDLE\"),
        (\"TOPPADDING\",    (0,0),(-1,-1), 10),
        (\"BOTTOMPADDING\", (0,0),(-1,-1), 10),
        (\"LEFTPADDING\",   (0,0),(-1,-1), 2),
        (\"RIGHTPADDING\",  (0,0),(-1,-1), 3),
    ]))

    inner_rows = [[Paragraph(item[\"title\"], S[\"mod_title\"])]]
    inner_rows.append([Paragraph(item[\"desc\"], S[\"mod_desc\"])])
    if item[\"bullets\"]:
        for b in item[\"bullets\"]:
            inner_rows.append([B(b)])
    if item[\"note\"]:
        inner_rows.append([Paragraph(item[\"note\"], S[\"mod_note\"])])

    inner_t = Table(inner_rows, colWidths=[tw])
    style_cmds = [
        (\"BACKGROUND\",    (0,0),(-1,-1), WHITE),
        (\"TOPPADDING\",    (0,0),(0,0),   8),
        (\"BOTTOMPADDING\", (0,-1),(0,-1), 8),
        (\"TOPPADDING\",    (0,1),(0,-1),  2),
        (\"BOTTOMPADDING\", (0,0),(0,-2),  2),
        (\"LEFTPADDING\",   (0,0),(-1,-1), 7),
        (\"RIGHTPADDING\",  (0,0),(-1,-1), 6),
        (\"VALIGN\",        (0,0),(-1,-1), \"TOP\"),
    ]
    inner_t.setStyle(TableStyle(style_cmds))

    tag_t = Table([[Paragraph(item[\"tag\"],  S[\"mod_tag\"])],
                   [Paragraph(item[\"scope\"], S[\"mod_note\"])]],
                  colWidths=[tagw])
    tag_t.setStyle(TableStyle([
        (\"BACKGROUND\",    (0,0),(-1,-1), BG),
        (\"VALIGN\",        (0,0),(-1,-1), \"MIDDLE\"),
        (\"TOPPADDING\",    (0,0),(-1,-1), 6),
        (\"BOTTOMPADDING\", (0,0),(-1,-1), 6),
        (\"LEFTPADDING\",   (0,0),(-1,-1), 5),
        (\"RIGHTPADDING\",  (0,0),(-1,-1), 5),
        (\"ALIGN\",         (0,0),(-1,-1), \"CENTER\"),
    ]))

    outer = Table([[num_t, inner_t, tag_t]], colWidths=[nw, tw, tagw])
    outer.setStyle(TableStyle([
        (\"LEFTPADDING\",   (0,0),(-1,-1), 0),
        (\"RIGHTPADDING\",  (0,0),(-1,-1), 0),
        (\"TOPPADDING\",    (0,0),(-1,-1), 0),
        (\"BOTTOMPADDING\", (0,0),(-1,-1), 0),
        (\"VALIGN\",        (0,0),(-1,-1), \"TOP\"),
        (\"BOX\",           (0,0),(-1,-1), 0.4, RULE),
        (\"LINEAFTER\",     (0,0),(0,-1),  0.4, RULE),
        (\"LINEAFTER\",     (1,0),(1,-1),  0.4, RULE),
    ]))
    return outer

def build():
    path = \"BloodMAPP_Final_Proposal.pdf\"
    doc  = SimpleDocTemplate(
        path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=MARGIN,
        title=\"BloodMAPP — Final System Proposal\",
        author=\"BloodMAPP FYP Team\"
    )
    story = []

    # ── TITLE BLOCK
    story += [sp(6),
              Paragraph(\"BloodMAPP\", S[\"h_title\"]),
              sp(5),
              Paragraph(\"AI-Powered Metabolic Health Intelligence Platform\", S[\"h_sub\"]),
              sp(3),
              Paragraph(
                  \"Final Year Project  ·  Industry Collaboration — WebClinic  ·  2025–2026\",
                  S[\"h_meta\"]),
              sp(12),
              hr(RULE, 0.8, 0, 6),
              sp(3)]

    story.append(Paragraph(
        \"Pakistan has one of the highest rates of diabetes in the world, with over 33 million adults affected — \"
        \"yet most patients receive a blood test result with no guidance on what it means. BloodMAPP closes \"
        \"that gap: an AI system that reads any Pakistani blood report and simultaneously assesses risk for \"
        \"Diabetes, Cardiovascular Disease, and Fatty Liver — in Urdu and English — making prevention \"
        \"accessible without requiring a medical degree.\",
        S[\"h_pitch\"]))
    story += [sp(10), hr(RULE, 0.5, 0, 8)]

    # ── KEY CONTRIBUTIONS BOX
    story.append(Paragraph(\"ACADEMIC CONTRIBUTIONS\", S[\"sec_lbl\"]))
    contribs = [
        (\"Simultaneous Multi-Disease Prediction\",
         \"A single blood panel input produces concurrent risk scores for three metabolic diseases plus \"
         \"Metabolic Syndrome detection — novel at the consumer health app level.\"),
        (\"Pakistan-Specific Design\",
         \"Handles non-standardised Pakistani lab formats, stores per-lab reference intervals, \"
         \"and is validated against a Pakistani patient population through WebClinic collaboration.\"),
        (\"Three-Tier Passive Health Monitoring\",
         \"Replaces subjective lifestyle self-report with objective sensor data — from phone step counting \"
         \"(all users) to smartwatch sleep/heart-rate to CGM glucose integration — improving ML accuracy.\"),
        (\"Ethical AI Architecture\",
         \"Built-in medical disclaimer, explicit consent framework, SHAP-based explainability, and a \"
         \"structured feedback loop for ongoing bias mitigation in an underrepresented population.\"),
        (\"Bilingual Health AI\",
         \"Urdu-language health risk communication — no commercial competitor globally provides \"
         \"clinically-grounded metabolic risk analysis in Urdu.\"),
    ]
    cw1, cw2 = AW * 0.28, AW * 0.72
    c_rows = [[Paragraph(c[0], S[\"contrib_hd\"]), Paragraph(c[1], S[\"contrib_bd\"])]
              for c in contribs]
    c_tbl = Table(c_rows, colWidths=[cw1, cw2])
    c_tbl.setStyle(TableStyle([
        (\"TOPPADDING\",    (0,0),(-1,-1), 5),
        (\"BOTTOMPADDING\", (0,0),(-1,-1), 5),
        (\"LEFTPADDING\",   (0,0),(-1,-1), 8),
        (\"RIGHTPADDING\",  (0,0),(-1,-1), 8),
        (\"ROWBACKGROUNDS\",(0,0),(-1,-1), [WHITE, BG]),
        (\"GRID\",          (0,0),(-1,-1), 0.3, RULE),
        (\"VALIGN\",        (0,0),(-1,-1), \"TOP\"),
    ]))
    story += [c_tbl, sp(10), hr(RULE, 0.5, 0, 8)]

    # ── USER ROLES
    story.append(Paragraph(\"USER ROLES\", S[\"sec_lbl\"]))
    rw = AW / 3
    roles = [
        (\"Patient\",
         \"Uploads blood reports. Views risk scores, recommendations, \"
         \"biomarker trends, and sensor health data. Shares reports with doctors. \"
         \"Manages profile and subscription.\"),
        (\"Doctor\",
         \"Receives reports shared by patients. Views full biomarker and risk \"
         \"summary. Adds clinical notes. Annotates or overrides AI assessments. \"
         \"Sends consultation messages. Requires admin approval.\"),
        (\"Admin (WebClinic)\",
         \"Verifies and approves doctor accounts. Monitors platform analytics \"
         \"and ML model performance. Manages flagged content. Oversees \"
         \"subscriptions and operational reporting.\"),
    ]
    r_data = [[
        Table([[Paragraph(r[0], S[\"role_hd\"])], [Paragraph(r[1], S[\"role_bd\"])]],
              colWidths=[rw - 4])
        for r in roles
    ]]
    r_tbl = Table(r_data, colWidths=[rw] * 3)
    r_tbl.setStyle(TableStyle([
        (\"BACKGROUND\",    (0,0),(-1,-1), BG),
        (\"TOPPADDING\",    (0,0),(-1,-1), 8),
        (\"BOTTOMPADDING\", (0,0),(-1,-1), 8),
        (\"LEFTPADDING\",   (0,0),(-1,-1), 8),
        (\"RIGHTPADDING\",  (0,0),(-1,-1), 8),
        (\"VALIGN\",        (0,0),(-1,-1), \"TOP\"),
        (\"GRID\",          (0,0),(-1,-1), 0.4, RULE),
    ]))
    story += [r_tbl, sp(10), hr(RULE, 0.5, 0, 8)]

    # ── MODULES
    story.append(Paragraph(\"SYSTEM MODULES — 15 TOTAL\", S[\"sec_lbl\"]))
    story.append(sp(4))

    for phase in PHASES:
        # Phase header
        ph_tbl = Table(
            [[Paragraph(phase[\"label\"], S[\"phase_hd\"]),
              Paragraph(phase[\"note\"], S[\"mod_note\"])]],
            colWidths=[AW * 0.55, AW * 0.45]
        )
        ph_tbl.setStyle(TableStyle([
            (\"BACKGROUND\",    (0,0),(-1,-1), BG),
            (\"TOPPADDING\",    (0,0),(-1,-1), 7),
            (\"BOTTOMPADDING\", (0,0),(-1,-1), 7),
            (\"LEFTPADDING\",   (0,0),(-1,-1), 10),
            (\"RIGHTPADDING\",  (0,0),(-1,-1), 8),
            (\"VALIGN\",        (0,0),(-1,-1), \"MIDDLE\"),
            (\"LINEBELOW\",     (0,0),(-1,-1), 1.2, ACCENT),
        ]))
        story.append(KeepTogether([ph_tbl]))
        story.append(sp(4))

        for m in phase[\"modules\"]:
            story.append(KeepTogether([row(m, AW), sp(3)]))

        story.append(sp(8))

    story.append(hr(RULE, 0.8, 0, 6))

    # ── TECH STACK
    story.append(Paragraph(\"TECHNOLOGY STACK\", S[\"sec_lbl\"]))
    story.append(sp(4))
    tech = [
        (\"Mobile App\",          \"Flutter — iOS & Android from a single codebase\"),
        (\"Web Portal\",          \"React.js — Admin panel and doctor web access\"),
        (\"Backend API\",         \"Python — REST API framework (Django REST or FastAPI)\"),
        (\"Database\",            \"SQL or NoSQL — selected based on data structure requirements (e.g. PostgreSQL or Firebase)\"),
        (\"ML / AI Models\",      \"Python — model selection via benchmarking across Logistic Regression, Random Forest, XGBoost, LightGBM\"),
        (\"Explainability\",      \"SHAP (SHapley Additive exPlanations) — identifies which biomarkers drive each risk score\"),
        (\"Report Extraction\",   \"Automated vision-based OCR pipeline — lab-format agnostic, handles Pakistani lab variability\"),
        (\"AI Text Generation\",  \"LLM-based generation for Urdu and English explanations and recommendations\"),
        (\"Passive Monitoring\",  \"Flutter health package — pedometer (steps), Health Connect (Android), HealthKit (iOS), CGM Bluetooth (Phase 3)\"),
        (\"Notifications\",       \"Push notification service (mobile) + email (web)\"),
        (\"Payments\",            \"Local Pakistani mobile wallet integration — to be finalised\"),
        (\"Hosting\",             \"Cloud-hosted backend — provider to be determined based on cost and performance\"),
    ]
    c1, c2 = AW * 0.25, AW * 0.75
    t_rows = [[Paragraph(t[0], S[\"tk\"]), Paragraph(t[1], S[\"tv\"])] for t in tech]
    t_tbl = Table(t_rows, colWidths=[c1, c2])
    t_tbl.setStyle(TableStyle([
        (\"TOPPADDING\",     (0,0),(-1,-1), 5),
        (\"BOTTOMPADDING\",  (0,0),(-1,-1), 5),
        (\"LEFTPADDING\",    (0,0),(-1,-1), 8),
        (\"RIGHTPADDING\",   (0,0),(-1,-1), 8),
        (\"ROWBACKGROUNDS\", (0,0),(-1,-1), [WHITE, BG]),
        (\"GRID\",           (0,0),(-1,-1), 0.3, RULE),
        (\"VALIGN\",         (0,0),(-1,-1), \"MIDDLE\"),
    ]))
    story += [t_tbl, sp(10), hr(RULE, 0.8, 0, 6)]

    # ── DISCLAIMER + FOOTER
    story.append(Paragraph(
        \"Medical Disclaimer — BloodMAPP provides AI-generated risk indicators for informational and \"
        \"educational purposes only. It does not constitute a medical diagnosis, clinical opinion, or \"
        \"treatment recommendation. Users are advised to consult a qualified healthcare professional \"
        \"before making any health-related decisions. This disclaimer appears on every risk result screen \"
        \"within the application, in both Urdu and English.\",
        S[\"disc\"]))
    story += [sp(6),
              Paragraph(
                  \"BloodMAPP  ·  Final Year Project — WebClinic Industry Collaboration  ·  \"
                  \"2025–2026  ·  Confidential Working Document\",
                  S[\"footer\"])]

    doc.build(story)
    print(\"Done:\", path)

build()
