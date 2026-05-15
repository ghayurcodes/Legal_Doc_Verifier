import { Link } from 'react-router-dom'

const features = [
  {
    icon: '✍️',
    bg: 'rgba(201,151,58,0.15)',
    title: 'Signature Verification',
    desc: 'Siamese CNN with VGG16 backbone compares pen-stroke micro-patterns invisible to the human eye. Trained on the CEDAR dataset — the forensic gold standard.',
  },
  {
    icon: '🧠',
    bg: 'rgba(100,149,237,0.12)',
    title: 'Predatory Clause Detection',
    desc: 'Fine-tuned RoBERTa analyses the linguistic patterns of the document text to flag unfair, high-risk, or predatory legal clauses.',
  },
  {
    icon: '🔥',
    bg: 'rgba(224,100,80,0.12)',
    title: 'Grad-CAM Heatmaps',
    desc: 'Gradient-weighted activation maps highlight exactly which pen strokes the model focused on — making the AI decision fully explainable.',
  },
  {
    icon: '📊',
    bg: 'rgba(76,175,125,0.12)',
    title: 'SHAP Word Attribution',
    desc: 'SHapley Additive exPlanations reveal which specific words in the document drove the risk score toward SAFE or UNFAIR.',
  },
  {
    icon: '⚖️',
    bg: 'rgba(201,151,58,0.15)',
    title: 'Supervisor Fusion Agent',
    desc: 'A weighted decision agent combines signature risk (60%) and text risk (40%) into a single AUTHENTIC / SUSPICIOUS verdict.',
  },
  {
    icon: '🔒',
    bg: 'rgba(150,100,200,0.12)',
    title: 'EU AI Act Compliant',
    desc: 'High-risk AI systems must be transparent. Every verdict comes with full XAI explanation — no black-box decisions.',
  },
]

const techStack = [
  { label: 'Signature Model', value: 'Siamese CNN', sub: 'VGG16 backbone · CEDAR dataset' },
  { label: 'Text Model',      value: 'RoBERTa-base', sub: 'Fine-tuned · LexGLUE UNFAIR-ToS' },
  { label: 'Explainability',  value: 'Grad-CAM + SHAP', sub: 'XAI for images and text' },
  { label: 'Backend',         value: 'FastAPI + PyTorch', sub: 'REST API · CPU inference' },
  { label: 'Frontend',        value: 'React + Vite', sub: 'Modern SPA' },
  { label: 'Training',        value: 'Google Colab', sub: 'T4 GPU · ~4 hrs total' },
]

export default function About() {
  return (
    <>
      {/* ── Hero ──────────────────────────────────────────────── */}
      <section className="hero">
        <div className="hero-inner">
          <div className="hero-badge glass-gold">
            <span>⚖️</span>
            <span>AI-Powered Forensic Document Analysis</span>
          </div>

          <h1>
            Detect Forged Signatures<br />
            &amp; <span className="gold">Predatory Clauses</span><br />
            with Deep Learning
          </h1>

          <p className="hero-sub">
            LegalVerify combines a Siamese CNN for signature verification
            and a fine-tuned RoBERTa transformer to detect unfair legal clauses —
            delivering explainable, forensic-grade authenticity verdicts.
          </p>

          <div className="hero-actions">
            <Link to="/verify" className="btn-primary">🔎 Verify a Document</Link>
            <Link to="/how-to-use" className="btn-secondary">📖 How It Works</Link>
          </div>
        </div>
      </section>

      <div className="divider" />

      {/* ── Stats ─────────────────────────────────────────────── */}
      <div className="section">
        <div className="stats-row">
          {[
            { v: '80.21%', l: 'Signature Accuracy' },
            { v: '95.83%', l: 'Clause Accuracy' },
            { v: '89.65%', l: 'Macro F1 Score' },
            { v: '125M',   l: 'RoBERTa Parameters' },
          ].map(s => (
            <div key={s.l} className="stat-card glass">
              <span className="stat-value">{s.v}</span>
              <span className="stat-label">{s.l}</span>
            </div>
          ))}
        </div>

        {/* ── Features ──────────────────────────────────────────── */}
        <div className="section-heading">
          <span className="eyebrow">Core Capabilities</span>
          <h2>Everything You Need for Document Forensics</h2>
          <p>Six tightly integrated AI modules working in concert to analyse legal documents end-to-end.</p>
        </div>

        <div className="feature-grid">
          {features.map(f => (
            <div key={f.title} className="feature-card glass">
              <div className="feature-icon" style={{ background: f.bg }}>{f.icon}</div>
              <h3>{f.title}</h3>
              <p>{f.desc}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="divider" />

      {/* ── Tech stack ────────────────────────────────────────── */}
      <div className="section">
        <div className="section-heading">
          <span className="eyebrow">Technology Stack</span>
          <h2>Built on Proven Research Foundations</h2>
          <p>Every component is grounded in published academic benchmarks.</p>
        </div>

        <div className="tech-grid">
          {techStack.map(t => (
            <div key={t.label} className="tech-card glass">
              <div className="tech-card-label">{t.label}</div>
              <div className="tech-card-value">{t.value}</div>
              <div className="tech-card-sub">{t.sub}</div>
            </div>
          ))}
        </div>

        <div style={{ textAlign: 'center', marginTop: 56 }}>
          <Link to="/verify" className="btn-primary">🔎 Try It Now</Link>
        </div>
      </div>
    </>
  )
}
