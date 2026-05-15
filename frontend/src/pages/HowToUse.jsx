import { useState } from 'react'
import { Link } from 'react-router-dom'

const steps = [
  {
    n: '01',
    title: 'Select Verification Mode',
    desc: 'Choose between Signature Only, Contract Scan, or Full Verification based on what you need to analyse.',
    tip: '💡 Tip: Full Verification provides the most comprehensive security check.',
  },
  {
    n: '02',
    title: 'Upload Signature Images',
    desc: 'Take a scan or clear photo of the known-genuine signature (your baseline), and the test signature (the one under examination). Make sure the images are clear and horizontal.',
    tip: '💡 Tip: Both images should be similar resolution and lighting if possible.',
  },
  {
    n: '03',
    title: 'Paste the Contract Text',
    desc: 'Paste the written content of the legal document into the text field. The AI will scan the text line-by-line looking for unfair, predatory, or high-risk legal clauses.',
    tip: '💡 Tip: More text = better analysis. Minimum 20 characters required.',
  },
  {
    n: '04',
    title: 'Run AI Analysis',
    desc: 'The system sends the data to the deep learning models. The Siamese CNN compares signature strokes, RoBERTa analyses the clauses, and the Supervisor Agent calculates the total risk.',
    tip: '⏱ Processing takes 10–20 seconds while the SHAP explainability runs.',
  },
  {
    n: '05',
    title: 'Review Visual Evidence (XAI)',
    desc: 'Our system does not make black-box decisions. It provides a Grad-CAM heatmap showing exactly which pen strokes failed the check, and SHAP highlights showing exactly which words are predatory.',
    tip: '🔴 Red regions in heatmaps/text = high risk or model attention.',
  },
  {
    n: '06',
    title: 'Interpret the Final Verdict',
    desc: 'If the risk score crosses the threshold, the document is flagged as SUSPICIOUS. Remember: this is a screening tool. A SUSPICIOUS verdict should trigger further human expert review.',
    tip: '⚖️ This system is an AI assistant for forensic screening — not a legal judgment.',
  },
]

const faqs = [
  {
    q: 'How accurate is the Unfair Clause detection?',
    a: 'The RoBERTa model was fine-tuned on the LexGLUE UNFAIR-ToS benchmark—a professional dataset of real terms of service. It achieves an outstanding 95.83% accuracy in flagging predatory or risky clauses.',
  },
  {
    q: 'Can a skilled forger fool the signature model?',
    a: 'Possibly, but it is very difficult. The model achieves 80.21% accuracy even on skilled forgeries. A forger copies what they see (overall shape), but the Siamese CNN compares 128-dimensional micro-pattern embeddings that are practically invisible to the human eye.',
  },
  {
    q: 'What image formats are supported?',
    a: 'PNG, JPG, and JPEG. Images must be at least 30×30 pixels and not be blank or a solid color. Portrait-orientation images (very tall) are rejected as they are unlikely to be signatures.',
  },
  {
    q: 'Why does the heatmap sometimes look flat?',
    a: 'Grad-CAM works best on classifiers. Since our Siamese network computes similarity (not a specific class), we approximate the heatmap by backpropagating through the embedding norm. For very uniform signatures, the gradients can flatten out.',
  },
]

function FaqItem({ q, a }) {
  const [isOpen, setIsOpen] = useState(false)
  return (
    <div className={`faq-item ${isOpen ? 'open' : ''}`}>
      <button className="faq-question" onClick={() => setIsOpen(!isOpen)}>
        {q}
        <span className="faq-icon">▼</span>
      </button>
      <div className="faq-answer">
        <p>{a}</p>
      </div>
    </div>
  )
}

export default function HowToUse() {
  return (
    <div className="section">
      {/* Header */}
      <div className="section-heading">
        <span className="eyebrow">User Guide</span>
        <h2>How to Verify a Document</h2>
        <p>Follow these steps to get a forensic-grade authenticity analysis in under a minute.</p>
      </div>

      {/* Steps */}
      <div className="steps-list">
        {steps.map(s => (
          <div key={s.n} className="step-card glass">
            <div className="step-num">{s.n}</div>
            <div className="step-body">
              <h3>{s.title}</h3>
              <p>{s.desc}</p>
              <p style={{ marginTop: 10, fontSize: '0.82rem', color: 'var(--gold)' }}>{s.tip}</p>
            </div>
          </div>
        ))}
      </div>

      {/* FAQ */}
      <div style={{ marginTop: 80 }}>
        <div className="section-heading">
          <span className="eyebrow">FAQ</span>
          <h2>Common Questions</h2>
        </div>

        <div className="faq-list">
          {faqs.map(f => (
            <FaqItem key={f.q} q={f.q} a={f.a} />
          ))}
        </div>
      </div>

      {/* CTA */}
      <div style={{ textAlign: 'center', marginTop: 64 }}>
        <Link to="/verify" className="btn-primary">🔎 Start Verifying Now</Link>
      </div>
    </div>
  )
}
