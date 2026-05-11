import { Link } from 'react-router-dom'

const steps = [
  {
    n: '01',
    title: 'Prepare Your Reference Signature',
    desc: 'Take a scan or clear photo of the known-genuine signature — this is your baseline. This is typically taken from an official document already on file (passport, bank form, ID). Make sure the image is clear, horizontal, and has a light background.',
    tip: '💡 Tip: Use a flatbed scanner for best results. Avoid dark or blurry photos.',
  },
  {
    n: '02',
    title: 'Prepare the Test Signature',
    desc: 'This is the signature you want to verify — the one on the document in question. Scan or photograph it under the same conditions as the reference for the most accurate comparison.',
    tip: '💡 Tip: Both images should be similar resolution and lighting if possible.',
  },
  {
    n: '03',
    title: 'Copy the Document Text',
    desc: 'Paste the written content of the legal document into the text field. This is the body of the contract, agreement, or claim. The AI will analyse linguistic patterns for deception signals. Minimum one full sentence required.',
    tip: '💡 Tip: More text = better analysis. Short phrases may not give meaningful results.',
  },
  {
    n: '04',
    title: 'Click "Verify Document Authenticity"',
    desc: 'The system sends both images and the text to the AI pipeline. The Siamese CNN compares signatures, RoBERTa analyses the text, and the Supervisor Agent combines both results into a final verdict.',
    tip: '⏱ Processing takes 10–30 seconds. SHAP analysis is the slowest step.',
  },
  {
    n: '05',
    title: 'Read the Analysis Results',
    desc: 'You will see: (1) A final AUTHENTIC or SUSPICIOUS verdict, (2) Individual scores for signature similarity and text deception, (3) A Grad-CAM heatmap showing which parts of the signature the model focused on, (4) SHAP word attribution showing which words drove the deception score.',
    tip: '🔴 Red regions in the heatmap = high model attention. Blue = low attention.',
  },
  {
    n: '06',
    title: 'Interpret the Scores',
    desc: 'Signature Similarity ≥ 0.70 → GENUINE. Text Deception ≥ 0.45 → DECEPTIVE. Combined Risk ≥ 0.40 → SUSPICIOUS overall. Remember: this is a screening tool. A SUSPICIOUS verdict should trigger further human expert review, not be used as a standalone verdict.',
    tip: '⚖️ This system is an AI assistant for forensic screening — not a legal judgment.',
  },
]

const faqs = [
  {
    q: 'Why does the text model sometimes miss fraud in contracts?',
    a: 'The text model was trained on political statements (LIAR dataset). Formal legal language patterns differ from political speech. It detects linguistic deception patterns, not domain-specific legal fraud.',
  },
  {
    q: 'Can a skilled forger fool the signature model?',
    a: 'Possibly — the model achieves 80.21% on skilled forgeries. A forger copies what they see (overall shape), but the CNN compares 128-dimensional micro-pattern embeddings invisible to the human eye. Most skilled forgeries still fail the comparison.',
  },
  {
    q: 'What image formats are supported?',
    a: 'PNG, JPG, and JPEG. Images must be at least 30×30 pixels and not be blank or a solid color. Portrait-orientation images (very tall) are rejected as they are unlikely to be signatures.',
  },
  {
    q: 'Why does the heatmap sometimes look flat?',
    a: 'Grad-CAM works best on classifiers. Since our Siamese network computes similarity (not a class), we approximate by backpropagating through the embedding norm. For uniform signatures, gradients can flatten out.',
  },
]

export default function HowToUse() {
  return (
    <div className="section">
      {/* Header */}
      <div className="section-heading">
        <span className="eyebrow">User Guide</span>
        <h2>How to Verify a Document</h2>
        <p>Follow these six steps to get a forensic-grade authenticity analysis in under a minute.</p>
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

        <div className="feature-grid">
          {faqs.map(f => (
            <div key={f.q} className="feature-card glass">
              <h3 style={{ marginBottom: 12, fontSize: '0.95rem' }}>❓ {f.q}</h3>
              <p>{f.a}</p>
            </div>
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
