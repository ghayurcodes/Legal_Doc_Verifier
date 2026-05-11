import { useState, useRef } from 'react'
import axios from 'axios'

/* ── Upload box component ────────────────────────────────────── */
function UploadBox({ label, sublabel, file, onFile, icon }) {
  const inputRef = useRef()
  const [drag, setDrag] = useState(false)

  const handleDrop = (e) => {
    e.preventDefault()
    setDrag(false)
    const f = e.dataTransfer.files[0]
    if (f) onFile(f)
  }

  return (
    <div
      className={`upload-box ${drag ? 'drag-over' : ''} ${file ? 'has-file' : ''}`}
      onDragOver={e => { e.preventDefault(); setDrag(true) }}
      onDragLeave={() => setDrag(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/jpg"
        onChange={e => e.target.files[0] && onFile(e.target.files[0])}
      />

      {file ? (
        <>
          <img
            className="upload-preview"
            src={URL.createObjectURL(file)}
            alt="preview"
          />
          <div className="upload-filename">✅ {file.name}</div>
        </>
      ) : (
        <>
          <div className="upload-icon">{icon}</div>
          <label>{label}</label>
          <span>{sublabel}</span>
        </>
      )}
    </div>
  )
}

/* ── Score row ───────────────────────────────────────────────── */
function ScoreRow({ name, score, verdict, verdictType }) {
  const cls = verdictType === 'good' ? 'green' : verdictType === 'bad' ? 'red' : 'gold'
  return (
    <div className="score-row">
      <span className="score-name">{name}</span>
      <div className="score-right">
        <span className="score-val">{score.toFixed(3)}</span>
        <span className={`badge ${cls}`}>{verdict}</span>
      </div>
    </div>
  )
}

/* ── Main Verify page ────────────────────────────────────────── */
export default function Verify() {
  const [refFile,  setRefFile]  = useState(null)
  const [testFile, setTestFile] = useState(null)
  const [text,     setText]     = useState('')
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState(null)
  const [result,   setResult]   = useState(null)

  const canSubmit = refFile && testFile && text.trim().length >= 20 && !loading

  const handleSubmit = async () => {
    setError(null)
    setResult(null)
    setLoading(true)

    const formData = new FormData()
    formData.append('ref_signature',  refFile)
    formData.append('test_signature', testFile)
    formData.append('document_text',  text)

    try {
      const { data } = await axios.post('/api/verify', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000,  // 2 min max for SHAP
      })
      setResult(data)
    } catch (err) {
      const msg = err.response?.data?.detail || err.message || 'Unknown error'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  const isAuthentic = result?.verdict === 'AUTHENTIC'

  return (
    <div className="verify-page">

      {/* ── Page header ────────────────────────────────────────── */}
      <div className="verify-header">
        <h1>Verify Document Authenticity</h1>
        <p>Upload both signature images and paste the document text. Results appear below within ~20 seconds.</p>
      </div>

      {/* ── Step 1 — Signatures ─────────────────────────────────── */}
      <div style={{ marginBottom: 8 }}>
        <label style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.8px' }}>
          Step 1 — Upload Signatures
        </label>
      </div>
      <div className="upload-row">
        <UploadBox
          label="Reference Signature"
          sublabel="Known genuine · drag & drop or click"
          icon="📋"
          file={refFile}
          onFile={setRefFile}
        />
        <UploadBox
          label="Test Signature"
          sublabel="Under examination · drag & drop or click"
          icon="🔍"
          file={testFile}
          onFile={setTestFile}
        />
      </div>

      {/* ── Step 2 — Document text ──────────────────────────────── */}
      <div className="text-area-wrapper">
        <label>Step 2 — Paste Document Text / Written Claims</label>
        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder="Paste the body of the legal document, contract, or written claim here…"
        />
        <span style={{ fontSize: '0.76rem', color: text.trim().length < 20 ? 'var(--red)' : 'var(--text-dim)' }}>
          {text.trim().length} chars {text.trim().length < 20 ? '(minimum 20 required)' : '✓'}
        </span>
      </div>

      {/* ── Error ───────────────────────────────────────────────── */}
      {error && (
        <div className="error-box">
          <span>⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {/* ── Submit ──────────────────────────────────────────────── */}
      <button
        className="btn-primary verify-submit"
        onClick={handleSubmit}
        disabled={!canSubmit}
      >
        {loading ? '⏳  Analysing…' : '🔎  Verify Document Authenticity'}
      </button>

      {/* ── Loading ─────────────────────────────────────────────── */}
      {loading && (
        <div className="spinner-wrap">
          <div className="spinner" />
          <p>Running signature verification + text analysis + XAI explanations…</p>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>SHAP analysis may take 15–30 seconds</p>
        </div>
      )}

      {/* ── Results ─────────────────────────────────────────────── */}
      {result && (
        <section className="results-section">
          <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.8px', marginBottom: 16 }}>
            Step 3 — Analysis Results
          </div>

          <div className="results-grid">

            {/* Verdict banner */}
            <div className={`verdict-banner ${isAuthentic ? 'authentic' : 'suspicious'}`}>
              <div className="verdict-icon">{isAuthentic ? '✅' : '🚨'}</div>
              <div>
                <div className="verdict-title">{isAuthentic ? 'AUTHENTIC' : 'SUSPICIOUS'}</div>
                <div className="verdict-sub">
                  {isAuthentic
                    ? 'Both the signature and document text passed the authenticity checks.'
                    : 'One or more checks flagged this document. Human expert review recommended.'}
                </div>
              </div>
            </div>

            {/* Scores */}
            <div className="scores-card glass">
              <h3>Detailed Scores</h3>

              <ScoreRow
                name="Signature Similarity"
                score={result.signature_score}
                verdict={result.signature_verdict}
                verdictType={result.signature_verdict === 'GENUINE' ? 'good' : 'bad'}
              />
              <ScoreRow
                name="Text Deception"
                score={result.deception_score}
                verdict={result.text_verdict}
                verdictType={result.text_verdict === 'TRUTHFUL' ? 'good' : 'bad'}
              />
              <ScoreRow
                name="Combined Risk"
                score={result.combined_risk}
                verdict={result.verdict}
                verdictType={result.verdict === 'AUTHENTIC' ? 'good' : 'bad'}
              />

              <div className="threshold-hint">
                Signature genuine if ≥ {result.thresholds.signature_genuine_min} &nbsp;·&nbsp;
                Text deceptive if ≥ {result.thresholds.text_deceptive_min} &nbsp;·&nbsp;
                Combined suspicious if ≥ {result.thresholds.combined_suspicious_min}
              </div>
            </div>

            {/* Heatmap */}
            <div className="heatmap-card glass">
              <h3>🔥 Grad-CAM Heatmap — Stroke Attention</h3>
              <img src={result.heatmap} alt="Grad-CAM heatmap" />
              <p style={{ marginTop: 12, fontSize: '0.78rem', color: 'var(--text-dim)' }}>
                Red = high model attention · Blue = low attention
              </p>
            </div>

            {/* SHAP words */}
            <div className="shap-card glass">
              <h3>🧠 Key Words — Deception Attribution</h3>
              <div className="shap-legend">
                🔴 Pushed toward DECEPTIVE &nbsp;|&nbsp; 🟢 Pushed toward TRUTHFUL
              </div>

              {result.shap_words.length > 0 ? (
                result.shap_words.map(({ word, score }) => (
                  <div key={word + score} className="shap-word-row">
                    <span className="shap-word">{word}</span>
                    <span className={`shap-score ${score > 0 ? 'pos' : 'neg'}`}>
                      {score > 0 ? '🔴' : '🟢'} {score > 0 ? '+' : ''}{score.toFixed(4)}
                    </span>
                  </div>
                ))
              ) : (
                <p style={{ color: 'var(--text-dim)', fontSize: '0.85rem' }}>
                  No significant individual words detected.
                </p>
              )}
            </div>

          </div>
        </section>
      )}
    </div>
  )
}
