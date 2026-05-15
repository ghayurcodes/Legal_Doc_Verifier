import { useState, useRef, useEffect } from 'react'
import axios from 'axios'

/* ── Mode selector tabs ──────────────────────────────────────── */
const MODES = [
  {
    id: 'signature_only',
    icon: '✍️',
    label: 'Signature Only',
    desc: 'Verify if a signature is genuine or forged',
  },
  {
    id: 'text_only',
    icon: '📄',
    label: 'Contract Scan',
    desc: 'Scan contract text for unfair or predatory clauses',
  },
  {
    id: 'combined',
    icon: '🔎',
    label: 'Full Verification',
    desc: 'Verify signature + scan contract text together',
  },
]

/* ── Upload box ──────────────────────────────────────────────── */
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
          <img className="upload-preview" src={URL.createObjectURL(file)} alt="preview" />
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
  const [mode,     setMode]     = useState('combined')
  const [refFile,  setRefFile]  = useState(null)
  const [testFile, setTestFile] = useState(null)
  const [text,     setText]     = useState('')
  const [loading,  setLoading]  = useState(false)
  const [progress, setProgress] = useState(0)
  const [error,    setError]    = useState(null)
  const [result,   setResult]   = useState(null)

  const needsSig  = mode === 'signature_only' || mode === 'combined'
  const needsText = mode === 'text_only'      || mode === 'combined'

  const sigReady  = !needsSig  || (refFile && testFile)
  const textReady = !needsText || text.trim().length >= 20
  const canSubmit = sigReady && textReady && !loading

  // Manage fake progress bar
  useEffect(() => {
    let interval;
    if (loading) {
      setProgress(0)
      // Animate up to 90% over ~15 seconds (SHAP takes a while)
      interval = setInterval(() => {
        setProgress(p => (p < 90 ? p + (90 - p) * 0.1 : 90))
      }, 500)
    } else {
      setProgress(100)
    }
    return () => clearInterval(interval)
  }, [loading])

  // Reset state when mode changes
  const handleModeChange = (newMode) => {
    setMode(newMode)
    setResult(null)
    setError(null)
    setProgress(0)
  }

  const handleSubmit = async () => {
    setError(null)
    setResult(null)
    setLoading(true)

    const formData = new FormData()
    formData.append('analysis_mode', mode)
    if (needsSig) {
      formData.append('ref_signature',  refFile)
      formData.append('test_signature', testFile)
    }
    if (needsText) {
      formData.append('document_text', text)
    }

    try {
      const { data } = await axios.post('/api/verify', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000,
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

      {/* ── Page header ──────────────────────────────────────── */}
      <div className="verify-header">
        <h1>Verify Document Authenticity</h1>
        <p>Select a verification mode, upload the required inputs, and click Verify.</p>
      </div>

      {/* ── Mode Selector ────────────────────────────────────── */}
      <div className="mode-selector">
        {MODES.map(m => (
          <button
            key={m.id}
            className={`mode-tab ${mode === m.id ? 'active' : ''}`}
            onClick={() => handleModeChange(m.id)}
          >
            <span className="mode-tab-icon">{m.icon}</span>
            <span className="mode-tab-label">{m.label}</span>
            <span className="mode-tab-desc">{m.desc}</span>
          </button>
        ))}
      </div>

      {/* ── Step 1 — Signatures (shown only when needed) ─────── */}
      {needsSig && (
        <div className="verify-section">
          <div className="verify-step-label">
            {mode === 'combined' ? 'Step 1 —' : 'Step 1 —'} Upload Signature Images
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
        </div>
      )}

      {/* ── Step 2 — Document text (shown only when needed) ──── */}
      {needsText && (
        <div className="verify-section">
          <div className="verify-step-label">
            {mode === 'combined' ? 'Step 2 —' : 'Step 1 —'} Paste Contract / Document Text
          </div>
          <div className="text-area-wrapper">
            <textarea
              value={text}
              onChange={e => setText(e.target.value)}
              placeholder="Paste the body of the legal document or contract here. The AI will scan each sentence for unfair or predatory clauses…"
            />
            <span style={{ fontSize: '0.76rem', color: text.trim().length < 20 ? 'var(--red)' : 'var(--text-dim)' }}>
              {text.trim().length} chars {text.trim().length < 20 ? '(minimum 20 required)' : '✓'}
            </span>
          </div>
        </div>
      )}

      {/* ── Error ────────────────────────────────────────────── */}
      {error && (
        <div className="error-box">
          <span>⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {/* ── Submit button ─────────────────────────────────────── */}
      <button
        className="btn-primary verify-submit"
        onClick={handleSubmit}
        disabled={!canSubmit}
      >
        {loading ? '⏳  Analysing…' : '🔎  Verify Document Authenticity'}
      </button>

      {/* ── Loading ───────────────────────────────────────────── */}
      {loading && (
        <div className="spinner-wrap" style={{ textAlign: 'center', padding: '30px 0' }}>
          <div className="progress-container">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
          <div className="progress-text">{Math.round(progress)}% Complete</div>
          
          <p style={{ marginTop: 16 }}>
            {mode === 'signature_only' && 'Running signature verification + Grad-CAM…'}
            {mode === 'text_only'      && 'Scanning contract clauses + SHAP attribution…'}
            {mode === 'combined'       && 'Running full verification + XAI explanations…'}
          </p>
          {(mode === 'text_only' || mode === 'combined') && (
            <p style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>
              SHAP analysis may take 15–30 seconds. Please do not close the window.
            </p>
          )}
        </div>
      )}

      {/* ── Results ───────────────────────────────────────────── */}
      {result && (
        <section className="results-section">
          <div className="verify-step-label" style={{ marginBottom: 16 }}>
            Analysis Results
          </div>

          <div className="results-grid">

            {/* Verdict banner */}
            <div className={`verdict-banner ${isAuthentic ? 'authentic' : 'suspicious'}`}>
              <div className="verdict-icon">{isAuthentic ? '✅' : '🚨'}</div>
              <div>
                <div className="verdict-title">{isAuthentic ? 'AUTHENTIC' : 'SUSPICIOUS'}</div>
                <div className="verdict-sub">
                  {isAuthentic
                    ? 'All checks passed. No forgery or predatory clauses detected.'
                    : 'One or more checks failed. Human expert review recommended.'}
                </div>
              </div>
            </div>

            {/* Score breakdown */}
            <div className="scores-card glass">
              <h3>Detailed Scores</h3>

              {result.signature_score !== null && result.signature_score !== undefined && (
                <ScoreRow
                  name="Signature Similarity"
                  score={result.signature_score}
                  verdict={result.signature_verdict}
                  verdictType={result.signature_verdict === 'GENUINE' ? 'good' : 'bad'}
                />
              )}

              {result.unfair_score !== null && result.unfair_score !== undefined && (
                <ScoreRow
                  name="Predatory Clause Risk"
                  score={result.unfair_score}
                  verdict={result.text_verdict}
                  verdictType={result.text_verdict === 'SAFE' ? 'good' : 'bad'}
                />
              )}

              <ScoreRow
                name="Overall Risk Score"
                score={result.combined_risk}
                verdict={result.verdict}
                verdictType={result.verdict === 'AUTHENTIC' ? 'good' : 'bad'}
              />

              <div className="threshold-hint">
                {result.signature_score !== null && result.signature_score !== undefined &&
                  `Signature genuine if ≥ ${result.thresholds.signature_genuine_min} · `}
                {result.unfair_score !== null && result.unfair_score !== undefined &&
                  `Unfair clause if ≥ ${result.thresholds.unfair_clause_min} · `}
                Suspicious if risk ≥ {result.thresholds.combined_suspicious_min}
              </div>
            </div>

            {/* Grad-CAM Heatmap */}
            {result.heatmap && (
              <div className="heatmap-card glass">
                <h3>🔥 Grad-CAM — Signature Stroke Attention</h3>
                <img src={result.heatmap} alt="Grad-CAM heatmap" />
                <p style={{ marginTop: 12, fontSize: '0.78rem', color: 'var(--text-dim)' }}>
                  Red = high model attention · Blue = low attention
                </p>
              </div>
            )}

            {/* SHAP Words */}
            {result.shap_words && result.shap_words.length > 0 && (
              <div className="shap-card glass">
                <h3>⚖️ Key Words — Clause Risk Attribution</h3>
                <div className="shap-legend">
                  🔴 Pushed toward UNFAIR &nbsp;|&nbsp; 🟢 Pushed toward SAFE
                </div>
                {result.shap_words.map(({ word, score }) => (
                  <div key={word + score} className="shap-word-row">
                    <span className="shap-word">{word}</span>
                    <span className={`shap-score ${score > 0 ? 'pos' : 'neg'}`}>
                      {score > 0 ? '🔴' : '🟢'} {score > 0 ? '+' : ''}{score.toFixed(4)}
                    </span>
                  </div>
                ))}
              </div>
            )}

          </div>
        </section>
      )}
    </div>
  )
}
