import { Link } from 'react-router-dom'

export default function Metrics() {
  return (
    <div className="section">
      <div className="section-heading">
        <span className="eyebrow">Scientific Validation</span>
        <h2>Evaluation Metrics</h2>
        <p>A transparent breakdown of how our deep learning models perform on unseen test data.</p>
      </div>

      <div className="metrics-container" style={{ display: 'flex', flexDirection: 'column', gap: '60px' }}>
        
        {/* ROberta Section */}
        <div className="metric-card glass" style={{ padding: '40px', borderRadius: '16px' }}>
          <h3 style={{ color: 'var(--gold)', marginBottom: '8px' }}>Text Analysis (Unfair Clause Detection)</h3>
          <p style={{ marginBottom: '32px', color: '#ccc' }}>
            Powered by RoBERTa-base (125M Parameters) fine-tuned on the LexGLUE UNFAIR-ToS benchmark.
          </p>
          
          <img 
            src="/metrics/roberta_eval.png" 
            alt="RoBERTa Evaluation Charts" 
            style={{ width: '100%', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.1)', marginBottom: '24px' }} 
          />
          
          <div className="metric-explanations" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '24px' }}>
            <div className="explanation-box" style={{ background: 'rgba(255,255,255,0.05)', padding: '20px', borderRadius: '12px' }}>
              <h4 style={{ marginBottom: '8px', color: '#fff' }}>1. Confusion Matrix</h4>
              <p style={{ fontSize: '0.9rem', color: '#aaa', lineHeight: '1.5' }}>
                Shows exactly where the model gets confused. It compares the model's predictions (Safe/Unfair) against the actual correct answers. A strong diagonal (top-left to bottom-right) means the model is highly accurate.
              </p>
            </div>
            
            <div className="explanation-box" style={{ background: 'rgba(255,255,255,0.05)', padding: '20px', borderRadius: '12px' }}>
              <h4 style={{ marginBottom: '8px', color: '#fff' }}>2. ROC Curve & AUC</h4>
              <p style={{ fontSize: '0.9rem', color: '#aaa', lineHeight: '1.5' }}>
                The ROC (Receiver Operating Characteristic) curve plots the true positive rate against the false alarm rate. An AUC (Area Under Curve) close to 1.0 (ours is 0.956) means the model has excellent distinguishing power.
              </p>
            </div>

            <div className="explanation-box" style={{ background: 'rgba(255,255,255,0.05)', padding: '20px', borderRadius: '12px' }}>
              <h4 style={{ marginBottom: '8px', color: '#fff' }}>3. Macro F1 Score</h4>
              <p style={{ fontSize: '0.9rem', color: '#aaa', lineHeight: '1.5' }}>
                F1 Score is a harsh metric that balances Precision (not crying wolf) and Recall (catching all unfair clauses). Our score of 89.65% proves the model is smart, not just guessing 'Safe' on every contract.
              </p>
            </div>
          </div>
        </div>

        {/* Siamese Section */}
        <div className="metric-card glass" style={{ padding: '40px', borderRadius: '16px' }}>
          <h3 style={{ color: 'var(--gold)', marginBottom: '8px' }}>Signature Verification</h3>
          <p style={{ marginBottom: '32px', color: '#ccc' }}>
            Powered by a Siamese CNN with a VGG16 backbone, trained on the CEDAR signature dataset. Tested on strictly unseen signers.
          </p>
          
          <img 
            src="/metrics/siamese_eval.png" 
            alt="Siamese CNN Evaluation Charts" 
            style={{ width: '100%', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.1)', marginBottom: '24px' }} 
          />
          
          <div className="metric-explanations" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '24px' }}>
            <div className="explanation-box" style={{ background: 'rgba(255,255,255,0.05)', padding: '20px', borderRadius: '12px' }}>
              <h4 style={{ marginBottom: '8px', color: '#fff' }}>1. Confusion Matrix</h4>
              <p style={{ fontSize: '0.9rem', color: '#aaa', lineHeight: '1.5' }}>
                Displays the model's ability to spot forgeries. It successfully identified the vast majority of genuine and forged signatures, even when the forgeries were highly skilled and drawn by humans.
              </p>
            </div>
            
            <div className="explanation-box" style={{ background: 'rgba(255,255,255,0.05)', padding: '20px', borderRadius: '12px' }}>
              <h4 style={{ marginBottom: '8px', color: '#fff' }}>2. ROC Curve & AUC</h4>
              <p style={{ fontSize: '0.9rem', color: '#aaa', lineHeight: '1.5' }}>
                With an AUC of 0.92, this curve proves that our model is mathematically reliable at separating true signatures from fakes across all possible threshold levels.
              </p>
            </div>

            <div className="explanation-box" style={{ background: 'rgba(255,255,255,0.05)', padding: '20px', borderRadius: '12px' }}>
              <h4 style={{ marginBottom: '8px', color: '#fff' }}>3. Score Separation</h4>
              <p style={{ fontSize: '0.9rem', color: '#aaa', lineHeight: '1.5' }}>
                Notice the clear visual gap between the Genuine (blue) and Forged (orange) score distributions. A strong separation means the AI has high confidence in its verdicts.
              </p>
            </div>
          </div>
        </div>

      </div>

      <div style={{ textAlign: 'center', marginTop: 64 }}>
        <Link to="/verify" className="btn-primary">🔎 Test the Models</Link>
      </div>
    </div>
  )
}
