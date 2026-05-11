import { Link } from 'react-router-dom'

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-left">
        © 2025 <strong>LegalVerify</strong> · Siamese CNN + RoBERTa · Grad-CAM + SHAP
      </div>
      <div className="footer-right">
        <Link to="/">About</Link>
        <Link to="/how-to-use">How to Use</Link>
        <Link to="/verify">Verify</Link>
      </div>
    </footer>
  )
}
