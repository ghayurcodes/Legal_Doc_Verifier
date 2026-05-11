import { NavLink } from 'react-router-dom'

export default function Navbar() {
  return (
    <nav className="navbar">
      <NavLink to="/" className="nav-logo">
        <div className="nav-logo-icon">⚖️</div>
        <span className="nav-logo-text">Legal<span>Verify</span></span>
      </NavLink>

      <ul className="nav-links">
        <li><NavLink to="/"            className={({isActive}) => isActive ? 'active' : ''}>About</NavLink></li>
        <li><NavLink to="/how-to-use"  className={({isActive}) => isActive ? 'active' : ''}>How to Use</NavLink></li>
        <li><NavLink to="/verify"      className={({isActive}) => isActive ? 'nav-cta active' : 'nav-cta'}>Verify Document</NavLink></li>
      </ul>
    </nav>
  )
}
