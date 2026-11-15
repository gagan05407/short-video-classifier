import React from 'react';

function Header({ activeTab, setActiveTab }) {
  return (
    <header className="app-header">
      <div className="header-inner">
        <div className="header-brand">
          <span className="brand-title">Short-Video Classifier</span>
          <span className="brand-sub">Reliability-Aware Multimodal Fusion</span>
        </div>
        <nav className="header-nav">
          <button
            type="button"
            className={`nav-btn ${activeTab === 'analyze' ? 'active' : ''}`}
            onClick={() => setActiveTab('analyze')}
          >
            Video Moderation
          </button>
          <button
            type="button"
            className={`nav-btn ${activeTab === 'results' ? 'active' : ''}`}
            onClick={() => setActiveTab('results')}
          >
            Benchmark Results
          </button>
        </nav>
      </div>
    </header>
  );
}

export default Header;
