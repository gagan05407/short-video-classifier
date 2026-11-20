import React, { useState, useEffect } from 'react';
import { getTableResults } from '../services/api';

function ResultsTab() {
  const [activeSubTab, setActiveSubTab] = useState(1);
  const [tableData, setTableData] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const tableTabs = [
    { id: 1, label: 'Table 1: Main Baselines' },
    { id: 2, label: 'Table 2: Ablation' },
    { id: 3, label: 'Table 3: Corruptions' },
    { id: 4, label: 'Table 4: Missing Modalities' },
    { id: 5, label: 'Table 5: Selective Review' },
  ];

  useEffect(() => {
    loadTable(activeSubTab);
  }, [activeSubTab]);

  const loadTable = async (id) => {
    if (tableData[id]) return;
    setLoading(true);
    setError('');
    try {
      const data = await getTableResults(id);
      setTableData((prev) => ({ ...prev, [id]: data }));
    } catch (err) {
      setError(err.message || `Failed to load Table ${id}`);
    } finally {
      setLoading(false);
    }
  };

  const renderTable1 = (data) => {
    // Data has: Method, Precision, Recall, F1, Note
    return (
      <div className="table-block">
        <table className="benchmark-table">
          <thead>
            <tr>
              <th>Method</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1 Score</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row, idx) => (
              <tr key={idx} className={row.Method.includes('Proposed') ? 'highlight-row' : ''}>
                <td>
                  <strong>{row.Method}</strong>
                  {row.Note && <div className="sub-note">{row.Note}</div>}
                </td>
                <td>{Number(row.Precision).toFixed(3)}</td>
                <td>{Number(row.Recall).toFixed(3)}</td>
                <td><strong>{Number(row.F1).toFixed(3)}</strong></td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="table-footer-note">Uses the original (frozen) gate.</div>

        {/* Small clean bar chart */}
        <div className="chart-container">
          <div className="chart-title">F1 Score Comparison across Moderation Approaches</div>
          <div className="bar-chart-flex">
            {data.map((row, idx) => {
              const f1Val = Number(row.F1) || 0;
              const heightPct = Math.round(f1Val * 100);
              const isProp = row.Method.includes('Proposed');
              return (
                <div className="bar-item" key={idx}>
                  <div className="bar-wrapper">
                    <div
                      className={`bar-fill ${isProp ? 'bar-proposed' : 'bar-baseline'}`}
                      style={{ height: `${heightPct}%` }}
                    >
                      <span className="bar-val">{f1Val.toFixed(3)}</span>
                    </div>
                  </div>
                  <span className="bar-label">{row.Method}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    );
  };

  const renderTable2 = (data) => {
    return (
      <div className="table-block">
        <table className="benchmark-table">
          <thead>
            <tr>
              <th>Configuration</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1 Score</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row, idx) => (
              <tr key={idx} className={row.Configuration === 'Full' ? 'highlight-row' : ''}>
                <td><strong>{row.Configuration}</strong></td>
                <td>{Number(row.Precision).toFixed(3)}</td>
                <td>{Number(row.Recall).toFixed(3)}</td>
                <td><strong>{Number(row.F1).toFixed(3)}</strong></td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="table-footer-note">
          When a modality is missing, the gate dynamically assigns it ~0% weight and shifts confidence to the remaining modalities.
        </div>
      </div>
    );
  };

  const renderTable3 = (data) => {
    return (
      <div className="table-block">
        <table className="benchmark-table grouped-table">
          <thead>
            <tr>
              <th rowSpan="2">Corruption Condition</th>
              <th colSpan="3" className="col-group">Fixed Weights</th>
              <th colSpan="3" className="col-group">Proposed Gate</th>
            </tr>
            <tr>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row, idx) => (
              <tr key={idx} className={row.Corruption === 'Clean' ? 'highlight-row' : ''}>
                <td><strong>{row.Corruption}</strong></td>
                <td>{Number(row.Fixed_Precision).toFixed(3)}</td>
                <td>{Number(row.Fixed_Recall).toFixed(3)}</td>
                <td>{Number(row.Fixed_F1).toFixed(3)}</td>
                <td>{Number(row.Proposed_Precision).toFixed(3)}</td>
                <td>{Number(row.Proposed_Recall).toFixed(3)}</td>
                <td><strong>{Number(row.Proposed_F1).toFixed(3)}</strong></td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="table-footer-note">
          Corruptions physically applied to test videos: Blur (ksize=9 Gaussian), Darkening (35% lum), Audio Noise (10 dB SNR).
        </div>
      </div>
    );
  };

  const renderTable4 = (data) => {
    return (
      <div className="table-block">
        <table className="benchmark-table grouped-table">
          <thead>
            <tr>
              <th rowSpan="2">Scenario</th>
              <th colSpan="3" className="col-group">Performance</th>
              <th colSpan="3" className="col-group">Assigned Gate Weights</th>
            </tr>
            <tr>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1</th>
              <th>Frames</th>
              <th>Clip</th>
              <th>Speech</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row, idx) => (
              <tr key={idx} className={row.Scenario === 'Full' ? 'highlight-row' : ''}>
                <td><strong>{row.Scenario}</strong></td>
                <td>{Number(row.Precision).toFixed(3)}</td>
                <td>{Number(row.Recall).toFixed(3)}</td>
                <td><strong>{Number(row.F1).toFixed(3)}</strong></td>
                <td>{(Number(row.Weight_Frames) * 100).toFixed(1)}%</td>
                <td>{(Number(row.Weight_Clip) * 100).toFixed(1)}%</td>
                <td>{(Number(row.Weight_Speech) * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="table-footer-note">
          When a modality is missing, the gate dynamically assigns it ~0% weight and shifts confidence to the remaining modalities.
        </div>
      </div>
    );
  };

  const renderTable5 = (data) => {
    return (
      <div className="table-block">
        <table className="benchmark-table">
          <thead>
            <tr>
              <th>Review Rate</th>
              <th>Coverage</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1 Score</th>
            </tr>
          </thead>
          <tbody>
            {data.map((row, idx) => (
              <tr key={idx}>
                <td><strong>{row.Review_Rate}</strong></td>
                <td>{row.Coverage}</td>
                <td>{Number(row.Precision).toFixed(3)}</td>
                <td>{Number(row.Recall).toFixed(3)}</td>
                <td><strong>{Number(row.F1).toFixed(3)}</strong></td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="table-footer-note">
          Ambiguous videos near the decision threshold are routed to human review, improving autonomous accuracy.
        </div>
      </div>
    );
  };

  const currentData = tableData[activeSubTab] || [];

  return (
    <div className="results-wrapper">
      <div className="results-header">
        <h2 className="section-title">Benchmark Results</h2>
        <p className="section-description">
          Empirical evaluation on the unseen test set comparing unimodal baselines, static fusion, and adaptive reliability gating.
        </p>
      </div>

      {/* Sub-Tabs */}
      <div className="subtabs-bar">
        {tableTabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={`subtab-btn ${activeSubTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveSubTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {loading && <div className="loading-state">Loading table metrics...</div>}
      {error && <div className="error-alert">{error}</div>}

      {!loading && !error && currentData.length > 0 && (
        <div className="table-card">
          {activeSubTab === 1 && renderTable1(currentData)}
          {activeSubTab === 2 && renderTable2(currentData)}
          {activeSubTab === 3 && renderTable3(currentData)}
          {activeSubTab === 4 && renderTable4(currentData)}
          {activeSubTab === 5 && renderTable5(currentData)}
        </div>
      )}
    </div>
  );
}

export default ResultsTab;
