import React, { useState } from 'react';
import { submitFeedback, resetGate } from '../services/api';

function AnalysisResults({ results, onReset }) {
  const [feedbackState, setFeedbackState] = useState({
    submitting: false,
    data: null,
    error: ''
  });
  const [gateStatus, setGateStatus] = useState(results.gate_status || { version: 0, reviews_count: 0 });
  const [showDetails, setShowDetails] = useState({
    static: false,
    temporal: false,
    speech: false
  });
  const [resetMessage, setResetMessage] = useState('');

  const {
    analysis_id,
    video_name,
    verdict,
    fused_score,
    weights = {},
    why_verdict,
    latency,
    streams = {}
  } = results;

  const toggleDetails = (streamKey) => {
    setShowDetails((prev) => ({ ...prev, [streamKey]: !prev[streamKey] }));
  };

  const handleFeedback = async (humanLabel) => {
    if (!analysis_id || feedbackState.submitting) return;

    setFeedbackState({ submitting: true, data: null, error: '' });
    try {
      const resp = await submitFeedback(analysis_id, humanLabel);
      setFeedbackState({ submitting: false, data: resp, error: '' });
      if (resp.version !== undefined) {
        setGateStatus((prev) => ({
          ...prev,
          version: resp.version,
          reviews_count: resp.reviews_count || (prev.reviews_count + 1)
        }));
      }
    } catch (err) {
      setFeedbackState({
        submitting: false,
        data: null,
        error: err.response?.data?.detail || err.message || 'Feedback update failed.'
      });
    }
  };

  const handleResetGate = async () => {
    try {
      const resp = await resetGate();
      setGateStatus({
        version: resp.version || 0,
        reviews_count: resp.reviews_count || 0,
        live_val_f1: resp.live_val_f1
      });
      setFeedbackState({ submitting: false, data: null, error: '' });
      setResetMessage('Live gate reset to frozen baseline (v0).');
      setTimeout(() => setResetMessage(''), 4000);
    } catch (err) {
      alert(`Reset failed: ${err.message}`);
    }
  };

  const renderStreamCard = (key, streamData) => {
    if (!streamData) return null;
    const isDisagreed = Boolean(streamData.disagrees);
    const isNoAudio = key === 'speech' && streamData.present === 0;

    return (
      <div className={`stream-card ${isDisagreed ? 'disagreed' : ''}`} key={key}>
        <div className="stream-header">
          <span className="stream-name">{streamData.name}</span>
          <span className={`badge ${isNoAudio ? 'badge-neutral' : `badge-${streamData.verdict.toLowerCase().replace(/\s+/g, '-')}`}`}>
            {isNoAudio ? 'NOT USED' : streamData.verdict}
          </span>
        </div>

        <div className="stream-body">
          <div className="stream-metric">
            <span className="label">Risk Score</span>
            <span className="value">{(streamData.score * 100).toFixed(1)}%</span>
          </div>
          <div className="stream-metric">
            <span className="label">Gate Weight</span>
            <span className="value">{(streamData.weight * 100).toFixed(1)}%</span>
          </div>

          {isNoAudio && (
            <div className="no-audio-banner">
              Audio track absent or silent &mdash; Gate dynamically assigned 0.0% weight.
            </div>
          )}

          {isDisagreed && streamData.disagree_reason && (
            <div className="disagree-reason-box">
              {streamData.disagree_reason}
            </div>
          )}

          <div className="details-toggle-row">
            <button
              type="button"
              className="btn-text"
              onClick={() => toggleDetails(key)}
            >
              {showDetails[key] ? 'Hide details' : 'Show details'}
            </button>
          </div>

          {showDetails[key] && (
            <div className="stream-details-expanded">
              <div className="detail-item">
                <span>Quality (q):</span>
                <span>{streamData.quality !== undefined ? streamData.quality.toFixed(3) : '-'}</span>
              </div>
              <div className="detail-item">
                <span>Uncertainty (u):</span>
                <span>{streamData.uncertainty !== undefined ? streamData.uncertainty.toFixed(3) : '-'}</span>
              </div>

              {key === 'static' && (
                <>
                  <div className="detail-item">
                    <span>Max Streak:</span>
                    <span>{streamData.max_consecutive || 0} frames in a row ({streamData.flagged_percent || 0}% total)</span>
                  </div>
                  {streamData.flag_reason && streamData.flag_reason !== 'clean' && (
                    <div className="detail-item">
                      <span>Reason:</span>
                      <span className="capitalize">{streamData.flag_reason.replace(/_/g, ' ')}</span>
                    </div>
                  )}
                </>
              )}

              {key === 'temporal' && (
                <>
                  {streamData.top_class && (
                    <div className="detail-item">
                      <span>Top Class:</span>
                      <span className="capitalize">{streamData.top_class}</span>
                    </div>
                  )}
                  {streamData.total_windows !== undefined && streamData.total_windows > 0 && (
                    <div className="detail-item">
                      <span>Window Streak:</span>
                      <span>
                        {streamData.max_consecutive_windows || 0} in a row ({streamData.total_windows} windows total)
                      </span>
                    </div>
                  )}
                </>
              )}

              {key === 'speech' && (
                <div className="speech-detail-block">
                  <div className="detail-item">
                    <span>Language:</span>
                    <span>{streamData.language || 'none'}</span>
                  </div>
                  <div className="transcript-box">
                    <strong>Transcript:</strong> {streamData.transcript || 'No speech detected'}
                  </div>

                  {streamData.toxic_bert_categories && Object.keys(streamData.toxic_bert_categories).length > 0 && (
                    <div className="toxicity-pills-container">
                      <span className="pills-title">Toxicity Sub-Categories:</span>
                      <div className="pills-flex">
                        {Object.entries(streamData.toxic_bert_categories).map(([cat, val]) => (
                          <span
                            className={`pill ${val >= 0.20 ? 'pill-flagged' : 'pill-neutral'}`}
                            key={cat}
                          >
                            {cat.replace(/_/g, ' ')}: {(val * 100).toFixed(0)}%
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {streamData.roberta_offensive !== undefined && streamData.roberta_offensive > 0 && (
                    <div className="detail-item" style={{ marginTop: '8px' }}>
                      <span>RoBERTa Offensive:</span>
                      <span>{(streamData.roberta_offensive * 100).toFixed(1)}%</span>
                    </div>
                  )}
                </div>
              )}

              {key === 'static' && streamData.top_frames && streamData.top_frames.length > 0 && (
                <div className="top-frames-row">
                  <div className="frames-label">High-scoring frames:</div>
                  <div className="thumbnails-flex">
                    {streamData.top_frames.map((tf, i) => (
                      <div className="thumb-item" key={i}>
                        <img src={tf.image_b64} alt={`Frame ${tf.timestamp}s`} />
                        <span>{tf.timestamp}s ({(tf.score * 100).toFixed(0)}%)</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };

  const currentVerdict = feedbackState.data?.after?.verdict || verdict;
  const currentScore = feedbackState.data?.after?.fused_score !== undefined
    ? feedbackState.data.after.fused_score
    : fused_score;

  return (
    <div className="results-container">
      {/* Top Status Bar */}
      <div className="gate-status-bar">
        <div className="status-meta">
          <span>Gate Version: <strong>v{gateStatus.version}</strong></span>
          <span className="dot-sep">&bull;</span>
          <span>Reviews Logged: <strong>{gateStatus.reviews_count}</strong></span>
          {gateStatus.live_val_f1 !== undefined && (
            <>
              <span className="dot-sep">&bull;</span>
              <span>Val F1: <strong>{gateStatus.live_val_f1.toFixed(3)}</strong></span>
            </>
          )}
        </div>
        <div className="status-actions">
          <button type="button" className="btn btn-secondary btn-sm" onClick={handleResetGate}>
            Reset to Frozen Gate
          </button>
        </div>
      </div>

      {resetMessage && <div className="info-alert">{resetMessage}</div>}

      {/* Main Verdict Card */}
      <div className={`verdict-card verdict-${currentVerdict.toLowerCase()}`}>
        <div className="verdict-main">
          <div className="verdict-label-group">
            <span className="verdict-title">{currentVerdict}</span>
            <span className="verdict-score">Fused Risk: {(currentScore * 100).toFixed(1)}%</span>
          </div>
          <div className="verdict-meta">
            <span>Video: {video_name}</span>
            <span className="dot-sep">&bull;</span>
            <span>Latency: {latency}s</span>
          </div>
        </div>

        <p className="verdict-why">{why_verdict}</p>

        {/* Human Review Actions */}
        <div className="review-action-section">
          <div className="review-action-label">Submit Human Moderation Review:</div>
          <div className="review-btn-group">
            <button
              type="button"
              className="btn btn-safe"
              disabled={feedbackState.submitting}
              onClick={() => handleFeedback('SAFE')}
            >
              Mark Safe
            </button>
            <button
              type="button"
              className="btn btn-unsafe"
              disabled={feedbackState.submitting}
              onClick={() => handleFeedback('UNSAFE')}
            >
              Mark Unsafe
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={onReset}
            >
              Upload Another Video
            </button>
          </div>
        </div>
      </div>

      {/* Feedback Update Feedback / Comparison Card */}
      {feedbackState.submitting && (
        <div className="info-alert">Updating live gate with 5 Adam steps...</div>
      )}

      {feedbackState.error && (
        <div className="error-alert">{feedbackState.error}</div>
      )}

      {feedbackState.data && (
        <div className="comparison-card">
          <div className="comparison-header">
            <strong>Gate Adaptation Result (v{feedbackState.data.version})</strong>
            <span className={feedbackState.data.verdict_changed ? 'badge badge-warning' : 'badge badge-neutral'}>
              {feedbackState.data.verdict_changed ? 'Verdict Changed' : 'Verdict Unchanged'}
            </span>
          </div>
          <p className="comparison-msg">{feedbackState.data.message}</p>

          <table className="comparison-table">
            <thead>
              <tr>
                <th>State</th>
                <th>Verdict</th>
                <th>Fused Score</th>
                <th>Frames Weight</th>
                <th>Clip Weight</th>
                <th>Speech Weight</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Before Review</td>
                <td>{feedbackState.data.before.verdict}</td>
                <td>{(feedbackState.data.before.fused_score * 100).toFixed(1)}%</td>
                <td>{(feedbackState.data.before.weights.static * 100).toFixed(1)}%</td>
                <td>{(feedbackState.data.before.weights.temporal * 100).toFixed(1)}%</td>
                <td>{(feedbackState.data.before.weights.speech * 100).toFixed(1)}%</td>
              </tr>
              <tr className="highlight-row">
                <td>After Review</td>
                <td>{feedbackState.data.after.verdict}</td>
                <td>{(feedbackState.data.after.fused_score * 100).toFixed(1)}%</td>
                <td>{(feedbackState.data.after.weights.static * 100).toFixed(1)}%</td>
                <td>{(feedbackState.data.after.weights.temporal * 100).toFixed(1)}%</td>
                <td>{(feedbackState.data.after.weights.speech * 100).toFixed(1)}%</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}

      {/* Stream Cards Breakdown */}
      <div className="section-title-sm">Multimodal Stream Breakdown</div>
      <div className="streams-grid">
        {renderStreamCard('static', streams.static)}
        {renderStreamCard('temporal', streams.temporal)}
        {renderStreamCard('speech', streams.speech)}
      </div>
    </div>
  );
}

export default AnalysisResults;
