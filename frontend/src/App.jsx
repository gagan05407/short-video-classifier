import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import VideoUpload from './components/VideoUpload';
import VideoPreview from './components/VideoPreview';
import AnalysisResults from './components/AnalysisResults';
import ResultsTab from './components/ResultsTab';
import { analyzeVideo } from './services/api';

/**
 * Main Application Component (Simplified 2-Page UI)
 * 1. Video Moderation (Upload, Run Fusion, View Clean Verdict & Streams)
 * 2. Benchmark Results (Tables 1-5 & Evaluation Charts)
 */
function App() {
  const [activeTab, setActiveTab] = useState('analyze');

  // Video analysis state
  const [videoFile, setVideoFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');

  // Memory management for object URLs
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleVideoSelect = (file) => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    const newUrl = URL.createObjectURL(file);
    setVideoFile(file);
    setPreviewUrl(newUrl);
    setAnalysisResult(null);
    setErrorMessage('');
  };

  const handleRemoveVideo = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setVideoFile(null);
    setPreviewUrl(null);
    setAnalysisResult(null);
    setErrorMessage('');
  };

  const handleReset = () => {
    handleRemoveVideo();
  };

  const handleAnalyze = async () => {
    if (!videoFile || isLoading) return;

    setIsLoading(true);
    setErrorMessage('');
    setAnalysisResult(null);

    try {
      const results = await analyzeVideo(videoFile);
      setAnalysisResult(results);
    } catch (err) {
      setErrorMessage(err.message || 'An error occurred during video analysis.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Header activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="main-content">
        <div className="container">
          {/* PAGE 1: VIDEO MODERATION */}
          {activeTab === 'analyze' && (
            <div className="tab-pane">
              {!analysisResult ? (
                <section className="card upload-section">
                  <div className="card-header">
                    <h2 className="section-title">Upload Video for Moderation</h2>
                    <p className="section-description">
                      Upload any video clip to evaluate multimodal safety across visual frames, motion dynamics, and speech acoustics.
                    </p>
                  </div>

                  {!videoFile && (
                    <VideoUpload
                      onVideoSelect={handleVideoSelect}
                      disabled={isLoading}
                    />
                  )}

                  {videoFile && (
                    <div className="video-selected-area">
                      <VideoPreview
                        videoFile={videoFile}
                        previewUrl={previewUrl}
                        onRemove={handleRemoveVideo}
                        disabled={isLoading}
                      />

                      <div className="action-row">
                        <button
                          type="button"
                          className="btn btn-primary"
                          onClick={handleAnalyze}
                          disabled={isLoading}
                        >
                          {isLoading ? (
                            <span className="btn-loading">
                              <span className="spinner-sm"></span>
                              Analyzing Multimodal Signals...
                            </span>
                          ) : (
                            'Analyze Content Safety'
                          )}
                        </button>
                      </div>

                      {isLoading && (
                        <div className="analysis-progress-card">
                          <div className="progress-bar-animated"></div>
                          <div className="progress-status-text">
                            Evaluating static frames (ResNet18), temporal windows, and speech audio...
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {errorMessage && (
                    <div className="error-alert">
                      <span className="error-alert-icon">⚠️</span>
                      <span>{errorMessage}</span>
                    </div>
                  )}
                </section>
              ) : (
                <AnalysisResults
                  results={analysisResult}
                  onReset={handleReset}
                />
              )}
            </div>
          )}

          {/* PAGE 2: BENCHMARK RESULTS & TABLES 1-5 */}
          {activeTab === 'results' && (
            <div className="tab-pane">
              <ResultsTab />
            </div>
          )}
        </div>
      </main>

      <footer className="app-footer">
        <div className="container footer-content">
          <span>Reliability-Aware Multimodal Fusion for Short-Video Moderation &bull; BTP Project</span>
          <span className="footer-status">NVIDIA GPU Accelerated &bull; FastAPI + React</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
