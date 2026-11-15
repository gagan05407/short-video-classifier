import React, { useState, useRef } from 'react';

/**
 * VideoUpload Component
 * 
 * Clean, minimal drag-and-drop upload zone with standard file browser.
 * Validates MP4, MOV, and AVI formats.
 */
function VideoUpload({ onVideoSelect, disabled }) {
  const [isDragging, setIsDragging] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const fileInputRef = useRef(null);

  const allowedExtensions = ['.mp4', '.mov', '.avi'];
  const allowedMimeTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo'];

  const handleFile = (file) => {
    if (!file) return;

    const lowerName = file.name.toLowerCase();
    const hasValidExt = allowedExtensions.some((ext) => lowerName.endsWith(ext));
    const hasValidMime = allowedMimeTypes.includes(file.type) || file.type.startsWith('video/');

    if (!hasValidExt && !hasValidMime) {
      setErrorMessage('Invalid file format. Please upload an MP4, MOV, or AVI video.');
      return;
    }

    setErrorMessage('');
    onVideoSelect(file);
  };

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) setIsDragging(true);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleInputChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  };

  const handleBrowseClick = () => {
    if (fileInputRef.current && !disabled) {
      fileInputRef.current.value = '';
      fileInputRef.current.click();
    }
  };

  return (
    <div className="upload-wrapper">
      <input
        ref={fileInputRef}
        type="file"
        accept=".mp4,.mov,.avi,video/mp4,video/quicktime,video/x-msvideo"
        onChange={handleInputChange}
        style={{ display: 'none' }}
        disabled={disabled}
      />

      <div
        className={`drop-area ${isDragging ? 'drop-area-active' : ''}`}
        onDragEnter={handleDragEnter}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <p className="drop-title">Drag and drop video here</p>
        <p className="drop-separator">or</p>
        <button
          type="button"
          className="btn btn-secondary"
          onClick={handleBrowseClick}
          disabled={disabled}
        >
          Browse Files
        </button>
        <p className="drop-formats">Supported formats: MP4, MOV, AVI</p>
      </div>

      {errorMessage && (
        <div className="upload-error-message">
          {errorMessage}
        </div>
      )}
    </div>
  );
}

export default VideoUpload;
