import React from 'react';

/**
 * VideoPreview Component
 * 
 * Displays the selected video's filename, formatted size, in-browser preview player,
 * and an option to remove the video.
 */
function VideoPreview({ videoFile, previewUrl, onRemove, disabled }) {
  if (!videoFile || !previewUrl) return null;

  // Format bytes to KB or MB
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="preview-box">
      <div className="preview-file-bar">
        <div className="preview-file-info">
          <span className="preview-filename">{videoFile.name}</span>
          <span className="preview-filesize">({formatFileSize(videoFile.size)})</span>
        </div>
        <button
          type="button"
          className="btn-text-remove"
          onClick={onRemove}
          disabled={disabled}
        >
          Remove Video
        </button>
      </div>

      <div className="preview-player-container">
        <video
          key={previewUrl}
          src={previewUrl}
          controls
          playsInline
          className="preview-video-element"
        >
          Your browser does not support the video tag.
        </video>
      </div>
    </div>
  );
}

export default VideoPreview;
