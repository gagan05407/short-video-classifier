import axios from 'axios';

const API_BASE = '/api';

export async function analyzeVideo(file) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('video', file);

  const res = await axios.post(`${API_BASE}/analyze`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 120000,
  });
  return res.data;
}

export async function submitFeedback(analysisId, humanLabel) {
  const res = await axios.post(`${API_BASE}/feedback`, {
    analysis_id: analysisId,
    human_label: humanLabel,
  });
  return res.data;
}

export async function getGateStatus() {
  const res = await axios.get(`${API_BASE}/gate/status`);
  return res.data;
}

export async function resetGate() {
  const res = await axios.post(`${API_BASE}/gate/reset`);
  return res.data;
}

export async function getTableResults(tableId) {
  const res = await axios.get(`${API_BASE}/results/${tableId}`);
  return res.data;
}
