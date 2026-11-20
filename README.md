# Reliability-Aware Multimodal Short-Video Classifier

A robust, real-time multimodal short-video classification and moderation system that dynamically combines **Static Frame Visuals**, **Temporal Motion Dynamics**, and **Speech Audio** using a learned **Adaptive Reliability Gate**. 

The system is designed specifically for fast-paced short-form videos (Reels, Shorts, TikToks). It handles real-world modality degradation (blurry footage, poor lighting, noisy audio), accounts for missing modalities (silent videos, music overlays), and supports **Live Human-in-the-Loop Review** with safe on-the-fly gate adaptation and automatic rollback guards.

---

## Table of Contents
1. [Theoretical Background & Problem Statement](#1-theoretical-background--problem-statement)
2. [System Architecture](#2-system-architecture)
3. [Unimodal Streams & Decision Logic](#3-unimodal-streams--decision-logic)
   - [Static Frame Stream (ResNet-18)](#stream-1-static-frame-stream-resnet-18)
   - [Temporal Action Stream (ResNeXt-50 + BiLSTM)](#stream-2-temporal-action-stream-resnext-50--bilstm)
   - [Speech Audio Stream (Whisper + Toxic-BERT)](#stream-3-speech-audio-stream-whisper--toxic-bert)
4. [Adaptive Reliability Gate & Human Review](#4-adaptive-reliability-gate--human-review)
5. [End-to-End Experimental Procedure](#5-end-to-end-experimental-procedure)
6. [Empirical Benchmark Results](#6-empirical-benchmark-results)
7. [System Requirements & Installation](#7-system-requirements--installation)
8. [How to Run (Quickstart & Manual)](#8-how-to-run-quickstart--manual)
9. [Project Directory Layout](#9-project-directory-layout)
10. [Git Setup & Remote Upload](#10-git-setup--remote-upload)

---

## 1. Theoretical Background & Problem Statement

Short-form videos present unique moderation challenges that make traditional unimodal or naive fusion classifiers fail:

1. **Context & Kinematic Ambiguity:** Single frames often appear provocative (e.g., gymnastics, beach volleyball, traditional dancing) while the full temporal movement is completely benign. Conversely, brief, high-velocity acts of violence occur in split-second bursts that temporal downsampling can easily overlook.
2. **Missing & Degraded Modalities:** A vast proportion of short videos are either completely silent, soundtracked by unrelated trending background music, or shot in adverse lighting with severe motion blur and camera shake.
3. **The Failure of Fixed-Weight Fusion:** Standard ensemble models apply fixed weights ($w_s = 0.50, w_t = 0.30, w_{sp} = 0.20$). When a video has noisy audio or blurry frames, fixed weights force the system to trust degraded signals, causing false alarms or missed violations.
4. **The Reliability Gate Solution:** Instead of assigning static importance, our system dynamically computes **Quality Indicators ($q$)**, **Prediction Uncertainties ($u$)**, and **Modality Presence Flags ($m$)** alongside raw predictions ($p$). An Adaptive Reliability Gate (MLP) dynamically re-allocates weights in real time:

$$\mathbf{x} = \big[ p_s, q_s, u_s, m_s, \; p_t, q_t, u_t, m_t, \; p_{sp}, q_{sp}, u_{sp}, m_{sp} \big]^T \in \mathbb{R}^{12}$$

$$\mathbf{w} = \text{Softmax}\big( G_{\theta}(\mathbf{x}) \big) = [w_s, w_t, w_{sp}], \quad \sum_{i} w_i = 1$$

$$p_{\text{fused}} = w_s p_s + w_t p_t + w_{sp} p_{sp}$$

If video frames are blurred or speech is silent, the gate immediately downweights them and transfers decision authority to the surviving, high-confidence stream.

---

## 2. System Architecture

```
                                  Input Video (MP4)
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            ▼                             ▼                             ▼
┌───────────────────────┐   ┌───────────────────────────┐   ┌───────────────────────┐
│  Stream 1: Frames     │   │  Stream 2: Temporal Clip  │   │  Stream 3: Speech     │
│  ResNet-18 (1 FPS)    │   │  ResNeXt50 + BiLSTM       │   │  Whisper + Toxic-BERT │
│                       │   │  (3.0s sliding windows)   │   │  + Twitter-RoBERTa    │
│  • Risk: p_s          │   │  • Risk: p_t              │   │  • Risk: p_sp         │
│  • Quality: q_s       │   │  • Quality: q_t           │   │  • Quality: q_sp      │
│  • Uncertainty: u_s   │   │  • Uncertainty: u_t       │   │  • Uncertainty: u_sp  │
│  • Presence: m_s      │   │  • Presence: m_t          │   │  • Presence: m_sp     │
└───────────┬───────────┘   └─────────────┬─────────────┘   └───────────┬───────────┘
            │                             │                             │
            └─────────────────────────────┼─────────────────────────────┘
                                          ▼
                      ┌───────────────────────────────────────┐
                      │    Adaptive Reliability Gate (MLP)    │
                      │  Inputs: 12-dim multimodal vector     │
                      │  Hidden: 32 units, BatchNorm, Dropout │
                      │  Outputs: [w_static, w_clip, w_speech]│
                      └───────────────────┬───────────────────┘
                                          ▼
                               Fused Risk Score & Policy
                          p_fused = Σ w_i * p_i
                                          │
                  ┌───────────────────────┼───────────────────────┐
                  ▼                       ▼                       ▼
            p_fused < 0.30        0.30 <= p <= 0.65         p_fused > 0.65
               [ SAFE ]          [ SELECTIVE REVIEW ]        [ UNSAFE ]
                                          │
                                          ▼
                                Live Human Reviewer
                                          │
                          ┌───────────────┴───────────────┐
                          │  5 Adam optimization steps    │
                          │  3x sample loss + 32 replay   │
                          │  Automatic Rollback if F1 < -0.05
                          └───────────────────────────────┘
```

---

## 3. Unimodal Streams & Decision Logic

### Stream 1: Static Frame Stream (ResNet-18)
* **Backbone:** ResNet-18 fine-tuned for visual adult content detection (`nsfw_resnet18_small_state.pth`).
* **Frame Extraction:** Sampled uniformly at 1 FPS across the entire video.
* **Temporal Persistence Condition:** Isolated frames with brief ambiguous exposure do not trigger an unsafe verdict unless verified by sustained continuity:
  - **Condition A (Sustained Streak & Coverage):** A consecutive streak $\ge 5$ unsafe frames AND an overall unsafe frame coverage $\ge 30\%$.
  - **Condition B (Severe Streak):** A continuous streak $\ge 15$ consecutive unsafe frames (regardless of total duration).
* **Quality Indicator ($q_s$):** Evaluated per frame using the Laplacian operator variance for sharpness ($\text{Var}(\nabla^2 I)$) and mean pixel luminance ($L = 0.299R + 0.587G + 0.114B$).
* **Uncertainty ($u_s$):** Normalized binary Shannon entropy:
  $$u_s = -\frac{p_s \ln(p_s) + (1 - p_s)\ln(1 - p_s)}{\ln(2)}$$

### Stream 2: Temporal Action Stream (ResNeXt-50 + BiLSTM)
* **Backbone:** ResNeXt-50 2D/3D CNN spatial backbone connected to a 2-layer Bidirectional LSTM temporal model (`best_model.pth`).
* **Classification Classes:** 4 distinct action categories: `safe`, `dance`, `adult_content`, `violence`.
* **Sliding Window:** 3.0-second sliding windows with 1.5-second stride (50% overlap).
* **Multi-Window Temporal Persistence Condition:**
  - **Condition A (Severe Window Peak):** If any single 3-second window reaches a severe risk peak $p_{\text{win}} \ge 0.85$, it immediately overrides to unsafe.
  - **Condition B (Class-Consistent Continuity):** Consecutive windows of the **same unsafe action class** $\ge 2$ windows OR an overall unsafe window coverage $\ge 30\%$.
* **Quality Indicator ($q_t$):** Inter-window frame difference consistency measuring motion continuity across sequential temporal slices.
* **Uncertainty ($u_t$):** 4-class categorical Shannon entropy normalized by $\ln(4)$.

### Stream 3: Speech Audio Stream (Whisper + Toxic-BERT)
* **Speech-to-Text:** OpenAI Whisper (`base` model) extracts timestamped transcripts and per-segment acoustic log-probability confidence.
* **Toxicity Analysis:** HuggingFace `unitary/toxic-bert` multi-label transformer scoring 6 subcategories:
  - `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.
* **Offensive Language Cross-Check:** `cardiffnlp/twitter-roberta-base-offensive`.
* **Silent & Audio-Free Videos:** If the video has no audio track or the track is silent:
  - Presence flag is set to $m_{sp} = 0$.
  - Acoustic quality is set to $q_{sp} = 0$.
  - UI displays a clear **`NOT USED / SILENT`** banner and pill, avoiding misinterpreting silence as proof of safety.

---

## 4. Adaptive Reliability Gate & Human Review

### The 12-Dimensional Feature Representation
For each video, the feature extractor builds:
$$\mathbf{x} = [p_s, q_s, u_s, m_s, \; p_t, q_t, u_t, m_t, \; p_{sp}, q_{sp}, u_{sp}, m_{sp}]$$

### Gate Architecture
* **Input Layer:** 12 features.
* **Hidden Layer:** 32 units with ReLU activation, Batch Normalization, and Dropout ($p = 0.20$).
* **Output Layer:** 3 units with Softmax activation yielding $\mathbf{w} = [w_s, w_t, w_{sp}]$.

### Decision Policy Bands
* **$\mathbf{p_{\text{fused}} < 0.30}$:** **SAFE** (Autonomous approval).
* **$\mathbf{0.30 \le p_{\text{fused}} \le 0.65}$:** **SELECTIVE REVIEW** (Routed to human reviewer).
* **$\mathbf{p_{\text{fused}} > 0.65}$:** **UNSAFE** (Autonomous flag / rejection).

### Live Human Review & Safety Rollback Guard
When an ambiguous or misclassified video is inspected in the UI:
1. Reviewer clicks **"Mark Safe"** or **"Mark Unsafe"**.
2. The backend (`LiveGateManager`) clones the live gate checkpoint and performs **5 Adam gradient steps** ($\eta = 10^{-3}$).
3. **Loss Function:** Evaluates a weighted combination of the user's feedback sample ($3\times$ loss weight) plus a replay buffer of **32 historical validation samples** to prevent catastrophic forgetting.
4. **Safety Rollback Guard:** Before deploying the updated checkpoint (`current.pt`), the model is evaluated against the validation set:
   - If validation $F_1$ drops by more than $0.05$ (i.e., $F_1 < 0.794$), the update is **rejected** and immediately rolled back to the prior checkpoint.
5. Reviewers can reset to the original baseline at any time with the **"Reset Gate"** button.

---

## 5. End-to-End Experimental Procedure

To build, train, corrupt, and validate this system, the following 6-step procedure was executed:

```
[ Step 1: Manifest ] ──> [ Step 2: Speech Muxing ] ──> [ Step 3: Feature Extraction ]
                                                                 │
[ Step 6: Evaluation ] <── [ Step 5: Corruptions ] <── [ Step 4: Gate Training ]
```

1. **Step 1: Manifest & Split Creation (`backend/pipeline/make_manifest.py`)**
   - Curated videos across 4 balanced categories: `safe`, `dance`, `adult_content`, `violence`.
   - Stratified the dataset into canonical train ($60\%$), validation ($20\%$), and unseen test ($20\%$) splits.
2. **Step 2: Synthetic Speech Generation (`backend/pipeline/make_manifest.py`)**
   - Since raw source reels were predominantly silent or music-overlaid, realistic toxic and neutral conversational sentences were generated (`speech_texts.csv`) and muxed into test videos via `pyttsx3` and FFmpeg.
3. **Step 3: Multimodal Feature Extraction (`backend/pipeline/extract_features.py`)**
   - Executed the ResNet18, ResNeXt50+BiLSTM, and Whisper+Toxic-BERT streams on all videos.
   - Saved the full 12-dimensional feature representations to `backend/data/features.csv`.
4. **Step 4: Gate Training (`backend/pipeline/train_gate.py`)**
   - Trained the ReliabilityGate MLP across 5 distinct random seeds on the training split using Binary Cross Entropy.
   - Saved the frozen baseline model checkpoint to `backend/weights/gate_frozen.pt` (mean test F1: $0.843 \pm 0.008$).
5. **Step 5: Physical Video Corruptions (`backend/pipeline/corrupt.py`)**
   - Applied three physical corruptions to the unseen test videos:
     - **Blur:** $9 \times 9$ Gaussian blur filter applied to all frames.
     - **Darkening:** 65% luminance attenuation ($0.35 \times$ pixel intensity).
     - **Audio Noise:** Injected white Gaussian noise at 10 dB Signal-to-Noise Ratio (SNR).
   - Re-extracted features under each corruption to evaluate real-world robustness.
6. **Step 6: Empirical Benchmark Evaluation (`backend/pipeline/evaluate.py`)**
   - Generated evaluation tables comparing all baselines, ablations, corruptions, and human review coverage.

---

## 6. Empirical Benchmark Results

### Table 1: Main Moderation Benchmark (Unseen Test Set)
Evaluation on the 84-video unseen test set comparing unimodal baselines, naive heuristics, static fusion, and the proposed Adaptive Reliability Gate:

| Method | Precision | Recall | F1 Score | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Frames only (ResNet-18)** | 0.623 | 0.896 | 0.735 | High false alarm rate on provocative dance poses |
| **Clip only (ResNeXt-50+BiLSTM)** | 0.688 | 0.917 | 0.786 | Misses brief visual-only violations |
| **Speech only (Whisper+BERT)** | 0.737 | 0.292 | 0.418 | Low recall due to non-verbal / silent video clips |
| **OR Fusion** (Any stream flags unsafe) | 0.608 | 1.000 | 0.756 | Catches all violations but triggers massive false positives |
| **Fixed Weights** ($0.50 / 0.30 / 0.20$) | 0.662 | 0.938 | 0.776 | Degrades when any individual modality is corrupted |
| **Proposed Reliability Gate** | **0.746** | **0.979** | **0.847** | **5-seed mean: 0.843 ± 0.008** |

### Table 2: Ablation Study
Ablation analysis demonstrating the critical contribution of quality indicators ($q$) and prediction uncertainty ($u$):

| Configuration | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: |
| **Full Proposed Model** | **0.746** | **0.979** | **0.847** |
| **Without Quality Indicators ($q$)** | 0.738 | 0.938 | 0.826 |
| **Without Prediction Uncertainty ($u$)** | 0.714 | 0.938 | 0.811 |
| **Without Quality & Uncertainty** (Predictions only) | 0.703 | 0.938 | 0.804 |

### Table 3: Robustness under Physical Corruptions
Comparison of Fixed Weights vs. the Proposed Adaptive Reliability Gate under real-world degradation:

| Corruption Condition | Fixed Weights F1 | Proposed Gate F1 | Gate Advantage | Gate Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| **Clean Video** | 0.776 | **0.847** | $+0.071$ | Optimal multimodal confidence weighting |
| **Blur** ($9 \times 9$ Gaussian) | 0.719 | **0.779** | $+0.060$ | Gate detects low Laplacian variance and downweights frames |
| **Darkening** (65% brightness reduction) | 0.797 | **0.814** | $+0.017$ | Gate shifts reliance to temporal motion and speech |
| **Audio Noise** (10 dB SNR white noise) | 0.776 | **0.847** | $+0.071$ | Gate recognizes low Whisper confidence and silences speech weight |

### Table 4: Selective Human Review Trade-Offs
Routing ambiguous borderline predictions ($0.30 \le p_{\text{fused}} \le 0.65$) to human reviewers yields substantial performance gains on autonomous verdicts:

| Human Review Rate | Autonomous Coverage | Precision | Recall | Autonomous F1 Score |
| :---: | :---: | :---: | :---: | :---: |
| **0%** (Full Autonomous) | 100.0% | 0.746 | 0.979 | 0.847 |
| **10%** | 90.0% | 0.778 | 0.977 | 0.866 |
| **20%** | 80.0% | 0.816 | 0.976 | 0.889 |
| **30%** | 70.0% | 0.860 | 0.974 | 0.914 |

---

## 7. System Requirements & Installation

### Prerequisites
* **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS
* **Python:** 3.10 or 3.11
* **Node.js:** 18.x or 20.x and npm
* **FFmpeg:** Installed and available in your system `PATH`
* **GPU (Optional):** NVIDIA GPU with CUDA support for accelerated inference

### Step 1: Clone the Repository
```bash
git clone https://github.com/gagan05407/btp_final.git
cd btp_final
```

### Step 2: Python Environment Setup
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
# source venv/bin/activate

pip install -r backend/requirements.txt
```

### Step 3: Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

### Step 4: Model Checkpoints
Ensure the model weight checkpoints are placed in `backend/weights/`:
- `gate_frozen.pt` (Included in repo)
- `gate_config.json` (Included in repo)
- `best_model.pth` (~160 MB, temporal ResNeXt50+BiLSTM)
- `nsfw_resnet18_small_state.pth` (~45 MB, static ResNet18)

---

## 8. How to Run (Quickstart & Manual)

### Method A: One-Click Launch (Recommended)
On Windows, simply double-click **`run.bat`** in the project root:
```bat
run.bat
```
This automatically launches both the FastAPI backend (`http://127.0.0.1:8000`) and the Vite React frontend (`http://localhost:3000`).

---

### Method B: Manual Step-by-Step Launch

#### 1. Start the Backend API
In your first terminal:
```bash
cd btp_final
# Activate venv:
venv\Scripts\activate
cd backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
You should see:
```
INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO: Application startup complete.
```

#### 2. Start the Frontend UI
In a second terminal:
```bash
cd btp_final/frontend
npm run dev -- --port 3000
```
You should see:
```
VITE v6.4.6  ready in ~250 ms
➜  Local:   http://localhost:3000/
```

#### 3. Open in Browser
Open Google Chrome and navigate to **`http://localhost:3000`**.  
*(Note: If you have an open session, press `Ctrl + F5` to perform a hard refresh).*

---

## 9. Project Directory Layout

```
btp_final/
├── backend/
│   ├── app.py                     # FastAPI server (analyze, feedback, reset, results endpoints)
│   ├── config.py                  # Centralized configuration (hyperparameters, paths, thresholds)
│   ├── gate.py                    # ReliabilityGate MLP, LiveGateManager, safety rollback logic
│   ├── requirements.txt           # Python package requirements
│   │
│   ├── streams/                   # Three unimodal processing streams
│   │   ├── static.py              # ResNet-18 frame sampler, blur/brightness indicators
│   │   ├── temporal.py            # ResNeXt50+BiLSTM sliding-window action classifier
│   │   ├── speech.py              # Whisper STT + Toxic-BERT + Twitter-RoBERTa + acoustic quality
│   │   └── video_classifier.py    # Temporal neural network architecture definition
│   │
│   ├── pipeline/                  # Offline training & empirical evaluation scripts
│   │   ├── make_manifest.py       # Stratified split manifest generator & synthetic speech muxer
│   │   ├── extract_features.py    # Multimodal feature extraction pipeline
│   │   ├── train_gate.py          # 5-seed Reliability Gate training
│   │   ├── corrupt.py             # Physical video corruption engine (Blur, Dark, Noise)
│   │   └── evaluate.py            # Generates benchmark Tables 1 to 4
│   │
│   ├── data/                      # Manifests and feature tables
│   │   ├── manifest.csv           # Stratified train/val/test video manifest
│   │   ├── features.csv           # Extracted 12-dim multimodal feature matrix
│   │   ├── speech_texts.csv       # Source transcripts for speech evaluation
│   │   └── features_corrupted_*.csv# Features extracted under Blur, Dark, and Noise
│   │
│   ├── weights/                   # Neural network weights
│   │   ├── gate_frozen.pt         # Frozen baseline Reliability Gate checkpoint (tracked)
│   │   ├── gate_config.json       # Normalization statistics and architecture config
│   │   ├── best_model.pth         # Temporal action classifier checkpoint (~160 MB)
│   │   └── nsfw_resnet18_small_state.pth # ResNet-18 visual checkpoint (~45 MB)
│   │
│   └── results/                   # Benchmark CSV results served to frontend
│       ├── table1.csv             # Main baselines comparison
│       ├── table2.csv             # Ablation study
│       ├── table3.csv             # Corruption robustness
│       ├── table5.csv             # Selective review trade-offs
│       └── feedback_log.csv       # Audit log of live human corrections
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.jsx          # Top navigation header ("Short-Video Classifier")
│   │   │   ├── VideoUpload.jsx     # Video uploader and sample test selector
│   │   │   ├── AnalysisResults.jsx # Stream breakdown cards, diagnostics, human review controls
│   │   │   └── ResultsTab.jsx      # Benchmark Tables 1-4 and F1 comparison bar chart
│   │   ├── services/
│   │   │   └── api.js              # Axios HTTP client connecting to FastAPI backend
│   │   ├── App.jsx                 # Main React application shell
│   │   ├── index.css               # Clean styling
│   │   └── main.jsx
│   ├── index.html                  # HTML root with updated page title
│   ├── package.json
│   └── vite.config.js
│
├── .gitignore                     # Ignores venv, node_modules, temp uploads, >100MB binaries
├── run.bat                        # Automated one-click launcher for Windows
└── README.md                      # Comprehensive system documentation
```

---

## 10. Git Setup & Remote Upload

To initialize and push this repository to your GitHub account (`gagan05407`):

### 1. Initialize Git & Create Main Branch
In the project root `D:\project\btp_final`:
```bash
git init
git branch -M main
```

### 2. Stage Tracked Files
The provided `.gitignore` automatically excludes `venv/`, `frontend/node_modules/`, `backend/temp_uploads/`, and large binaries (>100 MB).
```bash
git add .
git status
```

### 3. Commit the Project
```bash
git commit -m "Initial commit: Reliability-Aware Multimodal Short-Video Classifier with Live Adaptive Gating"
```

### 4. Create Repository on GitHub & Push
1. Go to [GitHub - Create New Repository](https://github.com/new).
2. Name the repository: `btp_final` (or `short-video-classifier`).
3. Leave "Initialize with README" **unchecked** (we already have a complete README).
4. Run the following commands in your terminal:
```bash
git remote add origin https://github.com/gagan05407/<YOUR_REPO_NAME>.git
git push -u origin main
```
