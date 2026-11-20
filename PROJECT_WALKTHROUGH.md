# BTP Project Walkthrough & Teacher Explanation Guide

> **Purpose:** A short, simple, and explainable summary of the project. Read this before meeting your professor so you can confidently explain **what happens where, why we did it, and how each component works**.

---

## 1. The Big Picture (In 2 Sentences)
Most moderation models fail on short videos (Reels/Shorts) because single frames lack motion context (e.g. gym/dance poses get flagged as nudity) and many videos are silent or blurry.  
Our project solves this using an **Adaptive Reliability Gate** that checks 3 streams (Frames, Video Clip, Speech), measures their quality and uncertainty, and dynamically gives more weight to the cleanest, most reliable stream.

---

## 2. End-to-End Flow of Work (What Happens When a Video is Uploaded)

```
[ Video Upload (.mp4) ]
         │
         ├─── Stream 1: Static Frames (ResNet-18 at 1 FPS)
         │        └── Computes: Frame Risk (p_s), Blur/Brightness (q_s), Uncertainty (u_s)
         │
         ├─── Stream 2: Video Clip (ResNeXt-50 + BiLSTM in 3s windows)
         │        └── Computes: Action Risk (p_t), Motion Continuity (q_t), Uncertainty (u_t)
         │
         ├─── Stream 3: Speech Audio (Whisper STT + Toxic-BERT)
         │        └── Computes: Speech Toxicity (p_sp), Confidence (q_sp), Presence Flag (m_sp)
         │
         ▼
[ 12-Dimensional Feature Vector: p, q, u, m ]
         │
         ▼
[ Adaptive Reliability Gate (MLP) ]
         │
         ▼
[ Dynamic Softmax Weights: w_frames, w_clip, w_speech (Sum = 100%) ]
         │
         ▼
[ Fused Risk Score: p_fused = w_s*p_s + w_t*p_t + w_sp*p_sp ]
         │
         ├── p_fused < 0.30  ──> [ SAFE ] (Auto-approved)
         ├── 0.30 to 0.65    ──> [ SELECTIVE REVIEW ] (Sent to Human Reviewer)
         └── p_fused > 0.65  ──> [ UNSAFE ] (Auto-blocked)
         │
         ▼
[ Live Human Feedback (Optional) ]
         └── Reviewer clicks "Mark Safe" or "Mark Unsafe"
         └── Gate updates in 5 Adam steps with 32 replay samples
         └── Safety Rollback: If validation F1 drops > 0.05, update is rejected!
```

---

## 3. How the 3 Streams Work (Short & Simple)

### Stream 1: Static Frames (ResNet-18)
* **What it does:** Extracts 1 frame per second to look for visual adult content.
* **Problem in old model:** A 1-second awkward camera angle or dance pose in a 30-second video would falsely flag the whole video.
* **Our solution (Temporal Persistence):** A video is only flagged if there is a **sustained streak of $\ge 5$ unsafe frames covering $\ge 30\%$ of the video**, or an uninterrupted streak of $\ge 15$ frames.
* **Quality Check:** Uses Laplacian variance to detect blurry frames, and average pixel brightness to detect dark videos.

### Stream 2: Video Clip Action (ResNeXt-50 + BiLSTM)
* **What it does:** Analyzes motion dynamics in 3-second overlapping windows across 4 classes: `safe`, `dance`, `adult_content`, `violence`.
* **Why LSTM?** The 2-layer Bidirectional LSTM understands movement before and after a frame (e.g. differentiating an athletic jump from a physical fight).
* **Our solution (Two-Tier Persistence):** A single severe peak ($\ge 0.85$) immediately flags violence, while moderate risks require at least 2 consecutive windows of the same class.

### Stream 3: Speech Audio (Whisper + Toxic-BERT)
* **What it does:** Extracts audio at 16 kHz, converts speech to text using OpenAI Whisper, and checks 6 toxic subcategories (insult, threat, obscenity, etc.) using Toxic-BERT.
* **Problem in old model:** Silent videos returned `0.0 toxicity`, which falsely pulled the total risk score down.
* **Our solution (Transparent Handling):** If a video is silent or has no speech, we set its presence flag to $0$. The gate gives it **0.0% weight**, and the UI displays an explicit **"NOT USED / SILENT"** banner.

---

## 4. How the Reliability Gate Works (The Core Theory)

Instead of using fixed static weights (like $50\%$ frames, $30\%$ video, $20\%$ speech):
1. The gate takes 4 indicators for each stream:
   * **$p$ (Risk Prediction):** How dangerous does this stream think the video is? (0 to 1)
   * **$q$ (Quality):** Is the frame blurry or dark? Is the audio noisy? (0 to 1)
   * **$u$ (Uncertainty / Shannon Entropy):** Is the model confident ($u \approx 0$) or confused ($u \approx 1$)?
   * **$m$ (Presence Mask):** Is this stream actually present ($1$) or silent/missing ($0$)?
2. The Gate (a small 2-layer neural network) computes **dynamic weights** ($w_s, w_t, w_{sp}$) that automatically adapt:
   * **If video is blurry:** Gate cuts frames weight to $\approx 5\%$ and shifts to motion and speech.
   * **If video is silent:** Gate sets speech weight to $0.0\%$ and trusts the video clip.

---

## 5. What Procedure We Followed (Offline Development Pipeline)

If your teacher asks *"How did you build and evaluate this project?"*, tell them this 6-step procedure:

1. **Dataset Split (`make_manifest.py`):** Curated videos across 4 classes (`safe`, `dance`, `adult`, `violence`) and split them into 60% Train, 20% Validation, and 20% unseen Test (84 test videos).
2. **Synthetic Speech Muxing (`speech_data/`):** Because short reels are often non-verbal, we paired realistic toxic and neutral conversational sentences using TTS and muxed them into videos with FFmpeg to test speech fusion properly.
3. **Feature Extraction (`extract_features.py`):** Ran all 3 streams on all videos to extract the 12 features into `features.csv`.
4. **Gate Training (`train_gate.py`):** Trained the Reliability Gate across 5 random seeds on the training split using Binary Cross-Entropy loss. Saved the best baseline as `gate_frozen.pt`.
5. **Physical Corruptions Testing (`corrupt.py`):** Injected real physical corruptions into test videos:
   * **Blur:** $9 \times 9$ Gaussian blur filter.
   * **Darkness:** 65% brightness reduction.
   * **Audio Noise:** 10 dB SNR white noise.
6. **Benchmark Generation (`evaluate.py`):** Generated the 5 evaluation tables on the unseen test set.

---

## 6. What the 5 Benchmark Tables Tell the Teacher

* **Table 1 (Main Baselines):**  
  * *What it shows:* Single streams get $0.41 - 0.78$ F1. Fixed weights get $0.776$ F1. Our Proposed Gate achieves **$0.847$ F1** (Precision $0.746$, Recall $0.979$).
  * *What to say:* *"Multimodal fusion with our gate significantly outperforms unimodal models and fixed weights."*
* **Table 2 (Feature Ablation Study):**  
  * *What it shows:* Full model is $0.847$ F1. Without Quality ($q$), F1 drops to $0.826$. Without Uncertainty ($u$), F1 drops to $0.811$. Without both, F1 drops to $0.804$.
  * *What to say:* *"This proves that calculating blur ($q$) and Shannon entropy ($u$) makes the gate smarter than using raw predictions alone."*
* **Table 3 (Physical Corruptions):**  
  * *What it shows:* Under blur and noise, fixed weights collapse, but our gate maintains high performance ($+0.060$ to $+0.071$ advantage).
  * *What to say:* *"When inputs degrade, our gate detects the low quality and downweights the corrupted stream."*
* **Table 4 (Missing Modalities):**  
  * *What it shows:* When audio is dropped, speech weight becomes $0.0\%$ and clip weight rises to $83.0\%$.
  * *What to say:* *"The gate seamlessly handles silent videos without crashing or distorting scores."*
* **Table 5 (Selective Review):**  
  * *What it shows:* Sending ambiguous borderline videos ($0.30 - 0.65$) to human review pushes autonomous F1 from $0.847 \to 0.943$.
  * *What to say:* *"Routing just 20% of hard cases to human review yields near-perfect autonomous accuracy."*

---

## 7. Viva Defense: 5 Quick Questions Teachers Love to Ask

1. **"Why use Shannon Entropy for uncertainty?"**  
   * *Answer:* *"If a model predicts 99% vs 1%, entropy is near 0 (confident). If it predicts 50% vs 50%, entropy is 1.0 (confused). The gate detects this confusion and downweights that stream."*
2. **"Why not train all models end-to-end together?"**  
   * *Answer:* *"End-to-end video backprop requires huge GPU VRAM ($>48$ GB) and causes catastrophic forgetting. Decoupling allows freezing proven backbones while training an ultra-fast, interpretable reliability gate."*
3. **"Why did you use ResNeXt-50 + BiLSTM instead of Video Transformers?"**  
   * *Answer:* *"Video Transformers require millions of clips to avoid overfitting. ResNeXt-50 with BiLSTM provides strong motion inductive bias and runs in real time on standard hardware."*
4. **"What happens if a human reviewer gives bad or wrong feedback?"**  
   * *Answer:* *"Our Safety Rollback Guard evaluates the validation set F1 before applying the update. If validation F1 drops by more than 0.05, the update is automatically rejected and rolled back."*
5. **"What is the difference between Ablation (Table 2) and Missing Modalities (Table 4)?"**  
   * *Answer:* *"Ablation tests our internal feature design (turning off quality $q$ and entropy $u$ to prove they are useful), whereas Missing Modalities tests real-world video uploads (handling silent videos or videos without audio)."*
