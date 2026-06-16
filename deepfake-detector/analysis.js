const MODEL_URL = chrome.runtime.getURL("mesonet.onnx");
ort.env.wasm.wasmPaths = chrome.runtime.getURL("lib/");
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

let modelSession = null;

const WEIGHTS = { model: 0.50, texture: 0.20, geometry: 0.20, metadata: 0.10 };

async function init() {
  let step = "initialising";
  try {
    step = "loading model";
    setLoading("Loading MesoNet model… (~2 s first run)");
    const modelBuffer = await fetch(MODEL_URL).then(r => r.arrayBuffer());
    modelSession = await ort.InferenceSession.create(modelBuffer, { executionProviders: ['wasm'] });

    step = "retrieving image";
    setLoading("Retrieving image…");
    const { dataUrl } = await getStoredImage();
    if (!dataUrl) {
      document.getElementById("loadingSection").classList.add("hidden");
      document.getElementById("errorSection").classList.remove("hidden");
      document.getElementById("errorText").textContent = "No image data — right-click an image and choose 'Check for Deepfake'";
      return;
    }

    // Metadata check happens before face crop (no face needed)
    const metadataResult = analyseMetadata(dataUrl);

    step = "detecting face";
    setLoading("Detecting face region…");
    const img = await loadImage(dataUrl);
    const { canvas: faceCanvas, count: faceCount } = await detectAndCropFace(img);
    const imageData = (faceCanvas || centreCrop(img)).getContext("2d").getImageData(0, 0, 256, 256);

    step = "running AI model";
    setLoading("Running MesoNet inference…");
    const modelResult    = await analyseModel(imageData);
    setLoading("Computing texture & illumination indicators…");
    const textureResult  = analyseTexture(imageData);
    const geometryResult = analyseGeometry(imageData);

    const riskScore =
      WEIGHTS.model    * modelResult.score +
      WEIGHTS.texture  * textureResult.score +
      WEIGHTS.geometry * geometryResult.score +
      WEIGHTS.metadata * metadataResult.score;

    showResults(riskScore, [modelResult, textureResult, geometryResult, metadataResult]);

  } catch (err) {
    console.error("[" + step + "]", err);
    document.getElementById("loadingSection").classList.add("hidden");
    document.getElementById("errorSection").classList.remove("hidden");
    document.getElementById("errorText").textContent = "Failed at [" + step + "]: " + (err.message || String(err));
  }
}

// ── Indicator 1: MesoNet AI model ────────────────────────────── weight: 50%
async function analyseModel(imageData) {
  const W = 256, H = 256;
  const { data } = imageData;
  const tensor = new Float32Array(3 * W * H);
  for (let i = 0; i < W * H; i++) {
    tensor[i]           = data[i * 4]     / 255;
    tensor[i + W * H]   = data[i * 4 + 1] / 255;
    tensor[i + 2 * W*H] = data[i * 4 + 2] / 255;
  }
  const feeds = { [modelSession.inputNames[0]]: new ort.Tensor("float32", tensor, [1, 3, W, H]) };
  const out   = await modelSession.run(feeds);
  const logit = out[modelSession.outputNames[0]].data[0];
  const score = 1 / (1 + Math.exp(-logit));

  let detail;
  if      (score > 0.65) detail = "Strong AI manipulation signal detected by MesoNet";
  else if (score > 0.40) detail = "Moderate manipulation probability — treat with caution";
  else                   detail = "Low manipulation probability detected by model";

  return { icon: "🤖", label: "AI Model Detection", weight: "50%", score, pct: Math.round(score * 100), detail };
}

// ── Indicator 2: Metadata ─────────────────────────────────────── weight: 10%
function analyseMetadata(dataUrl) {
  let score, detail;
  const isJpeg = dataUrl.startsWith("data:image/jpeg") || dataUrl.startsWith("data:image/jpg");

  if (!isJpeg) {
    score  = 0.30;
    detail = "Non-JPEG format — EXIF check not applicable";
  } else {
    // Decode first 64 bytes (88 base64 chars = 22 groups × 4, a multiple of 4).
    // Scan for APP1 marker (0xFF 0xE1 = EXIF). APP1 sits at byte 2 when there
    // is no JFIF prefix, or at ~byte 20 after an APP0 segment — so we scan the
    // window rather than hardcoding a fixed index.
    const bin = atob(dataUrl.split(",")[1].substring(0, 88));
    const validJpeg = bin.charCodeAt(0) === 0xFF && bin.charCodeAt(1) === 0xD8;
    let hasExif = false;
    if (validJpeg) {
      for (let i = 2; i < bin.length - 1; i++) {
        if (bin.charCodeAt(i) === 0xFF && bin.charCodeAt(i + 1) === 0xE1) {
          hasExif = true; break;
        }
      }
    }
    if (hasExif) {
      score  = 0.10;
      detail = "EXIF metadata present — image likely retains original camera data";
    } else {
      score  = 0.70;
      detail = "No EXIF metadata — stripped or generated image, common in deepfake content";
    }
  }

  return { icon: "📋", label: "Metadata Analysis", weight: "10%", score, pct: Math.round(score * 100), detail };
}

// ── Indicator 3: Texture — Laplacian variance ─────────────────── weight: 20%
// Low variance = over-smooth = characteristic of AI face synthesis
function analyseTexture(imageData) {
  const { data, width, height } = imageData;
  const values = [];

  for (let y = 1; y < height - 1; y += 2) {
    for (let x = 1; x < width - 1; x += 2) {
      const g = (px, py) => {
        const i = (py * width + px) * 4;
        return 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      };
      values.push(-4 * g(x, y) + g(x-1, y) + g(x+1, y) + g(x, y-1) + g(x, y+1));
    }
  }

  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length;

  let score, detail;
  if      (variance < 100) { score = 0.85; detail = "Very low sharpness variance — characteristic of AI face smoothing"; }
  else if (variance < 250) { score = 0.55; detail = "Below-average texture variance — possible AI smoothing artifact"; }
  else if (variance < 500) { score = 0.25; detail = "Normal texture sharpness detected"; }
  else                     { score = 0.05; detail = "High texture variance — image appears naturally sharp"; }

  return { icon: "🔬", label: "Texture Analysis", weight: "20%", score, pct: Math.round(score * 100), detail };
}

// ── Indicator 4: Illumination symmetry ───────────────────────── weight: 20%
// Real photos have asymmetric lighting; GAN/swap faces often show unnaturally
// uniform luminance or hard intensity discontinuities at blend boundaries.
function analyseGeometry(imageData) {
  const { data, width, height } = imageData;
  const halfW  = Math.floor(width / 2);
  const yStart = Math.floor(height * 0.20);
  const yEnd   = Math.floor(height * 0.80);
  let diffSum = 0, total = 0;

  for (let y = yStart; y < yEnd; y += 2) {
    for (let x = 0; x < halfW; x += 2) {
      const li = (y * width + x) * 4;
      const ri = (y * width + (width - 1 - x)) * 4;
      const lG = 0.299*data[li]   + 0.587*data[li+1]   + 0.114*data[li+2];
      const rG = 0.299*data[ri]   + 0.587*data[ri+1]   + 0.114*data[ri+2];
      diffSum += Math.abs(lG - rG);
      total++;
    }
  }

  const asymmetry = diffSum / total / 255;  // normalised [0, 1]

  let score, detail;
  if (asymmetry < 0.03) {
    score  = 0.72; detail = "Unnaturally uniform illumination — consistent with GAN face synthesis";
  } else if (asymmetry > 0.14) {
    score  = 0.65; detail = "Elevated illumination asymmetry — possible face-swap boundary mismatch";
  } else {
    score  = 0.18; detail = "Natural illumination distribution within expected range";
  }

  return { icon: "💡", label: "Illumination Symmetry", weight: "20%", score, pct: Math.round(score * 100), detail };
}

// ── UI ────────────────────────────────────────────────────────────────────────

function showResults(riskScore, indicators) {
  document.getElementById("loadingSection").classList.add("hidden");
  document.getElementById("resultSection").classList.remove("hidden");

  const pct    = Math.round(riskScore * 100);
  const banner = document.getElementById("riskBanner");
  banner.className = "risk-banner";

  if (riskScore < 0.35) {
    banner.classList.add("low");
    document.getElementById("riskLabel").textContent = "✅ Low Risk";
    document.getElementById("riskSub").textContent   = "No strong indicators of manipulation";
  } else if (riskScore < 0.60) {
    banner.classList.add("medium");
    document.getElementById("riskLabel").textContent = "⚠️ Medium Risk";
    document.getElementById("riskSub").textContent   = "Some indicators detected — verify source";
  } else {
    banner.classList.add("high");
    document.getElementById("riskLabel").textContent = "🚨 High Risk";
    document.getElementById("riskSub").textContent   = "Multiple deepfake indicators present";
  }
  document.getElementById("riskScore").textContent = pct + "%";

  // Indicator cards
  const container = document.getElementById("indicators");
  container.innerHTML = "";
  indicators.forEach(ind => {
    const level = ind.score > 0.60 ? "high" : ind.score > 0.35 ? "medium" : "low";
    const card  = document.createElement("div");
    card.className = "indicator-card";
    card.innerHTML = `
      <div class="ind-header">
        <span class="ind-icon">${ind.icon}</span>
        <span class="ind-label">${ind.label}</span>
        <span class="ind-weight">${ind.weight}</span>
        <span class="ind-pct ${level}">${ind.pct}%</span>
      </div>
      <div class="ind-bar-track">
        <div class="ind-bar ${level}" style="width:${ind.pct}%"></div>
      </div>
      <div class="ind-detail">${ind.detail}</div>
    `;
    container.appendChild(card);
  });

  // Security context
  document.getElementById("contextText").textContent = buildContext(riskScore, indicators);
}

function buildContext(riskScore, indicators) {
  const flagged = indicators.filter(i => i.score > 0.60).map(i => i.label.toLowerCase());

  if (riskScore < 0.35) {
    return "No significant manipulation indicators detected. Always verify media from untrusted sources before sharing or acting on it.";
  }
  if (riskScore >= 0.60) {
    const reasons = flagged.length ? flagged.join(", ") : "multiple indicators";
    return `High-risk signals detected (${reasons}). Deepfakes are used in disinformation campaigns, identity fraud, and social engineering attacks. Do not share or trust this content without independent verification.`;
  }
  const reasons = flagged.length ? flagged.join(", ") : "AI model score";
  return `Moderate risk detected (${reasons}). Exercise caution — verify the original source before sharing or acting on this content.`;
}

// ── Utilities ─────────────────────────────────────────────────────────────────

// Uses Chrome's built-in Shape Detection API — no external model files needed.
// Available in Chrome 70+ on Windows/macOS. Falls back to centre crop if absent.
async function detectAndCropFace(img) {
  const countEl = document.getElementById("faceCount");
  countEl.classList.remove("hidden");

  if (typeof FaceDetector === "undefined") {
    countEl.className = "face-count warning";
    countEl.textContent = "⚠ Face API unavailable — using centre crop";
    drawPreview(img, []);
    return { canvas: null, count: 0 };
  }

  let faces;
  try {
    const detector = new FaceDetector({ maxDetectedFaces: 10, fastMode: false });
    faces = await detector.detect(img);
  } catch (e) {
    console.warn("FaceDetector failed:", e.message);
    countEl.className = "face-count warning";
    countEl.textContent = "⚠ Detection failed — using centre crop";
    drawPreview(img, []);
    return { canvas: null, count: 0 };
  }

  if (faces.length === 0) {
    countEl.className = "face-count warning";
    countEl.textContent = "⚠ No face detected — using centre crop";
    drawPreview(img, []);
    return { canvas: null, count: 0 };
  }

  countEl.className = faces.length === 1 ? "face-count ok" : "face-count multi";
  countEl.textContent = faces.length === 1
    ? "1 face detected"
    : faces.length + " faces detected — analysing largest";

  // Normalise DOMRect → plain box objects for drawPreview
  const detections = faces.map(f => ({
    box: { x: f.boundingBox.x, y: f.boundingBox.y,
           width: f.boundingBox.width, height: f.boundingBox.height }
  }));
  drawPreview(img, detections);

  // Pick the largest face by area
  const box = detections
    .sort((a, b) => (b.box.width * b.box.height) - (a.box.width * a.box.height))[0].box;

  // 30% padding, then square up so forehead/chin/ears are fully included
  const pad  = 0.30;
  const rx   = Math.max(0, box.x - box.width  * pad);
  const ry   = Math.max(0, box.y - box.height * pad);
  const rw   = Math.min(img.width  - rx, box.width  * (1 + 2 * pad));
  const rh   = Math.min(img.height - ry, box.height * (1 + 2 * pad));
  const side = Math.max(rw, rh);
  const sx   = Math.max(0, rx - (side - rw) / 2);
  const sy   = Math.max(0, ry - (side - rh) / 2);
  const sw   = Math.min(img.width  - sx, side);
  const sh   = Math.min(img.height - sy, side);

  const src = document.createElement("canvas");
  src.width = img.width; src.height = img.height;
  src.getContext("2d").drawImage(img, 0, 0);

  const out = document.createElement("canvas");
  out.width = 256; out.height = 256;
  out.getContext("2d").drawImage(src, sx, sy, sw, sh, 0, 0, 256, 256);

  return { canvas: out, count: faces.length };
}

function drawPreview(img, detections) {
  const preview = document.getElementById("faceCanvas");
  if (!preview) return;
  const scale = Math.min(160 / img.width, 160 / img.height);
  const dw = img.width * scale, dh = img.height * scale;
  const ox = (160 - dw) / 2, oy = (160 - dh) / 2;
  const ctx = preview.getContext("2d");
  ctx.clearRect(0, 0, 160, 160);
  ctx.drawImage(img, ox, oy, dw, dh);
  // Draw a box for each detected face
  detections.forEach((d, i) => {
    ctx.strokeStyle = i === 0 ? "#7c7cff" : "#555";
    ctx.lineWidth   = i === 0 ? 2 : 1;
    ctx.strokeRect(
      ox + d.box.x * scale,
      oy + d.box.y * scale,
      d.box.width  * scale,
      d.box.height * scale
    );
  });
}

function centreCrop(img) {
  const size = Math.min(img.width, img.height) * 0.7;
  const c = document.createElement("canvas");
  c.width = 256; c.height = 256;
  c.getContext("2d").drawImage(img, (img.width - size) / 2, (img.height - size) / 2, size, size, 0, 0, 256, 256);
  drawPreview(img, []);
  return c;
}

function getStoredImage() {
  return new Promise(res => chrome.storage.local.get("pendingImage", d => res({ dataUrl: d.pendingImage })));
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload  = () => resolve(img);
    img.onerror = () => { const i2 = new Image(); i2.onload = () => resolve(i2); i2.onerror = reject; i2.src = src; };
    img.src = src;
  });
}

function setLoading(text) { document.getElementById("loadingText").textContent = text; }

document.addEventListener('DOMContentLoaded', init);
