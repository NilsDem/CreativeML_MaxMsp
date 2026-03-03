


import { MnistData } from "./data.js";
import {
  LATENT_DIM,
  decodePoint,
  encodeTestSetSample,
  loadVAE,
  saveVAE,
  trainVAE,
} from "./vae_train.js";

const ENCODED_FALLBACK_PATH = "./encoded_reference.json";

const refs = {
  loadDataBtn: document.getElementById("loadDataBtn"),
  trainBtn: document.getElementById("trainBtn"),
  saveBtn: document.getElementById("saveBtn"),
  loadSavedBtn: document.getElementById("loadSavedBtn"),
  statusText: document.getElementById("statusText"),
  trainProgress: document.getElementById("trainProgress"),
  progressText: document.getElementById("progressText"),
  log: document.getElementById("log"),
  latentCanvas: document.getElementById("latentCanvas"),
  decodeCanvas: document.getElementById("decodeCanvas"),
  epochInputCanvas: document.getElementById("epochInputCanvas"),
  epochReconCanvas: document.getElementById("epochReconCanvas"),
  epochPreviewText: document.getElementById("epochPreviewText"),
  hoverText: document.getElementById("hoverText"),
  epochsInput: document.getElementById("epochsInput"),
  stepsInput: document.getElementById("stepsInput"),
  batchInput: document.getElementById("batchInput"),
  lrInput: document.getElementById("lrInput"),
  klInput: document.getElementById("klInput"),
  plotCountInput: document.getElementById("plotCountInput"),
};

const latentCtx = refs.latentCanvas.getContext("2d");
const decodeCtx = refs.decodeCanvas.getContext("2d");
const epochInputCtx = refs.epochInputCanvas.getContext("2d");
const epochReconCtx = refs.epochReconCanvas.getContext("2d");

const decodeScratch = document.createElement("canvas");
decodeScratch.width = 28;
decodeScratch.height = 28;
const decodeScratchCtx = decodeScratch.getContext("2d");

const state = {
  mnistData: null,
  encoder: null,
  decoder: null,
  latentPoints: [],
  latentBounds: { xMin: -3, xMax: 3, yMin: -3, yMax: 3 },
  plotPad: { left: 38, right: 18, top: 18, bottom: 34 },
  hover: { x: 0, y: 0, active: false },
  isTraining: false,
  decodeBusy: false,
  pendingDecode: null,
};

function setStatus(msg) {
  refs.statusText.textContent = msg;
}

function log(msg) {
  refs.log.textContent += `${msg}\n`;
  refs.log.scrollTop = refs.log.scrollHeight;
}

function setUIBusy(isBusy) {
  state.isTraining = isBusy;
  refs.trainBtn.disabled = isBusy;
  refs.loadDataBtn.disabled = isBusy;
  refs.loadSavedBtn.disabled = isBusy;
  refs.saveBtn.disabled = isBusy;
  refs.trainBtn.textContent = isBusy ? "Training..." : "Train VAE";
}

function resetProgress(totalSteps = 100) {
  refs.trainProgress.max = totalSteps;
  refs.trainProgress.value = 0;
  refs.progressText.textContent = "Progress: 0%";
}

function updateProgress(currentStep, totalSteps, loss = null) {
  const boundedCurrent = Math.max(0, Math.min(totalSteps, currentStep));
  refs.trainProgress.max = totalSteps;
  refs.trainProgress.value = boundedCurrent;

  const pct = totalSteps > 0 ? (100 * boundedCurrent) / totalSteps : 0;
  const lossText = loss == null ? "" : ` | step loss=${loss.toFixed(3)}`;
  refs.progressText.textContent = `Progress: ${pct.toFixed(1)}% (${boundedCurrent}/${totalSteps})${lossText}`;
}

function readSettings() {
  return {
    epochs: Number(refs.epochsInput.value),
    stepsPerEpoch: Number(refs.stepsInput.value),
    batchSize: Number(refs.batchInput.value),
    learningRate: Number(refs.lrInput.value),
    klWeight: Number(refs.klInput.value),
    plotCount: Number(refs.plotCountInput.value),
  };
}

function labelColor(label) {
  return `hsl(${label * 36}, 85%, 62%)`;
}

function clearDecodeCanvas(text = "No decoder") {
  decodeCtx.fillStyle = "#05070a";
  decodeCtx.fillRect(0, 0, refs.decodeCanvas.width, refs.decodeCanvas.height);
  decodeCtx.fillStyle = "#9fb1c6";
  decodeCtx.font = "16px Avenir Next";
  decodeCtx.fillText(text, 12, 26);
}

function clearPreviewCanvas(ctx, canvas, text) {
  ctx.fillStyle = "#05070a";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#9fb1c6";
  ctx.font = "13px Avenir Next";
  ctx.fillText(text, 8, 18);
}

function clearLatentCanvas(text = "No latent points") {
  latentCtx.fillStyle = "#05070a";
  latentCtx.fillRect(0, 0, refs.latentCanvas.width, refs.latentCanvas.height);
  latentCtx.fillStyle = "#9fb1c6";
  latentCtx.font = "16px Avenir Next";
  latentCtx.fillText(text, 14, 26);
}

function clearLatentBackground() {
  latentCtx.fillStyle = "#05070a";
  latentCtx.fillRect(0, 0, refs.latentCanvas.width, refs.latentCanvas.height);
}

function computeBounds(points) {
  if (points.length === 0) {
    return { xMin: -3, xMax: 3, yMin: -3, yMax: 3 };
  }

  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;

  for (const [x, y] of points) {
    if (x < xMin) xMin = x;
    if (x > xMax) xMax = x;
    if (y < yMin) yMin = y;
    if (y > yMax) yMax = y;
  }

  const xSpan = Math.max(1e-4, xMax - xMin);
  const ySpan = Math.max(1e-4, yMax - yMin);
  const xPad = Math.max(0.2, xSpan * 0.12);
  const yPad = Math.max(0.2, ySpan * 0.12);

  return {
    xMin: xMin - xPad,
    xMax: xMax + xPad,
    yMin: yMin - yPad,
    yMax: yMax + yPad,
  };
}

function sanitizePoints(points) {
  return points.filter(
    (p) =>
      Array.isArray(p) &&
      p.length >= 3 &&
      Number.isFinite(p[0]) &&
      Number.isFinite(p[1]) &&
      Number.isFinite(p[2]),
  );
}

async function loadFallbackLatentPoints() {
  const response = await fetch(ENCODED_FALLBACK_PATH);
  if (!response.ok) {
    throw new Error(`Failed to load fallback encoded points (${response.status}).`);
  }
  const raw = await response.json();
  return sanitizePoints(raw);
}

function latentToCanvas(x, y) {
  const w = refs.latentCanvas.width;
  const h = refs.latentCanvas.height;
  const { left, right, top, bottom } = state.plotPad;
  const { xMin, xMax, yMin, yMax } = state.latentBounds;

  const px = left + ((x - xMin) / (xMax - xMin)) * (w - left - right);
  const py = h - bottom - ((y - yMin) / (yMax - yMin)) * (h - top - bottom);

  return { x: px, y: py };
}

function canvasToLatent(px, py) {
  const w = refs.latentCanvas.width;
  const h = refs.latentCanvas.height;
  const { left, right, top, bottom } = state.plotPad;
  const { xMin, xMax, yMin, yMax } = state.latentBounds;

  const x = xMin + ((px - left) / (w - left - right)) * (xMax - xMin);
  const y = yMin + ((h - bottom - py) / (h - top - bottom)) * (yMax - yMin);

  return { x, y };
}

function drawLatentAxes() {
  const w = refs.latentCanvas.width;
  const h = refs.latentCanvas.height;
  const { left, right, top, bottom } = state.plotPad;

  latentCtx.strokeStyle = "#1e2630";
  latentCtx.lineWidth = 1;

  for (let i = 0; i <= 10; i += 1) {
    const gx = left + (i / 10) * (w - left - right);
    latentCtx.beginPath();
    latentCtx.moveTo(gx, top);
    latentCtx.lineTo(gx, h - bottom);
    latentCtx.stroke();
  }

  for (let i = 0; i <= 10; i += 1) {
    const gy = top + (i / 10) * (h - top - bottom);
    latentCtx.beginPath();
    latentCtx.moveTo(left, gy);
    latentCtx.lineTo(w - right, gy);
    latentCtx.stroke();
  }

  latentCtx.strokeStyle = "#637286";
  latentCtx.lineWidth = 1.2;

  if (state.latentBounds.yMin <= 0 && state.latentBounds.yMax >= 0) {
    const yAxis = latentToCanvas(0, 0).y;
    latentCtx.beginPath();
    latentCtx.moveTo(left, yAxis);
    latentCtx.lineTo(w - right, yAxis);
    latentCtx.stroke();
  }

  if (state.latentBounds.xMin <= 0 && state.latentBounds.xMax >= 0) {
    const xAxis = latentToCanvas(0, 0).x;
    latentCtx.beginPath();
    latentCtx.moveTo(xAxis, top);
    latentCtx.lineTo(xAxis, h - bottom);
    latentCtx.stroke();
  }
}

function drawLatentPlot() {
  clearLatentBackground();

  if (state.latentPoints.length === 0) {
    clearLatentCanvas("Train/load a model and plot encoded points.");
    return;
  }

  drawLatentAxes();

  for (const point of state.latentPoints) {
    const [x, y, label] = point;
    const p = latentToCanvas(x, y);
    latentCtx.fillStyle = labelColor(label);
    latentCtx.fillRect(p.x, p.y, 2.5, 2.5);
  }

  if (state.hover.active) {
    const p = latentToCanvas(state.hover.x, state.hover.y);
    latentCtx.strokeStyle = "#f8fafc";
    latentCtx.lineWidth = 1;

    latentCtx.beginPath();
    latentCtx.arc(p.x, p.y, 6, 0, Math.PI * 2);
    latentCtx.stroke();

    latentCtx.beginPath();
    latentCtx.moveTo(p.x - 9, p.y);
    latentCtx.lineTo(p.x + 9, p.y);
    latentCtx.moveTo(p.x, p.y - 9);
    latentCtx.lineTo(p.x, p.y + 9);
    latentCtx.stroke();
  }

  latentCtx.fillStyle = "#b8c4d3";
  latentCtx.font = "13px Avenir Next";
  latentCtx.fillText("z1", refs.latentCanvas.width - 24, refs.latentCanvas.height - 10);
  latentCtx.fillText("z2", 8, 16);
}

/**
 * Adapted from VAE-Latent-Space-Explorer-master/src/components/ImageCanvas.jsx
 * where generated tensor pixels are converted into ImageData and scaled on canvas.
 */
function drawDecodedDigit(imgTensor) {
  drawTensorToCanvas(imgTensor, decodeCtx, refs.decodeCanvas);
}

function drawTensorToCanvas(imgTensor, targetCtx, targetCanvas) {
  const pixels = tf.tidy(() => imgTensor.mul(255).dataSync());
  const imageData = decodeScratchCtx.createImageData(28, 28);

  for (let i = 0; i < 28 * 28; i += 1) {
    const px = pixels[i];
    const j = i * 4;
    imageData.data[j] = px;
    imageData.data[j + 1] = px;
    imageData.data[j + 2] = px;
    imageData.data[j + 3] = 255;
  }

  decodeScratchCtx.putImageData(imageData, 0, 0);

  targetCtx.imageSmoothingEnabled = false;
  targetCtx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
  targetCtx.drawImage(decodeScratch, 0, 0, targetCanvas.width, targetCanvas.height);
}

function drawEpochPreview(encoder, decoder, epoch, totalEpochs) {
  if (!state.mnistData || !encoder || !decoder) return;

  const batch = state.mnistData.nextTestBatch(1);
  const input2d = batch.xs.reshape([28, 28]);
  const input4d = batch.xs.reshape([1, 28, 28, 1]);

  const outputs = encoder.predict(input4d);
  const zMean = Array.isArray(outputs) ? outputs[0] : outputs;
  const zLogVar = Array.isArray(outputs) ? outputs[1] : null;

  const reconLogits = decoder.predict(zMean);
  const recon2d = tf.sigmoid(reconLogits).reshape([28, 28]);

  drawTensorToCanvas(input2d, epochInputCtx, refs.epochInputCanvas);
  drawTensorToCanvas(recon2d, epochReconCtx, refs.epochReconCanvas);
  refs.epochPreviewText.textContent = `Epoch ${epoch + 1}/${totalEpochs}`;

  tf.dispose([batch.xs, batch.labels, input2d, input4d, zMean, zLogVar, reconLogits, recon2d]);
}

async function runDecodeLoop() {
  if (state.decodeBusy) return;
  state.decodeBusy = true;

  while (state.pendingDecode) {
    const [z0, z1] = state.pendingDecode;
    state.pendingDecode = null;

    if (!state.decoder) continue;

    const img = decodePoint(state.decoder, z0, z1);
    drawDecodedDigit(img);
    img.dispose();

    await tf.nextFrame();
  }

  state.decodeBusy = false;
}

function queueDecode(z0, z1) {
  state.pendingDecode = [z0, z1];
  runDecodeLoop();
}

function updateHoverText() {
  refs.hoverText.textContent = `z = (${state.hover.x.toFixed(3)}, ${state.hover.y.toFixed(3)})`;
}

/**
 * Hover interaction adapted from the local explorer's XYPlot concept:
 * map pointer to latent coordinates, then decode on movement.
 * Source: VAE-Latent-Space-Explorer-master/src/components/XYPlot.jsx
 */
function onLatentMouseMove(evt) {
  if (!state.decoder) return;

  const rect = refs.latentCanvas.getBoundingClientRect();
  const xPx = ((evt.clientX - rect.left) / rect.width) * refs.latentCanvas.width;
  const yPx = ((evt.clientY - rect.top) / rect.height) * refs.latentCanvas.height;

  const { x, y } = canvasToLatent(xPx, yPx);
  state.hover = { x, y, active: true };
  updateHoverText();
  drawLatentPlot();
  queueDecode(x, y);
}

function onLatentMouseLeave() {
  state.hover.active = false;
  drawLatentPlot();
}

async function ensureDataLoaded() {
  if (state.mnistData) return;

  setStatus("Loading MNIST sprite data...");
  log("Loading MNIST data...");

  state.mnistData = new MnistData();
  await state.mnistData.load();

  setStatus("MNIST data loaded.");
  log("MNIST data ready.");
}

async function refreshLatentPoints() {
  if (!state.encoder) return;

  const { plotCount } = readSettings();
  setStatus(`Encoding ${plotCount} test samples into latent space...`);

  let points = [];
  let usedFallback = false;

  try {
    if (!state.mnistData) {
      throw new Error("MNIST data not loaded");
    }
    points = sanitizePoints(await encodeTestSetSample(state.encoder, state.mnistData, plotCount));
    if (points.length < 50) {
      throw new Error("Encoded points are empty/invalid");
    }
  } catch (err) {
    usedFallback = true;
    log(`WARN: Live latent encoding failed (${err.message}). Using fallback encoded points.`);
    points = await loadFallbackLatentPoints();
  }

  state.latentPoints = points;
  state.latentBounds = computeBounds(state.latentPoints);
  drawLatentPlot();

  if (usedFallback) {
    setStatus(`Latent plot updated with fallback points (${state.latentPoints.length}).`);
  } else {
    setStatus(`Latent plot updated (${state.latentPoints.length} points).`);
  }
}

async function handleTrain() {
  try {
    await ensureDataLoaded();

    const settings = readSettings();
    setUIBusy(true);
    resetProgress(settings.epochs * settings.stepsPerEpoch);
    setStatus("Training VAE...");
    log(
      `Training start | epochs=${settings.epochs} steps=${settings.stepsPerEpoch} batch=${settings.batchSize} lr=${settings.learningRate} kl=${settings.klWeight}`,
    );

    const oldEncoder = state.encoder;
    const oldDecoder = state.decoder;

    const { encoder, decoder } = await trainVAE({
      data: state.mnistData,
      epochs: settings.epochs,
      stepsPerEpoch: settings.stepsPerEpoch,
      batchSize: settings.batchSize,
      learningRate: settings.learningRate,
      klWeight: settings.klWeight,
      onLog: (msg) => log(msg),
      onStepEnd: ({ epoch, step, totalLoss }) => {
        const current = epoch * settings.stepsPerEpoch + step + 1;
        const total = settings.epochs * settings.stepsPerEpoch;
        updateProgress(current, total, totalLoss);
      },
      onEpochEnd: ({ epoch, totalLoss, reconLoss, klLoss, encoder, decoder }) => {
        setStatus(
          `Epoch ${epoch + 1}/${settings.epochs} | total=${totalLoss.toFixed(3)} recon=${reconLoss.toFixed(3)} kl=${klLoss.toFixed(3)}`,
        );
        drawEpochPreview(encoder, decoder, epoch, settings.epochs);
      },
    });

    state.encoder = encoder;
    state.decoder = decoder;
    if (oldEncoder) oldEncoder.dispose();
    if (oldDecoder) oldDecoder.dispose();

    await refreshLatentPoints();

    state.hover = { x: 0, y: 0, active: true };
    updateHoverText();
    queueDecode(0, 0);
    drawLatentPlot();

    setStatus("Training complete. Hover over latent space to decode digits.");
    log("Training complete.");
    updateProgress(settings.epochs * settings.stepsPerEpoch, settings.epochs * settings.stepsPerEpoch);
  } catch (err) {
    console.error(err);
    setStatus("Training failed. Check console/log.");
    log(`ERROR: ${err.message}`);
  } finally {
    setUIBusy(false);
  }
}

async function handleLoadSaved() {
  try {
    setStatus("Loading saved models...");
    const oldEncoder = state.encoder;
    const oldDecoder = state.decoder;
    const { encoder, decoder } = await loadVAE();
    state.encoder = encoder;
    state.decoder = decoder;
    if (oldEncoder) oldEncoder.dispose();
    if (oldDecoder) oldDecoder.dispose();
    log("Loaded encoder/decoder from IndexedDB.");

    // Ensure latent map can be displayed immediately after loading a saved model.
    await ensureDataLoaded();
    await refreshLatentPoints();

    state.hover = { x: 0, y: 0, active: true };
    updateHoverText();
    queueDecode(0, 0);
    drawLatentPlot();

    setStatus("Saved model loaded.");
  } catch (err) {
    console.error(err);
    setStatus("No saved model found (or load failed).");
    log(`ERROR: ${err.message}`);
  }
}

async function handleSave() {
  if (!state.encoder || !state.decoder) {
    setStatus("Train or load a model first.");
    return;
  }

  try {
    setStatus("Saving models to IndexedDB...");
    await saveVAE(state.encoder, state.decoder);
    setStatus("Model saved to IndexedDB.");
    log("Saved encoder/decoder.");
  } catch (err) {
    console.error(err);
    setStatus("Save failed.");
    log(`ERROR: ${err.message}`);
  }
}

function wireEvents() {
  refs.loadDataBtn.addEventListener("click", async () => {
    try {
      await ensureDataLoaded();
      if (state.encoder) {
        await refreshLatentPoints();
      }
    } catch (err) {
      console.error(err);
      setStatus("Data load failed.");
      log(`ERROR: ${err.message}`);
    }
  });

  refs.trainBtn.addEventListener("click", handleTrain);
  refs.saveBtn.addEventListener("click", handleSave);
  refs.loadSavedBtn.addEventListener("click", handleLoadSaved);

  refs.latentCanvas.addEventListener("mousemove", onLatentMouseMove);
  refs.latentCanvas.addEventListener("mouseleave", onLatentMouseLeave);
}

function init() {
  log(`TensorFlow.js backend: ${tf.getBackend()}`);
  log(`Latent dimension: ${LATENT_DIM}`);
  setStatus("Load data, then train or load a saved model.");
  resetProgress();
  clearLatentCanvas("Load data, then train or load model.");
  clearDecodeCanvas("No decoder");
  clearPreviewCanvas(epochInputCtx, refs.epochInputCanvas, "Input");
  clearPreviewCanvas(epochReconCtx, refs.epochReconCanvas, "Recon");
  refs.epochPreviewText.textContent = "No epoch preview yet.";
  wireEvents();
}

init();
