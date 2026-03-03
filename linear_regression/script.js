const degreeSlider = document.getElementById("degreeSlider");
const noiseSlider = document.getElementById("noiseSlider");
const learningRateSlider = document.getElementById("learningRateSlider");
const pointsSlider = document.getElementById("pointsSlider");
const valSplitSlider = document.getElementById("valSplitSlider");

const showTruthToggle = document.getElementById("showTruthToggle");
const showFitToggle = document.getElementById("showFitToggle");
const showDistanceToggle = document.getElementById("showDistanceToggle");

const degreeValue = document.getElementById("degreeValue");
const noiseValue = document.getElementById("noiseValue");
const learningRateValue = document.getElementById("learningRateValue");
const pointsValue = document.getElementById("pointsValue");
const valSplitValue = document.getElementById("valSplitValue");
const equationText = document.getElementById("equationText");
const trainLossText = document.getElementById("trainLossText");
const valLossText = document.getElementById("valLossText");
const epochText = document.getElementById("epochText");

const generateBtn = document.getElementById("generateBtn");
const trainBtn = document.getElementById("trainBtn");

const canvas = document.getElementById("plotCanvas");
const ctx = canvas.getContext("2d");

const colors = {
  bg: "#050608",
  grid: "#1f2430",
  axis: "#596073",
  train: "#4ecdc4",
  val: "#ffd166",
  truth: "#9ca3af",
  fit: "#f43f5e",
  text: "#c5cad3",
};

const state = {
  degree: Number(degreeSlider.value),
  noise: Number(noiseSlider.value),
  learningRate: Number(learningRateSlider.value),
  totalPoints: Number(pointsSlider.value),
  valSplit: Number(valSplitSlider.value) / 100,
  showTruth: showTruthToggle.checked,
  showFit: showFitToggle.checked,
  showDistance: showDistanceToggle.checked,
  trainData: [],
  valData: [],
  coeffs: null,
  trainLoss: null,
  valLoss: null,
  isTraining: false,
  currentEpoch: 0,
  totalEpochs: 0,
  trainRunId: 0,
};

function trueFunction(x) {
  return 0.9 * x * x - 0.3 * x + 0.2;
}

function randomGaussian() {
  let u = 0;
  let v = 0;

  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();

  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function formatMagnitude(value) {
  const abs = Math.abs(value);
  if (abs >= 100 || (abs > 0 && abs < 0.001)) {
    return abs.toExponential(2);
  }
  return abs.toFixed(3);
}

function formatTerm(coeff, power, isFirst) {
  if (Math.abs(coeff) < 1e-10) return "";

  const mag = formatMagnitude(coeff);
  let term = mag;
  if (power === 1) {
    term = `${mag}x`;
  } else if (power > 1) {
    term = `${mag}x^${power}`;
  }

  if (isFirst) {
    return coeff < 0 ? `-${term}` : term;
  }

  return coeff < 0 ? ` - ${term}` : ` + ${term}`;
}

function formatEquation(coeffs) {
  if (!coeffs) return "Equation: train model to show coefficients.";

  let eq = "";
  let isFirst = true;

  // Print from degree 0 upward: a0 + a1x + a2x^2 + ...
  for (let power = 0; power < coeffs.length; power += 1) {
    const piece = formatTerm(coeffs[power], power, isFirst);
    if (piece) {
      eq += piece;
      isFirst = false;
    }
  }

  if (!eq) eq = "0";
  return `Equation: y = ${eq}`;
}

function updateEquationText() {
  const equation = formatEquation(state.coeffs);
  equationText.textContent = equation;
  equationText.title = equation;
}

function updateStatsText() {
  trainLossText.textContent = state.trainLoss === null ? "-" : state.trainLoss.toFixed(5);
  valLossText.textContent = state.valLoss === null ? "-" : state.valLoss.toFixed(5);

  if (state.totalEpochs === 0) {
    epochText.textContent = "-";
  } else {
    epochText.textContent = `${state.currentEpoch}/${state.totalEpochs}`;
  }
}

function updateLabels() {
  degreeValue.textContent = String(state.degree);
  noiseValue.textContent = state.noise.toFixed(2);
  learningRateValue.textContent = state.learningRate.toFixed(3);
  pointsValue.textContent = String(state.totalPoints);
  valSplitValue.textContent = `${Math.floor(state.valSplit * 100)}%`;
}

function shuffle(array) {
  for (let i = array.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

function setTrainingUI(isTraining) {
  generateBtn.disabled = isTraining;
  trainBtn.disabled = isTraining;
  trainBtn.textContent = isTraining ? "Training..." : "Train model";
}

function clearModelState() {
  state.coeffs = null;
  state.trainLoss = null;
  state.valLoss = null;
  state.currentEpoch = 0;
  state.totalEpochs = 0;
  updateEquationText();
  updateStatsText();
}

function createRandomCoefficients(degree) {
  const coeffs = [];
  for (let i = 0; i <= degree; i += 1) {
    const scale = 0.35 / (1 + i * 0.6);
    coeffs.push((Math.random() * 2 - 1) * scale);
  }
  return coeffs;
}

function initializeModelForCurrentDegree() {
  state.trainRunId += 1;
  state.isTraining = false;
  setTrainingUI(false);

  if (state.trainData.length === 0 || state.valData.length === 0) {
    clearModelState();
    return;
  }

  state.currentEpoch = 0;
  state.totalEpochs = computeEpochBudget(state.degree);
  state.coeffs = createRandomCoefficients(state.degree);
  state.trainLoss = mse(state.trainData, state.coeffs);
  state.valLoss = mse(state.valData, state.coeffs);
  updateEquationText();
  updateStatsText();
}

function generateData() {
  state.trainRunId += 1;
  state.isTraining = false;
  setTrainingUI(false);

  const data = [];
  for (let i = 0; i < state.totalPoints; i += 1) {
    const x = -1 + 2 * Math.random();
    const y = trueFunction(x) + state.noise * randomGaussian();
    data.push({ x, y });
  }

  shuffle(data);

  const valCount = Math.max(1, Math.floor(data.length * state.valSplit));
  state.valData = data.slice(0, valCount);
  state.trainData = data.slice(valCount);

  clearModelState();
}

function buildFeatures(x, degree) {
  const features = [1];
  for (let p = 1; p <= degree; p += 1) {
    features[p] = features[p - 1] * x;
  }
  return features;
}

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    sum += a[i] * b[i];
  }
  return sum;
}

function predict(x, coeffs) {
  if (!coeffs) return 0;
  return dot(buildFeatures(x, coeffs.length - 1), coeffs);
}

function mse(data, coeffs) {
  if (data.length === 0 || !coeffs) return null;

  let sum = 0;
  for (const point of data) {
    const err = predict(point.x, coeffs) - point.y;
    sum += err * err;
  }

  return sum / data.length;
}

function gradientStep(data, coeffs, lr, ridge) {
  const grads = Array.from({ length: coeffs.length }, () => 0);
  const n = data.length;

  if (n === 0) return coeffs;

  for (const point of data) {
    const feats = buildFeatures(point.x, coeffs.length - 1);
    const pred = dot(feats, coeffs);
    const err = pred - point.y;

    for (let i = 0; i < coeffs.length; i += 1) {
      grads[i] += (2 * err * feats[i]) / n;
    }
  }

  for (let i = 1; i < coeffs.length; i += 1) {
    grads[i] += 2 * ridge * coeffs[i];
  }

  for (let i = 0; i < coeffs.length; i += 1) {
    coeffs[i] -= lr * grads[i];
  }

  return coeffs;
}

function computeEpochBudget(degree) {
  return 500 + degree * 150;
}

function nextFrame() {
  return new Promise((resolve) => {
    requestAnimationFrame(resolve);
  });
}

async function trainModelProgressive() {
  if (state.trainData.length === 0 || state.valData.length === 0 || state.isTraining) {
    return;
  }

  state.trainRunId += 1;
  const runId = state.trainRunId;

  state.isTraining = true;
  state.currentEpoch = 0;
  state.totalEpochs = computeEpochBudget(state.degree);
  state.coeffs = createRandomCoefficients(state.degree);
  state.trainLoss = mse(state.trainData, state.coeffs);
  state.valLoss = mse(state.valData, state.coeffs);

  setTrainingUI(true);
  updateEquationText();
  updateStatsText();
  renderPlot();

  const stepsPerFrame = 6;
  const ridge = 1e-4;

  while (state.currentEpoch < state.totalEpochs && runId === state.trainRunId) {
    for (let step = 0; step < stepsPerFrame && state.currentEpoch < state.totalEpochs; step += 1) {
      gradientStep(state.trainData, state.coeffs, state.learningRate, ridge);
      state.currentEpoch += 1;
    }

    state.trainLoss = mse(state.trainData, state.coeffs);
    state.valLoss = mse(state.valData, state.coeffs);

    if (!Number.isFinite(state.trainLoss) || !Number.isFinite(state.valLoss)) {
      break;
    }

    updateEquationText();
    updateStatsText();
    renderPlot();
    await nextFrame();
  }

  if (runId === state.trainRunId) {
    state.isTraining = false;
    setTrainingUI(false);
    updateEquationText();
    updateStatsText();
    renderPlot();
  }
}

function getDataRange(points, key) {
  let min = Infinity;
  let max = -Infinity;

  for (const point of points) {
    const value = point[key];
    if (value < min) min = value;
    if (value > max) max = value;
  }

  return { min, max };
}

function drawLine(points, color, width = 2, dashed = false) {
  if (points.length === 0) return;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  if (dashed) ctx.setLineDash([6, 6]);
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i += 1) {
    ctx.lineTo(points[i].x, points[i].y);
  }
  ctx.stroke();
  ctx.restore();
}

function drawPoints(points, color, toCanvas) {
  ctx.fillStyle = color;
  for (const point of points) {
    const c = toCanvas(point.x, point.y);
    ctx.beginPath();
    ctx.arc(c.x, c.y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawResiduals(points, color, toCanvas) {
  if (!state.coeffs) return;

  ctx.save();
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.5;
  ctx.lineWidth = 1.2;
  ctx.setLineDash([3, 3]);

  for (const point of points) {
    const yPred = predict(point.x, state.coeffs);
    const a = toCanvas(point.x, point.y);
    const b = toCanvas(point.x, yPred);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  ctx.restore();
}

function renderPlot() {
  const width = canvas.width;
  const height = canvas.height;

  ctx.fillStyle = colors.bg;
  ctx.fillRect(0, 0, width, height);

  const allData = [...state.trainData, ...state.valData];
  if (allData.length === 0) {
    ctx.fillStyle = colors.text;
    ctx.font = "18px Avenir Next";
    ctx.fillText("Generate data to begin.", 30, 36);
    return;
  }

  const xRangeRaw = getDataRange(allData, "x");
  const yRangeRaw = getDataRange(allData, "y");

  const xSpan = Math.max(1e-6, xRangeRaw.max - xRangeRaw.min);
  const ySpan = Math.max(1e-6, yRangeRaw.max - yRangeRaw.min);

  const xMargin = Math.max(0.03, xSpan * 0.14);
  const yMargin = Math.max(0.04, ySpan * 0.16);

  const xRadius = Math.max(Math.abs(xRangeRaw.min), Math.abs(xRangeRaw.max)) + xMargin;
  const xMin = -xRadius;
  const xMax = xRadius;
  const yMin = yRangeRaw.min - yMargin;
  const yMax = yRangeRaw.max + yMargin;

  const pad = { left: 62, right: 20, top: 20, bottom: 46 };

  const toCanvas = (x, y) => ({
    x: pad.left + ((x - xMin) / (xMax - xMin)) * (width - pad.left - pad.right),
    y: height - pad.bottom - ((y - yMin) / (yMax - yMin)) * (height - pad.top - pad.bottom),
  });

  ctx.strokeStyle = colors.grid;
  ctx.lineWidth = 1;
  for (let i = 0; i <= 8; i += 1) {
    const gx = pad.left + (i / 8) * (width - pad.left - pad.right);
    ctx.beginPath();
    ctx.moveTo(gx, pad.top);
    ctx.lineTo(gx, height - pad.bottom);
    ctx.stroke();
  }
  for (let i = 0; i <= 8; i += 1) {
    const gy = pad.top + (i / 8) * (height - pad.top - pad.bottom);
    ctx.beginPath();
    ctx.moveTo(pad.left, gy);
    ctx.lineTo(width - pad.right, gy);
    ctx.stroke();
  }

  ctx.strokeStyle = colors.axis;
  ctx.lineWidth = 1.4;

  if (yMin <= 0 && yMax >= 0) {
    const axisY = toCanvas(0, 0).y;
    ctx.beginPath();
    ctx.moveTo(pad.left, axisY);
    ctx.lineTo(width - pad.right, axisY);
    ctx.stroke();
  }

  const axisX = toCanvas(0, 0).x;
  ctx.beginPath();
  ctx.moveTo(axisX, pad.top);
  ctx.lineTo(axisX, height - pad.bottom);
  ctx.stroke();

  const linePoints = 300;
  const trueCurve = [];
  const fitCurve = [];

  for (let i = 0; i <= linePoints; i += 1) {
    const x = xMin + ((xMax - xMin) * i) / linePoints;
    if (state.showTruth) {
      trueCurve.push(toCanvas(x, trueFunction(x)));
    }
    if (state.showFit && state.coeffs) {
      fitCurve.push(toCanvas(x, predict(x, state.coeffs)));
    }
  }

  if (state.showTruth) {
    drawLine(trueCurve, colors.truth, 1.8, true);
  }

  if (fitCurve.length > 0) {
    drawLine(fitCurve, colors.fit, 2.8, false);
  }

  if (state.showDistance && state.showFit && state.coeffs) {
    drawResiduals(state.trainData, colors.train, toCanvas);
    drawResiduals(state.valData, colors.val, toCanvas);
  }

  drawPoints(state.trainData, colors.train, toCanvas);
  drawPoints(state.valData, colors.val, toCanvas);

  ctx.fillStyle = colors.text;
  ctx.font = "13px Avenir Next";
  ctx.fillText("x", width - pad.right + 2, height - pad.bottom + 18);
  ctx.fillText("y", pad.left - 16, pad.top + 10);
}

function syncStateFromInputs() {
  state.degree = Number(degreeSlider.value);
  state.noise = Number(noiseSlider.value);
  state.learningRate = Number(learningRateSlider.value);
  state.totalPoints = Number(pointsSlider.value);
  state.valSplit = Number(valSplitSlider.value) / 100;
  updateLabels();
}

function syncToggles() {
  state.showTruth = showTruthToggle.checked;
  state.showFit = showFitToggle.checked;
  state.showDistance = showDistanceToggle.checked;
}

degreeSlider.addEventListener("input", () => {
  syncStateFromInputs();
  initializeModelForCurrentDegree();
  renderPlot();
});

for (const input of [noiseSlider, learningRateSlider, pointsSlider, valSplitSlider]) {
  input.addEventListener("input", () => {
    syncStateFromInputs();
  });
}

for (const toggle of [showTruthToggle, showFitToggle, showDistanceToggle]) {
  toggle.addEventListener("change", () => {
    syncToggles();
    renderPlot();
  });
}

generateBtn.addEventListener("click", () => {
  syncStateFromInputs();
  syncToggles();
  generateData();
  renderPlot();
});

trainBtn.addEventListener("click", () => {
  syncStateFromInputs();
  syncToggles();
  trainModelProgressive();
});

syncStateFromInputs();
syncToggles();
generateData();
renderPlot();
