"use strict";

// ========================================
// 基本設定
// ========================================

const INPUT = 256;
const HIDDEN = 32;
const OUTPUT = 2;

const EPOCHS = 300;
const LEARNING_RATE = 0.03;

const STORAGE_KEY = "oxRecognitionDataset_v2";

let dataset = [];
let model = null;
let lossHistory = [];
let isTraining = false;

// ========================================
// DOM取得
// ========================================

const $ = (id) => document.getElementById(id);

const canvas = $("canvas");
const ctx = canvas.getContext("2d");

const lossCanvas = $("lossChart");
const lossCtx = lossCanvas.getContext("2d");

const networkCanvas = $("networkCanvas");
const networkCtx = networkCanvas.getContext("2d");

// ========================================
// キャンバス描画
// ========================================

function clearCanvas() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "#000000";
  ctx.lineWidth = 12;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
}

clearCanvas();

let drawing = false;

function getPosition(event) {
  const rect = canvas.getBoundingClientRect();

  return {
    x: ((event.clientX - rect.left) * canvas.width) / rect.width,
    y: ((event.clientY - rect.top) * canvas.height) / rect.height,
  };
}

canvas.addEventListener("pointerdown", (event) => {
  event.preventDefault();

  drawing = true;
  canvas.setPointerCapture(event.pointerId);

  const pos = getPosition(event);

  ctx.beginPath();
  ctx.moveTo(pos.x, pos.y);
  ctx.lineTo(pos.x + 0.1, pos.y + 0.1);
  ctx.stroke();
});

canvas.addEventListener("pointermove", (event) => {
  if (!drawing) return;

  event.preventDefault();

  const pos = getPosition(event);

  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();
});

function stopDrawing() {
  drawing = false;
  ctx.beginPath();
}

canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointercancel", stopDrawing);

$("clearButton").addEventListener("click", clearCanvas);

// ========================================
// 画像を16×16に変換
// ========================================

function getImageData() {
  const smallCanvas = document.createElement("canvas");

  smallCanvas.width = 16;
  smallCanvas.height = 16;

  const smallCtx = smallCanvas.getContext("2d");

  smallCtx.drawImage(canvas, 0, 0, 280, 280, 0, 0, 16, 16);

  const image = smallCtx.getImageData(0, 0, 16, 16);

  const input = [];

  for (let i = 0; i < image.data.length; i += 4) {
    const brightness =
      (image.data[i] + image.data[i + 1] + image.data[i + 2]) / 3;

    input.push(1 - brightness / 255);
  }

  return input;
}

// ========================================
// 学習データ保存・読み込み
// ========================================

function saveDataset() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(dataset));
}

function loadDataset() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);

    if (!saved) return;

    const parsed = JSON.parse(saved);

    if (!Array.isArray(parsed)) return;

    dataset = parsed.filter(
      (item) =>
        item &&
        Array.isArray(item.input) &&
        item.input.length === INPUT &&
        item.input.every(
          (x) =>
            typeof x === "number" && Number.isFinite(x) && x >= 0 && x <= 1,
        ) &&
        (item.label === 0 || item.label === 1),
    );
  } catch (error) {
    console.error("データ読み込みエラー", error);
    dataset = [];
  }
}

// ========================================
// 学習データ登録
// ========================================

function addSample(label) {
  if (isTraining) return;

  const input = getImageData();

  if (input.every((x) => x < 0.01)) {
    alert("先に画像を描いてください");
    return;
  }

  dataset.push({ input, label });

  saveDataset();
  updateDatasetInfo();

  // 登録したら自動消去
  clearCanvas();

  // データ変更後は再学習が必要
  model = null;

  $("modelStatus").textContent = "再学習が必要";
  $("predictButton").disabled = true;

  $("predictionStatus").textContent =
    "データが変更されました。再学習してください。";

  $("resultSymbol").textContent = "？";
  $("resultPercent").textContent = "---";

  $("circleProbability").textContent = "---";
  $("crossProbability").textContent = "---";

  $("circleBar").style.width = "0%";
  $("crossBar").style.width = "0%";

  drawNetwork(null);
}

$("circleButton").addEventListener("click", () => addSample(0));
$("crossButton").addEventListener("click", () => addSample(1));

// ========================================
// 学習データ表示
// ========================================

function updateDatasetInfo() {
  const circles = dataset.filter((x) => x.label === 0).length;
  const crosses = dataset.filter((x) => x.label === 1).length;

  $("circleCount").textContent = circles;
  $("crossCount").textContent = crosses;

  $("datasetInfo").textContent = `登録データ：${dataset.length}件`;

  $("systemDataset").textContent = `${dataset.length}件`;
}

// ========================================
// ニューラルネットワーク初期化
// ========================================

function randomWeight(size) {
  return (Math.random() * 2 - 1) * Math.sqrt(2 / size);
}

function createModel() {
  return {
    W1: Array.from({ length: HIDDEN }, () =>
      Array.from({ length: INPUT }, () => randomWeight(INPUT)),
    ),

    b1: Array(HIDDEN).fill(0),

    W2: Array.from({ length: OUTPUT }, () =>
      Array.from({ length: HIDDEN }, () => randomWeight(HIDDEN)),
    ),

    b2: Array(OUTPUT).fill(0),
  };
}

// ========================================
// 活性化関数
// ========================================

function relu(x) {
  return Math.max(0, x);
}

function reluDerivative(x) {
  return x > 0 ? 1 : 0;
}

function softmax(values) {
  const max = Math.max(...values);

  const exps = values.map((x) => Math.exp(x - max));

  const sum = exps.reduce((a, b) => a + b, 0);

  return exps.map((x) => x / sum);
}

// ========================================
// 順伝播
// ========================================

function forward(input, m = model) {
  const z1 = [];
  const h = [];

  // 入力層 → 隠れ層
  for (let i = 0; i < HIDDEN; i++) {
    let sum = m.b1[i];

    for (let j = 0; j < INPUT; j++) {
      sum += m.W1[i][j] * input[j];
    }

    z1[i] = sum;
    h[i] = relu(sum);
  }

  // 隠れ層 → 出力層
  const z2 = [];

  for (let i = 0; i < OUTPUT; i++) {
    let sum = m.b2[i];

    for (let j = 0; j < HIDDEN; j++) {
      sum += m.W2[i][j] * h[j];
    }

    z2[i] = sum;
  }

  const probabilities = softmax(z2);

  return {
    input,
    z1,
    h,
    z2,
    probabilities,
  };
}

// ========================================
// 損失関数
// ========================================

function crossEntropy(probabilities, label) {
  const epsilon = 1e-7;

  return -Math.log(Math.max(probabilities[label], epsilon));
}

// ========================================
// 誤差逆伝播・重み更新
// ========================================

function trainSample(sample) {
  const input = sample.input;
  const label = sample.label;

  const result = forward(input);

  const h = result.h;
  const z1 = result.z1;
  const probabilities = result.probabilities;

  // 出力層の誤差
  const delta2 = probabilities.slice();
  delta2[label] -= 1;

  // 隠れ層の誤差
  const delta1 = Array(HIDDEN).fill(0);

  for (let j = 0; j < HIDDEN; j++) {
    let error = 0;

    for (let i = 0; i < OUTPUT; i++) {
      error += model.W2[i][j] * delta2[i];
    }

    delta1[j] = error * reluDerivative(z1[j]);
  }

  // 出力層の重み更新
  for (let i = 0; i < OUTPUT; i++) {
    for (let j = 0; j < HIDDEN; j++) {
      model.W2[i][j] -= LEARNING_RATE * delta2[i] * h[j];
    }

    model.b2[i] -= LEARNING_RATE * delta2[i];
  }

  // 入力層の重み更新
  for (let i = 0; i < HIDDEN; i++) {
    for (let j = 0; j < INPUT; j++) {
      model.W1[i][j] -= LEARNING_RATE * delta1[i] * input[j];
    }

    model.b1[i] -= LEARNING_RATE * delta1[i];
  }
}

// ========================================
// Loss評価
// ========================================

function evaluateLoss() {
  let totalLoss = 0;

  for (const sample of dataset) {
    const result = forward(sample.input);

    totalLoss += crossEntropy(result.probabilities, sample.label);
  }

  return totalLoss / dataset.length;
}

// ========================================
// 正解率評価
// ========================================

function evaluateAccuracy() {
  let correct = 0;

  for (const sample of dataset) {
    const result = forward(sample.input);

    const predicted = result.probabilities[1] > result.probabilities[0] ? 1 : 0;

    if (predicted === sample.label) {
      correct++;
    }
  }

  return (correct / dataset.length) * 100;
}

// ========================================
// Lossグラフ
// ========================================

function drawLossGraph() {
  const width = lossCanvas.width;
  const height = lossCanvas.height;

  const left = 65;
  const right = 20;
  const top = 25;
  const bottom = 45;

  const graphWidth = width - left - right;
  const graphHeight = height - top - bottom;

  lossCtx.clearRect(0, 0, width, height);
  lossCtx.fillStyle = "#0f172a";
  lossCtx.fillRect(0, 0, width, height);

  if (lossHistory.length === 0) {
    lossCtx.fillStyle = "#94a3b8";
    lossCtx.font = "15px sans-serif";
    lossCtx.textAlign = "center";
    lossCtx.fillText(
      "学習を開始するとグラフが表示されます",
      width / 2,
      height / 2,
    );
    return;
  }

  const maxLoss = Math.max(0.01, ...lossHistory.map((x) => x.loss));

  // グリッド
  for (let i = 0; i <= 4; i++) {
    const y = top + (graphHeight * i) / 4;
    const value = maxLoss * (1 - i / 4);

    lossCtx.strokeStyle = "#334155";
    lossCtx.beginPath();
    lossCtx.moveTo(left, y);
    lossCtx.lineTo(width - right, y);
    lossCtx.stroke();

    lossCtx.fillStyle = "#94a3b8";
    lossCtx.font = "12px sans-serif";
    lossCtx.textAlign = "right";
    lossCtx.fillText(value.toFixed(3), left - 9, y + 4);
  }

  // X軸
  lossCtx.textAlign = "center";

  for (let i = 0; i <= 3; i++) {
    const x = left + (graphWidth * i) / 3;
    const epoch = Math.round((EPOCHS * i) / 3);

    lossCtx.fillText(epoch, x, height - 17);
  }

  // Loss折れ線
  lossCtx.strokeStyle = "#38bdf8";
  lossCtx.lineWidth = 2.5;
  lossCtx.beginPath();

  lossHistory.forEach((point, index) => {
    const x = left + (graphWidth * point.epoch) / EPOCHS;
    const y = top + graphHeight * (1 - point.loss / maxLoss);

    if (index === 0) {
      lossCtx.moveTo(x, y);
    } else {
      lossCtx.lineTo(x, y);
    }
  });

  lossCtx.stroke();

  // 最終点
  const last = lossHistory[lossHistory.length - 1];

  const lastX = left + (graphWidth * last.epoch) / EPOCHS;
  const lastY = top + graphHeight * (1 - last.loss / maxLoss);

  lossCtx.fillStyle = "#facc15";
  lossCtx.beginPath();
  lossCtx.arc(lastX, lastY, 5, 0, Math.PI * 2);
  lossCtx.fill();

  lossCtx.fillStyle = "#facc15";
  lossCtx.textAlign = "left";
  lossCtx.fillText(`Loss: ${last.loss.toFixed(6)}`, left, 15);
}

// ========================================
// ニューラルネットワーク可視化
// ========================================

function drawNetwork(result) {
  const ctx = networkCtx;
  const width = networkCanvas.width;
  const height = networkCanvas.height;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(0, 0, width, height);

  if (!model || !result) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "16px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("学習後に画像を判定すると表示されます", width / 2, height / 2);
    $("networkInfo").textContent = "ニューロンの反応はまだありません。";
    return;
  }

  const inputX = 100;
  const hiddenX = 450;
  const outputX = 800;

  const inputY = Array.from({ length: INPUT }, (_, i) => 35 + i * 1.75);

  const hiddenY = Array.from({ length: HIDDEN }, (_, i) => 35 + i * 14.5);

  const outputY = [220, 300];

  // 入力層 → 隠れ層の接続
  for (let i = 0; i < HIDDEN; i++) {
    for (let j = 0; j < INPUT; j++) {
      const weight = model.W1[i][j];

      ctx.beginPath();
      ctx.moveTo(inputX, inputY[j]);
      ctx.lineTo(hiddenX, hiddenY[i]);

      const alpha = Math.min(Math.abs(weight) * 0.5, 0.35);

      ctx.strokeStyle =
        weight >= 0 ? `rgba(249,115,22,${alpha})` : `rgba(56,189,248,${alpha})`;

      ctx.lineWidth = Math.min(Math.abs(weight) * 1.5, 2);

      ctx.stroke();
    }
  }

  // 隠れ層 → 出力層の接続
  for (let i = 0; i < OUTPUT; i++) {
    for (let j = 0; j < HIDDEN; j++) {
      const weight = model.W2[i][j];

      ctx.beginPath();
      ctx.moveTo(hiddenX, hiddenY[j]);
      ctx.lineTo(outputX, outputY[i]);

      const alpha = Math.min(Math.abs(weight) * 0.5, 0.7);

      ctx.strokeStyle =
        weight >= 0 ? `rgba(249,115,22,${alpha})` : `rgba(56,189,248,${alpha})`;

      ctx.lineWidth = Math.min(Math.abs(weight) * 2, 3);

      ctx.stroke();
    }
  }

  // 入力ニューロン
  for (let i = 0; i < INPUT; i++) {
    const value = result.input[i];

    ctx.beginPath();
    ctx.arc(inputX, inputY[i], 2.5, 0, Math.PI * 2);

    ctx.fillStyle = `rgba(56,189,248,${0.15 + value * 0.85})`;

    ctx.fill();
  }

  // 隠れ層ニューロン
  for (let i = 0; i < HIDDEN; i++) {
    const value = result.h[i];
    const intensity = Math.min(value / 3, 1);

    ctx.beginPath();
    ctx.arc(hiddenX, hiddenY[i], 7, 0, Math.PI * 2);

    ctx.fillStyle = `rgba(167,139,250,${0.15 + intensity * 0.85})`;

    ctx.fill();

    ctx.strokeStyle = "#c4b5fd";
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // 出力ニューロン
  for (let i = 0; i < OUTPUT; i++) {
    const probability = result.probabilities[i];

    ctx.beginPath();
    ctx.arc(outputX, outputY[i], 22, 0, Math.PI * 2);

    ctx.fillStyle =
      i === 0
        ? `rgba(74,222,128,${0.15 + probability * 0.85})`
        : `rgba(96,165,250,${0.15 + probability * 0.85})`;

    ctx.fill();

    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 17px sans-serif";
    ctx.textAlign = "center";

    ctx.fillText(i === 0 ? "○" : "×", outputX, outputY[i] + 6);
  }

  // 層の名前
  ctx.fillStyle = "#e2e8f0";
  ctx.font = "bold 18px sans-serif";
  ctx.textAlign = "center";

  ctx.fillText("入力層", inputX, 485);
  ctx.fillText("隠れ層", hiddenX, 485);
  ctx.fillText("出力層", outputX, 485);

  ctx.font = "13px sans-serif";
  ctx.fillStyle = "#94a3b8";

  ctx.fillText("256ニューロン", inputX, 505);
  ctx.fillText("32ニューロン", hiddenX, 505);
  ctx.fillText("2ニューロン", outputX, 505);

  $("networkInfo").textContent =
    `隠れ層の最大活性化値：${Math.max(...result.h).toFixed(4)}`;
}

// ========================================
// 学習処理
// ========================================

async function train() {
  if (isTraining) return;

  if (dataset.length < 2) {
    alert("学習データを2件以上登録してください");
    return;
  }

  const circles = dataset.filter((x) => x.label === 0).length;
  const crosses = dataset.filter((x) => x.label === 1).length;

  if (circles === 0 || crosses === 0) {
    alert("○と×の両方のデータを登録してください");
    return;
  }

  isTraining = true;

  const button = $("trainButton");
  button.disabled = true;
  button.textContent = "学習中...";

  model = createModel();
  lossHistory = [];

  drawLossGraph();
  drawNetwork(null);

  $("predictButton").disabled = true;
  $("modelStatus").textContent = "学習中";
  $("trainStatus").textContent = "ニューラルネットワークを学習しています...";
  $("predictionStatus").textContent = "学習完了後に判定できます";

  try {
    for (let epoch = 1; epoch <= EPOCHS; epoch++) {
      // データをシャッフル
      const shuffled = dataset.slice();

      for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
      }

      // 全データを学習
      for (const sample of shuffled) {
        trainSample(sample);
      }

      // 更新後のモデルでLossを評価
      const loss = evaluateLoss();

      lossHistory.push({ epoch, loss });

      $("epochDisplay").textContent = epoch;
      $("lossDisplay").textContent = loss.toFixed(6);
      $("accuracyDisplay").textContent = evaluateAccuracy().toFixed(2) + "%";

      if (epoch % 10 === 0 || epoch === EPOCHS) {
        drawLossGraph();

        $("trainStatus").textContent = `学習中：${epoch} / ${EPOCHS} Epoch`;

        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }

    $("trainStatus").textContent = "学習完了！";
    $("modelStatus").textContent = "学習済み";
    $("predictButton").disabled = false;
  } catch (error) {
    console.error(error);

    $("trainStatus").textContent = "学習中にエラーが発生しました";

    $("modelStatus").textContent = "エラー";
    model = null;
  } finally {
    isTraining = false;
    button.disabled = false;
    button.textContent = "学習開始";
  }
}

$("trainButton").addEventListener("click", train);

// ========================================
// 画像認識
// ========================================

function predict() {
  if (!model) {
    alert("先に学習してください");
    return;
  }

  const input = getImageData();

  if (input.every((x) => x < 0.01)) {
    alert("画像を描いてください");
    return;
  }

  const result = forward(input);

  const circle = result.probabilities[0];
  const cross = result.probabilities[1];

  const predicted = circle > cross ? 0 : 1;
  const confidence = Math.max(circle, cross) * 100;

  $("resultSymbol").textContent = predicted === 0 ? "○" : "×";

  $("resultSymbol").style.color = predicted === 0 ? "#4ade80" : "#60a5fa";

  $("resultPercent").textContent = confidence.toFixed(2) + "%";

  $("circleProbability").textContent = (circle * 100).toFixed(2) + "%";

  $("crossProbability").textContent = (cross * 100).toFixed(2) + "%";

  $("circleBar").style.width = circle * 100 + "%";

  $("crossBar").style.width = cross * 100 + "%";

  $("predictionStatus").textContent =
    `判定結果：${predicted === 0 ? "○" : "×"}`;

  // 実際の順伝播結果でネットワークを描画
  drawNetwork(result);
}

$("predictButton").addEventListener("click", predict);

// ========================================
// 学習データ全削除
// ========================================

$("deleteDataButton").addEventListener("click", () => {
  if (isTraining) return;

  if (!confirm("すべての学習データを削除しますか？")) {
    return;
  }

  dataset = [];
  model = null;
  lossHistory = [];

  saveDataset();
  updateDatasetInfo();

  $("epochDisplay").textContent = "0";
  $("lossDisplay").textContent = "---";
  $("accuracyDisplay").textContent = "---";

  $("resultSymbol").textContent = "？";
  $("resultPercent").textContent = "---";

  $("circleProbability").textContent = "---";
  $("crossProbability").textContent = "---";

  $("circleBar").style.width = "0%";
  $("crossBar").style.width = "0%";

  $("trainStatus").textContent = "まだ学習していません";
  $("predictionStatus").textContent = "まだ学習していません";
  $("modelStatus").textContent = "未学習";

  $("predictButton").disabled = true;

  drawLossGraph();
  drawNetwork(null);
});

// ========================================
// 初期化
// ========================================

loadDataset();
updateDatasetInfo();
drawLossGraph();
drawNetwork(null);
