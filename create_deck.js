const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "True Markets";
pres.title = "True Markets — AI-Powered BTC Prediction Engine";

const BG = "0A0A0F", CARD = "13131A", BORDER = "1E1E2E", ACCENT = "6C5CE7";
const GREEN = "00D4AA", RED = "FF6B6B", YELLOW = "FFD93D", BLUE = "4DABF7";
const WHITE = "FFFFFF", MUTED = "71717A", LIGHT = "E4E4E7";
const chartDir = path.join(__dirname, "charts");

function darkSlide() { const s = pres.addSlide(); s.background = { color: BG }; return s; }
function img(name) { return fs.readFileSync(path.join(chartDir, name)); }

// ═══ SLIDE 1: Title ═══
let s1 = darkSlide();
s1.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 2.0, w: 0.08, h: 1.2, fill: { color: ACCENT } });
s1.addText("True Markets", { x: 0.8, y: 1.9, w: 8, h: 0.7, fontSize: 44, fontFace: "Arial Black", color: WHITE, bold: true, margin: 0 });
s1.addText("AI-Powered BTC Prediction Engine", { x: 0.8, y: 2.6, w: 8, h: 0.5, fontSize: 22, fontFace: "Arial", color: ACCENT, margin: 0 });
s1.addText("YC Application  |  April 2026", { x: 0.8, y: 3.4, w: 8, h: 0.4, fontSize: 14, fontFace: "Arial", color: MUTED, margin: 0 });

// ═══ SLIDE 2: The Problem ═══
let s2 = darkSlide();
s2.addText("The Problem", { x: 0.7, y: 0.4, w: 8, h: 0.6, fontSize: 36, fontFace: "Arial Black", color: WHITE, bold: true, margin: 0 });
[
  { t: "Volatile Markets", d: "BTC swings 5-10% in a day. Retail investors get wiped out." },
  { t: "No Institutional Tools", d: "Hedge funds use on-chain analytics, ML models. Retail has nothing." },
  { t: "Black Box Predictions", d: "Existing tools show a number with no reasoning. No transparency." },
].forEach((p, i) => {
  const y = 1.3 + i * 1.3;
  s2.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y, w: 8.6, h: 1.1, fill: { color: CARD }, rectRadius: 0.1 });
  s2.addText(p.t, { x: 1.0, y: y+0.15, w: 7, h: 0.35, fontSize: 18, fontFace: "Arial", color: RED, bold: true, margin: 0 });
  s2.addText(p.d, { x: 1.0, y: y+0.55, w: 8, h: 0.4, fontSize: 13, fontFace: "Arial", color: MUTED, margin: 0 });
});

// ═══ SLIDE 3: Our Solution ═══
let s3 = darkSlide();
s3.addText("Our Solution", { x: 0.7, y: 0.4, w: 8, h: 0.6, fontSize: 36, fontFace: "Arial Black", color: WHITE, bold: true, margin: 0 });
[
  { n: "6", l: "Real-Time Signals", d: "Polymarket, Binance order flow, ML model, technicals, sentiment, F&G" },
  { n: "67%", l: "Model Accuracy", d: "On-chain ML model, 359 out-of-sample test days" },
  { n: "30s", l: "Refresh Rate", d: "Every signal updates every 30 seconds" },
  { n: "0", l: "Black Boxes", d: "Every signal shows its reasoning under BUY/SELL" },
].forEach((s, i) => {
  const x = 0.7 + (i%2)*4.5, y = 1.3 + Math.floor(i/2)*1.9;
  s3.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 4.1, h: 1.6, fill: { color: CARD }, rectRadius: 0.1 });
  s3.addText(s.n, { x: x+0.3, y: y+0.2, w: 1.5, h: 0.5, fontSize: 32, fontFace: "Arial Black", color: GREEN, bold: true, margin: 0 });
  s3.addText(s.l, { x: x+0.3, y: y+0.7, w: 3.5, h: 0.3, fontSize: 16, fontFace: "Arial", color: WHITE, bold: true, margin: 0 });
  s3.addText(s.d, { x: x+0.3, y: y+1.0, w: 3.5, h: 0.4, fontSize: 11, fontFace: "Arial", color: MUTED, margin: 0 });
});

// ═══ SLIDE 4: The 6 Signals ═══
let s4 = darkSlide();
s4.addText("The 6 Signals", { x: 0.7, y: 0.4, w: 8, h: 0.6, fontSize: 36, fontFace: "Arial Black", color: WHITE, bold: true, margin: 0 });
const sigs = [
  { name: "Polymarket", weight: "20%", source: "Gamma API", color: BLUE },
  { name: "Order Flow", weight: "15%", source: "Binance.US + Coinbase", color: ACCENT },
  { name: "Our Model", weight: "20%", source: "BGeometrics on-chain", color: GREEN },
  { name: "Technical", weight: "20%", source: "TrueMarkets MCP", color: YELLOW },
  { name: "TM Sentiment", weight: "10%", source: "TrueMarkets MCP", color: "A855F7" },
  { name: "Fear & Greed", weight: "15%", source: "alternative.me", color: RED },
];
s4.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: 1.2, w: 8.6, h: 0.4, fill: { color: BORDER }, rectRadius: 0.05 });
s4.addText("Signal", { x: 0.9, y: 1.22, w: 2.5, h: 0.35, fontSize: 11, fontFace: "Arial", color: MUTED, bold: true, margin: 0 });
s4.addText("Weight", { x: 3.5, y: 1.22, w: 1, h: 0.35, fontSize: 11, fontFace: "Arial", color: MUTED, bold: true, margin: 0 });
s4.addText("Source", { x: 5.0, y: 1.22, w: 3, h: 0.35, fontSize: 11, fontFace: "Arial", color: MUTED, bold: true, margin: 0 });
sigs.forEach((sig, i) => {
  const y = 1.7 + i * 0.55;
  s4.addShape(pres.shapes.OVAL, { x: 0.9, y: y+0.1, w: 0.18, h: 0.18, fill: { color: sig.color } });
  s4.addText(sig.name, { x: 1.25, y, w: 2, h: 0.4, fontSize: 14, fontFace: "Arial", color: WHITE, bold: true, margin: 0 });
  s4.addText(sig.weight, { x: 3.5, y, w: 1, h: 0.4, fontSize: 14, fontFace: "Arial", color: GREEN, bold: true, margin: 0 });
  s4.addText(sig.source, { x: 5.0, y, w: 3.5, h: 0.4, fontSize: 12, fontFace: "Arial", color: LIGHT, margin: 0 });
});

// ═══ SLIDE 5: Accuracy Chart ═══
let s5 = darkSlide();
s5.addText("Model Accuracy", { x: 0.7, y: 0.3, w: 5, h: 0.5, fontSize: 30, fontFace: "Arial Black", color: WHITE, bold: true, margin: 0 });
s5.addText("67.4% on 359 out-of-sample days", { x: 0.7, y: 0.8, w: 5, h: 0.3, fontSize: 14, fontFace: "Arial", color: MUTED, margin: 0 });
s5.addImage({ data: "image/png;base64," + img("accuracy.png").toString("base64"), x: 0.5, y: 1.2, w: 5.5, h: 3.2 });
// Right side: key stats
s5.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.3, y: 1.2, w: 3.3, h: 3.8, fill: { color: CARD }, rectRadius: 0.1 });
s5.addText("Key Metrics", { x: 6.5, y: 1.3, w: 3, h: 0.3, fontSize: 13, fontFace: "Arial", color: ACCENT, bold: true, margin: 0 });
[["Accuracy","67.4%"],["Precision","67.2%"],["Recall","66.9%"],["F1 Score","67.0%"],["AUC-ROC","0.7365"],["MCC","0.348"],["Test Days","359"],["Train Days","1,435"]].forEach((r,i) => {
  const y = 1.7 + i*0.35;
  s5.addText(r[0], { x: 6.5, y, w: 1.5, h: 0.3, fontSize: 11, fontFace: "Arial", color: MUTED, margin: 0 });
  s5.addText(r[1], { x: 8.0, y, w: 1.4, h: 0.3, fontSize: 12, fontFace: "Arial", color: WHITE, bold: true, align: "right", margin: 0 });
});

// ═══ SLIDE 6: Confusion Matrix ═══
let s6 = darkSlide();
s6.addText("Confusion Matrix", { x: 0.7, y: 0.3, w: 5, h: 0.5, fontSize: 30, fontFace: "Arial Black", color: WHITE, bold: true, margin: 0 });
s6.addImage({ data: "image/png;base64," + img("confusion.png").toString("base64"), x: 0.5, y: 1.0, w: 4.5, h: 3.6 });
s6.addImage({ data: "image/png;base64," + img("metrics.png").toString("base64"), x: 5.2, y: 1.0, w: 4.5, h: 3.6 });

// ═══ SLIDE 7: Feature Importance ═══
let s7 = darkSlide();
s7.addText("Top 10 Features", { x: 0.7, y: 0.3, w: 6, h: 0.5, fontSize: 30, fontFace: "Arial Black", color: WHITE, bold: true, margin: 0 });
s7.addText("Rate-of-change features dominate \u2014 the key innovation", { x: 0.7, y: 0.8, w: 8, h: 0.3, fontSize: 14, fontFace: "Arial", color: MUTED, margin: 0 });
s7.addImage({ data: "image/png;base64," + img("features.png").toString("base64"), x: 0.3, y: 1.2, w: 9.4, h: 4.0 });

// ═══ SLIDE 8: All Models Tested ═══
let s8 = darkSlide();
s8.addText("All Models Tested", { x: 0.7, y: 0.3, w: 6, h: 0.5, fontSize: 30, fontFace: "Arial Black", color: WHITE, bold: true, margin: 0 });
s8.addText("12 model configurations compared on OOS BTC direction prediction", { x: 0.7, y: 0.8, w: 8, h: 0.3, fontSize: 13, fontFace: "Arial", color: MUTED, margin: 0 });
s8.addImage({ data: "image/png;base64," + img("all_models.png").toString("base64"), x: 0.2, y: 1.1, w: 9.6, h: 4.3 });

// ═══ SLIDE 9: Where It Goes From Here ═══
let s9r = darkSlide();
s9r.addText("Where It Goes From Here", { x: 0.7, y: 0.3, w: 8, h: 0.6, fontSize: 32, fontFace: "Arial Black", color: WHITE, bold: true, margin: 0 });

// Left: Next steps
s9r.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: 1.1, w: 4.3, h: 4.2, fill: { color: CARD }, rectRadius: 0.1 });
s9r.addText("Next Steps", { x: 0.9, y: 1.2, w: 3.5, h: 0.35, fontSize: 16, fontFace: "Arial", color: GREEN, bold: true, margin: 0 });

const steps = [
  { label: "Glassnode Pro (87 features)", desc: "Paper achieves 82.44% with full Glassnode data. We have 42 features from BGeometrics. More data = higher accuracy.", color: GREEN },
  { label: "CNN-LSTM architecture", desc: "Omole & Enke 2024 CNN-LSTM with 87 Glassnode features reached 82.44%. Our GradientBoosting at 67.4% is limited by data, not architecture.", color: ACCENT },
  { label: "Expand to ETH, SOL", desc: "Same 6-signal architecture per coin. BGeometrics supports ETH on-chain. Polymarket has ETH/SOL markets.", color: BLUE },
  { label: "Hourly predictions", desc: "Currently daily. With Glassnode hourly data, can predict intraday moves for higher-frequency trading.", color: YELLOW },
];
steps.forEach((s, i) => {
  const y = 1.65 + i * 0.85;
  s9r.addShape(pres.shapes.OVAL, { x: 0.95, y: y + 0.05, w: 0.15, h: 0.15, fill: { color: s.color } });
  s9r.addText(s.label, { x: 1.25, y, w: 3.5, h: 0.25, fontSize: 12, fontFace: "Arial", color: WHITE, bold: true, margin: 0 });
  s9r.addText(s.desc, { x: 1.25, y: y + 0.28, w: 3.5, h: 0.5, fontSize: 10, fontFace: "Arial", color: MUTED, margin: 0 });
});

// Right: The data gap
s9r.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 5.3, y: 1.1, w: 4.3, h: 4.2, fill: { color: CARD }, rectRadius: 0.1 });
s9r.addText("The Data Gap", { x: 5.5, y: 1.2, w: 3.5, h: 0.35, fontSize: 16, fontFace: "Arial", color: YELLOW, bold: true, margin: 0 });

s9r.addText("Current", { x: 5.5, y: 1.7, w: 1.5, h: 0.25, fontSize: 11, fontFace: "Arial", color: MUTED, bold: true, margin: 0 });
s9r.addText("67.4%", { x: 7.5, y: 1.7, w: 1.8, h: 0.25, fontSize: 14, fontFace: "Arial", color: GREEN, bold: true, align: "right", margin: 0 });
s9r.addText("42 features from BGeometrics", { x: 5.5, y: 1.95, w: 4, h: 0.2, fontSize: 10, fontFace: "Arial", color: MUTED, margin: 0 });

s9r.addShape(pres.shapes.LINE, { x: 5.5, y: 2.3, w: 3.8, h: 0, line: { color: BORDER, width: 1 } });

s9r.addText("Target", { x: 5.5, y: 2.5, w: 1.5, h: 0.25, fontSize: 11, fontFace: "Arial", color: MUTED, bold: true, margin: 0 });
s9r.addText("82.4%", { x: 7.5, y: 2.5, w: 1.8, h: 0.25, fontSize: 14, fontFace: "Arial", color: YELLOW, bold: true, align: "right", margin: 0 });
s9r.addText("87 features from Glassnode Pro ($799/mo)", { x: 5.5, y: 2.75, w: 4, h: 0.2, fontSize: 10, fontFace: "Arial", color: MUTED, margin: 0 });

s9r.addShape(pres.shapes.LINE, { x: 5.5, y: 3.1, w: 3.8, h: 0, line: { color: BORDER, width: 1 } });

s9r.addText("What\u2019s missing:", { x: 5.5, y: 3.3, w: 4, h: 0.25, fontSize: 11, fontFace: "Arial", color: WHITE, bold: true, margin: 0 });
const missing = [
  "HODL waves (all 15 age bands vs our 13)",
  "Realized cap HODL waves",
  "Entity-adjusted dormancy flow",
  "Stock-to-flow ratio",
  "Difficulty ribbon compression",
  "45 additional on-chain indicators",
];
missing.forEach((m, i) => {
  s9r.addText("\u2022 " + m, { x: 5.7, y: 3.6 + i * 0.25, w: 3.8, h: 0.22, fontSize: 9, fontFace: "Arial", color: MUTED, margin: 0 });
});

s9r.addText("More data is the single biggest lever to improve accuracy.", { x: 5.5, y: 5.0, w: 3.8, h: 0.2, fontSize: 9, fontFace: "Arial", color: YELLOW, italic: true, margin: 0 });

// ═══ SLIDE 10: Tech Stack ═══
let s10 = darkSlide();
s10.addText("Tech Stack", { x: 0.7, y: 0.4, w: 8, h: 0.6, fontSize: 36, fontFace: "Arial Black", color: WHITE, bold: true, margin: 0 });
[
  { c: "Backend", i: "FastAPI + Python, scikit-learn, PyTorch", cl: GREEN },
  { c: "Frontend", i: "Next.js 14, React 18, TailwindCSS", cl: BLUE },
  { c: "Data", i: "BGeometrics Premium, Binance.US, Coinbase, Polymarket", cl: ACCENT },
  { c: "Price", i: "TrueMarkets MCP (same source everywhere)", cl: YELLOW },
  { c: "Refresh", i: "30 seconds, all 6 signals", cl: RED },
].forEach((s, i) => {
  const y = 1.2 + i * 0.8;
  s10.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y, w: 8.6, h: 0.65, fill: { color: CARD }, rectRadius: 0.08 });
  s10.addShape(pres.shapes.RECTANGLE, { x: 0.7, y, w: 0.06, h: 0.65, fill: { color: s.cl } });
  s10.addText(s.c, { x: 1.1, y: y+0.05, w: 2, h: 0.3, fontSize: 15, fontFace: "Arial", color: WHITE, bold: true, margin: 0 });
  s10.addText(s.i, { x: 1.1, y: y+0.33, w: 7.5, h: 0.25, fontSize: 12, fontFace: "Arial", color: MUTED, margin: 0 });
});

// ═══ SLIDE 11: Product ═══
let s11 = darkSlide();
s11.addText("The Product", { x: 0.7, y: 0.4, w: 8, h: 0.6, fontSize: 36, fontFace: "Arial Black", color: WHITE, bold: true, margin: 0 });
s11.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: 1.2, w: 4.1, h: 3.8, fill: { color: CARD }, rectRadius: 0.1 });
s11.addText("Market Page", { x: 0.9, y: 1.3, w: 3.5, h: 0.4, fontSize: 18, fontFace: "Arial", color: GREEN, bold: true, margin: 0 });
s11.addText([
  { text: "Live BTC price + chart", options: { bullet: true, breakLine: true } },
  { text: "Market cap, volume, highs/lows", options: { bullet: true, breakLine: true } },
  { text: "Fear & Greed Index", options: { bullet: true, breakLine: true } },
  { text: "TM Sentiment (30+ sources)", options: { bullet: true } },
], { x: 0.9, y: 1.8, w: 3.7, h: 2.5, fontSize: 12, fontFace: "Arial", color: LIGHT });

s11.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 5.2, y: 1.2, w: 4.3, h: 3.8, fill: { color: CARD }, rectRadius: 0.1 });
s11.addText("Prediction Page", { x: 5.4, y: 1.3, w: 3.5, h: 0.4, fontSize: 18, fontFace: "Arial", color: ACCENT, bold: true, margin: 0 });
s11.addText([
  { text: "BUY / SELL with 6-signal reasoning", options: { bullet: true, breakLine: true } },
  { text: "Polymarket table (18 thresholds)", options: { bullet: true, breakLine: true } },
  { text: "Binance order flow visualization", options: { bullet: true, breakLine: true } },
  { text: "RSI, MACD, Bollinger live", options: { bullet: true, breakLine: true } },
  { text: "Portfolio + order management", options: { bullet: true } },
], { x: 5.4, y: 1.8, w: 3.9, h: 2.5, fontSize: 12, fontFace: "Arial", color: LIGHT });

// ═══ SLIDE 12: The Ask ═══
let s12 = darkSlide();
s12.addText("The Ask", { x: 0.7, y: 0.4, w: 8, h: 0.6, fontSize: 36, fontFace: "Arial Black", color: WHITE, bold: true, margin: 0 });
s12.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: 1.2, w: 4, h: 1.5, fill: { color: CARD }, rectRadius: 0.1 });
s12.addText("$500K", { x: 0.9, y: 1.3, w: 3.5, h: 0.6, fontSize: 40, fontFace: "Arial Black", color: GREEN, bold: true, margin: 0 });
s12.addText("at $5M valuation", { x: 0.9, y: 1.9, w: 3.5, h: 0.3, fontSize: 16, fontFace: "Arial", color: MUTED, margin: 0 });
s12.addText("Use of Funds", { x: 0.7, y: 3.0, w: 8, h: 0.4, fontSize: 18, fontFace: "Arial", color: WHITE, bold: true, margin: 0 });
[
  { i: "Glassnode Pro ($799/mo)", d: "87 features \u2192 target 82%+ accuracy", c: GREEN },
  { i: "Expand to ETH, SOL, top 10", d: "Same architecture, per-coin models", c: BLUE },
  { i: "Mobile app (React Native)", d: "Push notifications on signal changes", c: ACCENT },
  { i: "Team: ML eng + full-stack", d: "Scale infra, more data sources", c: YELLOW },
].forEach((f, i) => {
  const y = 3.5 + i * 0.45;
  s12.addShape(pres.shapes.OVAL, { x: 0.9, y: y+0.08, w: 0.18, h: 0.18, fill: { color: f.c } });
  s12.addText(f.i, { x: 1.3, y, w: 4, h: 0.25, fontSize: 13, fontFace: "Arial", color: WHITE, bold: true, margin: 0 });
  s12.addText(f.d, { x: 5.5, y, w: 4, h: 0.25, fontSize: 11, fontFace: "Arial", color: MUTED, margin: 0 });
});

pres.writeFile({ fileName: path.join(__dirname, "TrueMarkets_Pitch_Deck.pptx") })
  .then(() => console.log("Saved: TrueMarkets_Pitch_Deck.pptx (10 slides with charts)"))
  .catch(e => console.error(e));
