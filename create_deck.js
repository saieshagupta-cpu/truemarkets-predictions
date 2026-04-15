const pptxgen = require("pptxgenjs");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "True Markets";
pres.title = "True Markets — AI-Powered BTC Prediction Engine";

// Colors
const BG = "0A0A0F";
const CARD = "13131A";
const BORDER = "1E1E2E";
const ACCENT = "6C5CE7";
const GREEN = "00D4AA";
const RED = "FF6B6B";
const YELLOW = "FFD93D";
const BLUE = "4DABF7";
const WHITE = "FFFFFF";
const MUTED = "71717A";
const LIGHT = "E4E4E7";

function darkSlide() {
  const s = pres.addSlide();
  s.background = { color: BG };
  return s;
}

// ═══════════════════════════════════════════════
// SLIDE 1: Title
// ═══════════════════════════════════════════════
let s1 = darkSlide();
s1.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 5.625, fill: { color: BG } });
s1.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 2.0, w: 0.08, h: 1.2, fill: { color: ACCENT } });
s1.addText("True Markets", { x: 0.8, y: 1.9, w: 8, h: 0.7, fontSize: 44, fontFace: "Arial Black", color: WHITE, bold: true, margin: 0 });
s1.addText("AI-Powered BTC Prediction Engine", { x: 0.8, y: 2.6, w: 8, h: 0.5, fontSize: 22, fontFace: "Arial", color: ACCENT, margin: 0 });
s1.addText("YC Application  |  April 2026", { x: 0.8, y: 3.4, w: 8, h: 0.4, fontSize: 14, fontFace: "Arial", color: MUTED, margin: 0 });

// ═══════════════════════════════════════════════
// SLIDE 2: The Problem
// ═══════════════════════════════════════════════
let s2 = darkSlide();
s2.addText("The Problem", { x: 0.7, y: 0.4, w: 8, h: 0.6, fontSize: 36, fontFace: "Arial Black", color: WHITE, bold: true, margin: 0 });
s2.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 1.05, w: 1.5, h: 0.04, fill: { color: RED } });

const problems = [
  { title: "Volatile Markets", desc: "BTC swings 5-10% in a day. Retail investors get wiped out by emotional trading." },
  { title: "No Institutional Tools", desc: "Hedge funds use on-chain analytics, order flow, ML models. Retail has nothing comparable." },
  { title: "Black Box Predictions", desc: "Existing tools show a number with no reasoning. Users can't verify or understand why." },
];
problems.forEach((p, i) => {
  const y = 1.5 + i * 1.3;
  s2.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y, w: 8.6, h: 1.1, fill: { color: CARD }, rectRadius: 0.1 });
  s2.addText(p.title, { x: 1.0, y: y + 0.15, w: 7, h: 0.35, fontSize: 18, fontFace: "Arial", color: RED, bold: true, margin: 0 });
  s2.addText(p.desc, { x: 1.0, y: y + 0.55, w: 8, h: 0.4, fontSize: 13, fontFace: "Arial", color: MUTED, margin: 0 });
});

// ═══════════════════════════════════════════════
// SLIDE 3: Our Solution
// ═══════════════════════════════════════════════
let s3 = darkSlide();
s3.addText("Our Solution", { x: 0.7, y: 0.4, w: 8, h: 0.6, fontSize: 36, fontFace: "Arial Black", color: WHITE, bold: true, margin: 0 });
s3.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 1.05, w: 1.5, h: 0.04, fill: { color: GREEN } });

const solutions = [
  { num: "6", label: "Real-Time Signals", desc: "Polymarket, Binance order flow, ML model, technicals, sentiment, Fear & Greed" },
  { num: "67%", label: "Model Accuracy", desc: "On-chain ML model tested on 359 out-of-sample days (Apr 2025 - Apr 2026)" },
  { num: "30s", label: "Refresh Rate", desc: "Every signal updates every 30 seconds with real data from verified sources" },
  { num: "0", label: "Black Boxes", desc: "Every signal shows its reasoning. Users see WHY buy or sell, not just the answer." },
];
solutions.forEach((s, i) => {
  const x = 0.7 + (i % 2) * 4.5;
  const y = 1.5 + Math.floor(i / 2) * 1.9;
  s3.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 4.1, h: 1.6, fill: { color: CARD }, rectRadius: 0.1 });
  s3.addText(s.num, { x: x + 0.3, y: y + 0.2, w: 1.5, h: 0.5, fontSize: 32, fontFace: "Arial Black", color: GREEN, bold: true, margin: 0 });
  s3.addText(s.label, { x: x + 0.3, y: y + 0.7, w: 3.5, h: 0.3, fontSize: 16, fontFace: "Arial", color: WHITE, bold: true, margin: 0 });
  s3.addText(s.desc, { x: x + 0.3, y: y + 1.0, w: 3.5, h: 0.4, fontSize: 11, fontFace: "Arial", color: MUTED, margin: 0 });
});

// ═══════════════════════════════════════════════
// SLIDE 4: The 6 Signals
// ═══════════════════════════════════════════════
let s4 = darkSlide();
s4.addText("The 6 Signals", { x: 0.7, y: 0.4, w: 8, h: 0.6, fontSize: 36, fontFace: "Arial Black", color: WHITE, bold: true, margin: 0 });
s4.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 1.05, w: 1.5, h: 0.04, fill: { color: ACCENT } });

const signals = [
  { name: "Polymarket", weight: "20%", source: "Gamma API", color: BLUE },
  { name: "Order Flow", weight: "15%", source: "Binance.US + Coinbase", color: ACCENT },
  { name: "Our Model", weight: "20%", source: "BGeometrics on-chain", color: GREEN },
  { name: "Technical", weight: "20%", source: "TrueMarkets MCP", color: YELLOW },
  { name: "TM Sentiment", weight: "10%", source: "TrueMarkets MCP", color: "A855F7" },
  { name: "Fear & Greed", weight: "15%", source: "alternative.me", color: RED },
];
// Header
s4.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: 1.4, w: 8.6, h: 0.45, fill: { color: BORDER }, rectRadius: 0.05 });
s4.addText("Signal", { x: 0.9, y: 1.42, w: 2.5, h: 0.4, fontSize: 12, fontFace: "Arial", color: MUTED, bold: true, margin: 0 });
s4.addText("Weight", { x: 3.5, y: 1.42, w: 1.2, h: 0.4, fontSize: 12, fontFace: "Arial", color: MUTED, bold: true, margin: 0 });
s4.addText("Source", { x: 5.0, y: 1.42, w: 3, h: 0.4, fontSize: 12, fontFace: "Arial", color: MUTED, bold: true, margin: 0 });
s4.addText("Update", { x: 8.2, y: 1.42, w: 1, h: 0.4, fontSize: 12, fontFace: "Arial", color: MUTED, bold: true, margin: 0 });

signals.forEach((sig, i) => {
  const y = 1.95 + i * 0.55;
  s4.addShape(pres.shapes.OVAL, { x: 0.9, y: y + 0.1, w: 0.2, h: 0.2, fill: { color: sig.color } });
  s4.addText(sig.name, { x: 1.25, y, w: 2.2, h: 0.4, fontSize: 14, fontFace: "Arial", color: WHITE, bold: true, margin: 0 });
  s4.addText(sig.weight, { x: 3.5, y, w: 1.2, h: 0.4, fontSize: 14, fontFace: "Arial", color: GREEN, bold: true, margin: 0 });
  s4.addText(sig.source, { x: 5.0, y, w: 3, h: 0.4, fontSize: 12, fontFace: "Arial", color: LIGHT, margin: 0 });
  s4.addText("30s", { x: 8.2, y, w: 1, h: 0.4, fontSize: 12, fontFace: "Arial", color: MUTED, margin: 0 });
});

// ═══════════════════════════════════════════════
// SLIDE 5: Backtest Results
// ═══════════════════════════════════════════════
let s5 = darkSlide();
s5.addText("67.4% Accuracy", { x: 0.7, y: 0.3, w: 6, h: 0.7, fontSize: 40, fontFace: "Arial Black", color: GREEN, bold: true, margin: 0 });
s5.addText("on 359 Out-of-Sample Days", { x: 0.7, y: 0.95, w: 6, h: 0.4, fontSize: 18, fontFace: "Arial", color: MUTED, margin: 0 });

// Left: details
const details = [
  "GradientBoosting + Boruta feature selection",
  "216 features: 42 on-chain + 168 rate-of-change + 6 price",
  "Data: 5 years daily from BGeometrics (27 endpoints)",
  "Train: 1,435 days (May 2021 - Apr 2025)",
  "Test: 359 days (Apr 2025 - Apr 2026)",
];
s5.addText(details.map((d, i) => ({ text: d, options: { bullet: true, breakLine: i < details.length - 1, fontSize: 12, color: LIGHT } })),
  { x: 0.7, y: 1.5, w: 5, h: 2.5, fontFace: "Arial" });

// Right: metrics table
s5.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 5.8, y: 1.5, w: 3.7, h: 3.5, fill: { color: CARD }, rectRadius: 0.1 });
s5.addText("Metrics", { x: 6.0, y: 1.6, w: 3, h: 0.3, fontSize: 14, fontFace: "Arial", color: ACCENT, bold: true, margin: 0 });

const metrics = [
  ["Accuracy", "67.4%"], ["Precision", "67.2%"], ["Recall", "66.9%"],
  ["F1 Score", "67.0%"], ["AUC-ROC", "0.7365"], ["MCC", "0.348"],
];
metrics.forEach((m, i) => {
  const y = 2.05 + i * 0.4;
  s5.addText(m[0], { x: 6.0, y, w: 1.8, h: 0.35, fontSize: 12, fontFace: "Arial", color: MUTED, margin: 0 });
  s5.addText(m[1], { x: 7.8, y, w: 1.5, h: 0.35, fontSize: 14, fontFace: "Arial", color: WHITE, bold: true, align: "right", margin: 0 });
});

// Confusion matrix
s5.addText("Confusion Matrix", { x: 6.0, y: 4.1, w: 3, h: 0.25, fontSize: 11, fontFace: "Arial", color: ACCENT, bold: true, margin: 0 });
s5.addText("TN=123  FP=58  |  FN=59  TP=119", { x: 6.0, y: 4.4, w: 3.3, h: 0.25, fontSize: 10, fontFace: "Consolas", color: MUTED, margin: 0 });

// ═══════════════════════════════════════════════
// SLIDE 6: Key Innovation
// ═══════════════════════════════════════════════
let s6 = darkSlide();
s6.addText("Key Innovation", { x: 0.7, y: 0.4, w: 8, h: 0.6, fontSize: 36, fontFace: "Arial Black", color: WHITE, bold: true, margin: 0 });
s6.addText("Rate-of-Change Features", { x: 0.7, y: 0.95, w: 8, h: 0.4, fontSize: 18, fontFace: "Arial", color: ACCENT, margin: 0 });

s6.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: 1.6, w: 8.6, h: 1.0, fill: { color: CARD }, rectRadius: 0.1 });
s6.addText("Raw on-chain values (MVRV, NUPL) are weakly predictive. Their RATE OF CHANGE is the breakthrough.",
  { x: 0.9, y: 1.7, w: 8.2, h: 0.8, fontSize: 14, fontFace: "Arial", color: LIGHT, margin: 0 });

const features = [
  { name: "mvrv_ratio_chg1d", pct: "14.3%", desc: "1-day change in MVRV ratio" },
  { name: "hash_ribbons", pct: "10.4%", desc: "Hash rate SMA30/SMA60 crossover" },
  { name: "puell_multiple_chg1d", pct: "4.8%", desc: "1-day change in Puell multiple" },
  { name: "hodl_age_10y", pct: "4.5%", desc: "Coins held 10+ years (deep holders)" },
  { name: "avg_dormancy", pct: "3.7%", desc: "Average coin dormancy (holding time)" },
];
features.forEach((f, i) => {
  const y = 2.9 + i * 0.5;
  const barW = parseFloat(f.pct) / 14.3 * 5;
  s6.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y, w: barW, h: 0.35, fill: { color: GREEN }, rectRadius: 0.05 });
  s6.addText(f.pct, { x: 0.9, y, w: 0.8, h: 0.35, fontSize: 11, fontFace: "Arial", color: BG, bold: true, margin: 0 });
  s6.addText(f.name, { x: barW + 1.0, y, w: 3, h: 0.2, fontSize: 11, fontFace: "Consolas", color: WHITE, margin: 0 });
  s6.addText(f.desc, { x: barW + 1.0, y: y + 0.18, w: 4, h: 0.2, fontSize: 9, fontFace: "Arial", color: MUTED, margin: 0 });
});

// ═══════════════════════════════════════════════
// SLIDE 7: Tech Stack
// ═══════════════════════════════════════════════
let s7 = darkSlide();
s7.addText("Tech Stack", { x: 0.7, y: 0.4, w: 8, h: 0.6, fontSize: 36, fontFace: "Arial Black", color: WHITE, bold: true, margin: 0 });
s7.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 1.05, w: 1.5, h: 0.04, fill: { color: BLUE } });

const stack = [
  { cat: "Backend", items: "FastAPI + Python, scikit-learn, PyTorch", color: GREEN },
  { cat: "Frontend", items: "Next.js 14, React 18, TailwindCSS", color: BLUE },
  { cat: "Data", items: "BGeometrics Premium, Binance.US, Coinbase, Polymarket", color: ACCENT },
  { cat: "Price", items: "TrueMarkets MCP (same source everywhere)", color: YELLOW },
  { cat: "Refresh", items: "30 seconds, all 6 signals", color: RED },
];
stack.forEach((s, i) => {
  const y = 1.4 + i * 0.8;
  s7.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y, w: 8.6, h: 0.65, fill: { color: CARD }, rectRadius: 0.08 });
  s7.addShape(pres.shapes.RECTANGLE, { x: 0.7, y, w: 0.06, h: 0.65, fill: { color: s.color } });
  s7.addText(s.cat, { x: 1.1, y: y + 0.05, w: 2, h: 0.3, fontSize: 15, fontFace: "Arial", color: WHITE, bold: true, margin: 0 });
  s7.addText(s.items, { x: 1.1, y: y + 0.33, w: 7.5, h: 0.25, fontSize: 12, fontFace: "Arial", color: MUTED, margin: 0 });
});

// ═══════════════════════════════════════════════
// SLIDE 8: Product
// ═══════════════════════════════════════════════
let s8 = darkSlide();
s8.addText("The Product", { x: 0.7, y: 0.4, w: 8, h: 0.6, fontSize: 36, fontFace: "Arial Black", color: WHITE, bold: true, margin: 0 });
s8.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 1.05, w: 1.5, h: 0.04, fill: { color: GREEN } });

// Two columns
s8.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: 1.4, w: 4.1, h: 3.8, fill: { color: CARD }, rectRadius: 0.1 });
s8.addText("Market Page", { x: 0.9, y: 1.5, w: 3.5, h: 0.4, fontSize: 18, fontFace: "Arial", color: GREEN, bold: true, margin: 0 });
s8.addText([
  { text: "Live BTC price + 24h chart", options: { bullet: true, breakLine: true } },
  { text: "Market cap, volume, highs/lows", options: { bullet: true, breakLine: true } },
  { text: "Fear & Greed Index gauge", options: { bullet: true, breakLine: true } },
  { text: "Performance: 7d, 30d, 1y", options: { bullet: true, breakLine: true } },
  { text: "TM Sentiment from 30+ news sources", options: { bullet: true } },
], { x: 0.9, y: 2.0, w: 3.7, h: 2.5, fontSize: 12, fontFace: "Arial", color: LIGHT });

s8.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 5.2, y: 1.4, w: 4.3, h: 3.8, fill: { color: CARD }, rectRadius: 0.1 });
s8.addText("Prediction Page", { x: 5.4, y: 1.5, w: 3.5, h: 0.4, fontSize: 18, fontFace: "Arial", color: ACCENT, bold: true, margin: 0 });
s8.addText([
  { text: "BUY / SELL with 6-signal reasoning", options: { bullet: true, breakLine: true } },
  { text: "Polymarket probability table (18 thresholds)", options: { bullet: true, breakLine: true } },
  { text: "Binance order flow (buy vs sell volume)", options: { bullet: true, breakLine: true } },
  { text: "RSI, MACD, Bollinger live indicators", options: { bullet: true, breakLine: true } },
  { text: "Portfolio + order management", options: { bullet: true } },
], { x: 5.4, y: 2.0, w: 3.9, h: 2.5, fontSize: 12, fontFace: "Arial", color: LIGHT });

s8.addText("Same price source (TrueMarkets MCP) on both pages", { x: 0.7, y: 5.05, w: 9, h: 0.3, fontSize: 11, fontFace: "Arial", color: MUTED, align: "center", margin: 0 });

// ═══════════════════════════════════════════════
// SLIDE 9: The Ask
// ═══════════════════════════════════════════════
let s9 = darkSlide();
s9.addText("The Ask", { x: 0.7, y: 0.4, w: 8, h: 0.6, fontSize: 36, fontFace: "Arial Black", color: WHITE, bold: true, margin: 0 });
s9.addShape(pres.shapes.RECTANGLE, { x: 0.7, y: 1.05, w: 1.5, h: 0.04, fill: { color: ACCENT } });

s9.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.7, y: 1.4, w: 4, h: 1.5, fill: { color: CARD }, rectRadius: 0.1 });
s9.addText("$500K", { x: 0.9, y: 1.5, w: 3.5, h: 0.6, fontSize: 40, fontFace: "Arial Black", color: GREEN, bold: true, margin: 0 });
s9.addText("at $5M valuation", { x: 0.9, y: 2.1, w: 3.5, h: 0.3, fontSize: 16, fontFace: "Arial", color: MUTED, margin: 0 });

s9.addText("Use of Funds", { x: 0.7, y: 3.2, w: 8, h: 0.4, fontSize: 18, fontFace: "Arial", color: WHITE, bold: true, margin: 0 });

const funds = [
  { item: "Glassnode Professional ($799/mo)", desc: "87 on-chain features, target 82%+ accuracy", color: GREEN },
  { item: "Expand to ETH, SOL, top 10 coins", desc: "Same 6-signal architecture, per-coin models", color: BLUE },
  { item: "Mobile app (React Native)", desc: "Push notifications on signal changes", color: ACCENT },
  { item: "Team: ML engineer + full-stack dev", desc: "Scale infrastructure, add more data sources", color: YELLOW },
];
funds.forEach((f, i) => {
  const y = 3.7 + i * 0.45;
  s9.addShape(pres.shapes.OVAL, { x: 0.9, y: y + 0.08, w: 0.18, h: 0.18, fill: { color: f.color } });
  s9.addText(f.item, { x: 1.3, y, w: 4, h: 0.25, fontSize: 13, fontFace: "Arial", color: WHITE, bold: true, margin: 0 });
  s9.addText(f.desc, { x: 5.5, y, w: 4, h: 0.25, fontSize: 11, fontFace: "Arial", color: MUTED, margin: 0 });
});

// Save
pres.writeFile({ fileName: "/Users/saieshagupta/Desktop/claude/truemarkets-predictions/TrueMarkets_Pitch_Deck.pptx" })
  .then(() => console.log("Saved: TrueMarkets_Pitch_Deck.pptx"))
  .catch(e => console.error(e));
