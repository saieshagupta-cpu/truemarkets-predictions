const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const { FaBrain, FaChartLine, FaExchangeAlt, FaNewspaper, FaBolt, FaRocket, FaMobileAlt, FaCoins, FaExclamationTriangle, FaDice, FaTwitter, FaChartBar } = require("react-icons/fa");

// ── Colors ──
const BG = "0F0F1A";
const BG2 = "161627";
const CARD = "1E1E36";
const TEAL = "00D4AA";
const RED = "FF6B6B";
const YELLOW = "FFE66D";
const PURPLE = "A29BFE";
const BLUE = "4ECDC4";
const WHITE = "FFFFFF";
const MUTED = "8B8BA3";
const DIM = "6B6B80";

function renderIconSvg(IconComponent, color, size = 256) {
  return ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color: "#" + color, size: String(size) })
  );
}

async function iconToBase64Png(IconComponent, color, size = 256) {
  const svg = renderIconSvg(IconComponent, color, size);
  const pngBuffer = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + pngBuffer.toString("base64");
}

async function main() {
  let pres = new pptxgen();
  pres.layout = "LAYOUT_16x9";
  pres.author = "True Markets";
  pres.title = "True Markets - YC Pitch Deck";

  // Pre-render icons
  const icons = {
    brain: await iconToBase64Png(FaBrain, TEAL),
    chart: await iconToBase64Png(FaChartLine, YELLOW),
    exchange: await iconToBase64Png(FaExchangeAlt, PURPLE),
    news: await iconToBase64Png(FaNewspaper, BLUE),
    bolt: await iconToBase64Png(FaBolt, TEAL),
    rocket: await iconToBase64Png(FaRocket, TEAL),
    mobile: await iconToBase64Png(FaMobileAlt, TEAL),
    coins: await iconToBase64Png(FaCoins, YELLOW),
    warning: await iconToBase64Png(FaExclamationTriangle, RED),
    dice: await iconToBase64Png(FaDice, RED),
    twitter: await iconToBase64Png(FaTwitter, RED),
    bar: await iconToBase64Png(FaChartBar, RED),
  };

  // ════════════════════════════════════════════════════════
  // SLIDE 1: Title / Hero
  // ════════════════════════════════════════════════════════
  let s1 = pres.addSlide();
  s1.background = { color: BG };

  // Subtle accent bar at top
  s1.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.04, fill: { color: TEAL } });

  // Large teal circle decoration (subtle background element)
  s1.addShape(pres.shapes.OVAL, {
    x: 6.5, y: 0.5, w: 4.5, h: 4.5,
    fill: { color: TEAL, transparency: 92 },
  });

  // Title
  s1.addText("True Markets", {
    x: 0.8, y: 1.0, w: 8, h: 1.2,
    fontSize: 52, fontFace: "Arial Black", color: WHITE, bold: true, margin: 0,
  });

  // Subtitle
  s1.addText("AI-Powered BTC Direction Prediction", {
    x: 0.8, y: 2.1, w: 8, h: 0.6,
    fontSize: 22, fontFace: "Calibri", color: MUTED, margin: 0,
  });

  // Hero stat
  s1.addShape(pres.shapes.RECTANGLE, {
    x: 0.8, y: 3.0, w: 4.5, h: 1.1,
    fill: { color: TEAL, transparency: 85 },
  });
  s1.addText([
    { text: "95%", options: { fontSize: 44, bold: true, color: TEAL } },
    { text: "  validated accuracy on consensus signals", options: { fontSize: 16, color: MUTED } },
  ], { x: 1.0, y: 3.05, w: 4.3, h: 1.0, valign: "middle", margin: 0 });

  // Tagline
  s1.addText("Replacing gut instinct with backtested signals.", {
    x: 0.8, y: 4.5, w: 6, h: 0.5,
    fontSize: 14, fontFace: "Calibri", color: DIM, italic: true, margin: 0,
  });

  // ════════════════════════════════════════════════════════
  // SLIDE 2: The Problem
  // ════════════════════════════════════════════════════════
  let s2 = pres.addSlide();
  s2.background = { color: BG };
  s2.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.04, fill: { color: RED } });

  s2.addText("The Problem", {
    x: 0.8, y: 0.3, w: 8, h: 0.7,
    fontSize: 36, fontFace: "Arial Black", color: WHITE, bold: true, margin: 0,
  });

  // Big stat
  s2.addText("90%", {
    x: 6.5, y: 0.8, w: 3, h: 1.5,
    fontSize: 72, fontFace: "Arial Black", color: RED, bold: true, align: "center", margin: 0,
  });
  s2.addText("of retail crypto\ntraders lose money", {
    x: 6.5, y: 2.2, w: 3, h: 0.8,
    fontSize: 14, color: MUTED, align: "center", margin: 0,
  });

  // Problem cards
  const problems = [
    { icon: icons.dice, title: "Gut Feeling", desc: "Traders rely on intuition\nand FOMO-driven decisions" },
    { icon: icons.twitter, title: "Twitter Alpha", desc: "Following influencers with\nno track record or backtests" },
    { icon: icons.bar, title: "Single Indicators", desc: "RSI alone, MACD alone\u2014\nno multi-signal approach" },
  ];

  for (let i = 0; i < problems.length; i++) {
    const y = 1.2 + i * 1.3;
    s2.addShape(pres.shapes.RECTANGLE, {
      x: 0.8, y: y, w: 5.2, h: 1.1,
      fill: { color: CARD },
    });
    s2.addImage({ data: problems[i].icon, x: 1.05, y: y + 0.25, w: 0.5, h: 0.5 });
    s2.addText(problems[i].title, {
      x: 1.75, y: y + 0.1, w: 3.5, h: 0.4,
      fontSize: 16, fontFace: "Calibri", color: WHITE, bold: true, margin: 0,
    });
    s2.addText(problems[i].desc, {
      x: 1.75, y: y + 0.5, w: 4, h: 0.5,
      fontSize: 12, fontFace: "Calibri", color: MUTED, margin: 0,
    });
  }

  s2.addText("Current tools show data.\nThey don\u2019t tell you what to do.", {
    x: 0.8, y: 4.8, w: 8, h: 0.5,
    fontSize: 14, fontFace: "Calibri", color: DIM, italic: true, margin: 0,
  });

  // ════════════════════════════════════════════════════════
  // SLIDE 3: Solution / How It Works
  // ════════════════════════════════════════════════════════
  let s3 = pres.addSlide();
  s3.background = { color: BG };
  s3.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.04, fill: { color: TEAL } });

  s3.addText("4-Signal Ensemble", {
    x: 0.8, y: 0.3, w: 8, h: 0.7,
    fontSize: 32, fontFace: "Arial Black", color: WHITE, bold: true, margin: 0,
  });
  s3.addText("Backtested weights on 2 years of daily BTC (Apr 2023 \u2013 Apr 2025)", {
    x: 0.8, y: 0.9, w: 8, h: 0.4,
    fontSize: 13, fontFace: "Calibri", color: MUTED, margin: 0,
  });

  // Signal cards (2x2 grid)
  const signals = [
    { icon: icons.chart, color: YELLOW, name: "Technical", weight: "40%", desc: "RSI mean-reversion at extremes\nMACD trend direction", x: 0.8, y: 1.5 },
    { icon: icons.brain, color: TEAL, name: "TCN Model", weight: "30%", desc: "Temporal Convolutional Network\n3 years of BTC training data", x: 5.2, y: 1.5 },
    { icon: icons.exchange, color: PURPLE, name: "Order Flow", weight: "20%", desc: "Polymarket buy/sell pressure\nTrue Markets order history", x: 0.8, y: 3.2 },
    { icon: icons.news, color: BLUE, name: "Sentiment", weight: "10%", desc: "True Markets AI (30+ sources)\nFear & Greed contrarian", x: 5.2, y: 3.2 },
  ];

  for (const sig of signals) {
    // Card background
    s3.addShape(pres.shapes.RECTANGLE, {
      x: sig.x, y: sig.y, w: 4.2, h: 1.4,
      fill: { color: CARD },
    });
    // Weight accent bar
    s3.addShape(pres.shapes.RECTANGLE, {
      x: sig.x, y: sig.y, w: 0.06, h: 1.4,
      fill: { color: sig.color },
    });
    // Icon
    s3.addImage({ data: sig.icon, x: sig.x + 0.3, y: sig.y + 0.35, w: 0.55, h: 0.55 });
    // Name + weight
    s3.addText([
      { text: sig.name, options: { fontSize: 16, bold: true, color: WHITE } },
      { text: `  ${sig.weight}`, options: { fontSize: 20, bold: true, color: sig.color } },
    ], { x: sig.x + 1.0, y: sig.y + 0.15, w: 3, h: 0.45, margin: 0 });
    // Description
    s3.addText(sig.desc, {
      x: sig.x + 1.0, y: sig.y + 0.65, w: 3, h: 0.6,
      fontSize: 11, color: MUTED, margin: 0,
    });
  }

  // Flow arrow at bottom
  s3.addText("Signals  \u2192  Weighted Blend  \u2192  BUY / SELL / HOLD  +  Reasoning", {
    x: 0.8, y: 4.9, w: 8.5, h: 0.4,
    fontSize: 13, fontFace: "Calibri", color: TEAL, align: "center", margin: 0,
  });

  // ════════════════════════════════════════════════════════
  // SLIDE 4: Performance / Results
  // ════════════════════════════════════════════════════════
  let s4 = pres.addSlide();
  s4.background = { color: BG };
  s4.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.04, fill: { color: YELLOW } });

  s4.addText("Backtest Results", {
    x: 0.8, y: 0.3, w: 8, h: 0.7,
    fontSize: 36, fontFace: "Arial Black", color: WHITE, bold: true, margin: 0,
  });

  // Big metric cards (3 across top)
  const topMetrics = [
    { value: "95%", label: "Consensus\nAccuracy", color: TEAL },
    { value: "51.1", label: "Sharpe\nRatio", color: YELLOW },
    { value: "8.06x", label: "Profit\nFactor", color: PURPLE },
  ];

  for (let i = 0; i < topMetrics.length; i++) {
    const x = 0.8 + i * 3.1;
    s4.addShape(pres.shapes.RECTANGLE, {
      x: x, y: 1.2, w: 2.8, h: 1.6,
      fill: { color: CARD },
    });
    s4.addText(topMetrics[i].value, {
      x: x, y: 1.3, w: 2.8, h: 0.9,
      fontSize: 44, fontFace: "Arial Black", color: topMetrics[i].color, bold: true, align: "center", margin: 0,
    });
    s4.addText(topMetrics[i].label, {
      x: x, y: 2.15, w: 2.8, h: 0.5,
      fontSize: 12, color: MUTED, align: "center", margin: 0,
    });
  }

  // Bottom details (2 columns)
  const leftDetails = [
    "Trained on 3 years of daily BTC (1,096 candles)",
    "Walk-forward validated across 23 monthly folds",
    "TCN model: 80%+ validated directional accuracy",
  ];
  const rightDetails = [
    "Multi-horizon consensus: 1h, 2h, 3h, 4h, 6h",
    "Filters out 38% of noisy / ambiguous signals",
    "Separate threshold model for Polymarket comparison",
  ];

  for (let i = 0; i < leftDetails.length; i++) {
    s4.addShape(pres.shapes.OVAL, { x: 1.0, y: 3.2 + i * 0.55, w: 0.12, h: 0.12, fill: { color: TEAL } });
    s4.addText(leftDetails[i], {
      x: 1.3, y: 3.1 + i * 0.55, w: 4, h: 0.4,
      fontSize: 11, color: MUTED, margin: 0,
    });
  }
  for (let i = 0; i < rightDetails.length; i++) {
    s4.addShape(pres.shapes.OVAL, { x: 5.5, y: 3.2 + i * 0.55, w: 0.12, h: 0.12, fill: { color: YELLOW } });
    s4.addText(rightDetails[i], {
      x: 5.8, y: 3.1 + i * 0.55, w: 4, h: 0.4,
      fontSize: 11, color: MUTED, margin: 0,
    });
  }

  s4.addText("Only predicts when signals have high conviction \u2014 quality over quantity.", {
    x: 0.8, y: 4.8, w: 8.5, h: 0.4,
    fontSize: 13, fontFace: "Calibri", color: DIM, italic: true, margin: 0,
  });

  // ════════════════════════════════════════════════════════
  // SLIDE 4B: TCN Performance — Visual Proof
  // ════════════════════════════════════════════════════════
  const CHARTS_DIR = "/Users/saieshagupta/Desktop/claude/truemarkets-predictions/backend/backtest_results";

  let s4b = pres.addSlide();
  s4b.background = { color: BG };
  s4b.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.04, fill: { color: TEAL } });

  // TCN vs everyone — the money chart
  s4b.addImage({
    path: CHARTS_DIR + "/yc_accuracy.png",
    x: 0.3, y: 0.15, w: 5.0, h: 2.75,
  });

  // Consensus = accuracy — why we filter
  s4b.addImage({
    path: CHARTS_DIR + "/yc_consensus.png",
    x: 5.0, y: 0.15, w: 4.8, h: 2.75,
  });

  // Cumulative returns — full width
  s4b.addImage({
    path: CHARTS_DIR + "/yc_returns.png",
    x: 0.3, y: 3.0, w: 9.4, h: 2.5,
  });

  // ════════════════════════════════════════════════════════
  // SLIDE 6: Live Product + What's Next
  // ════════════════════════════════════════════════════════
  let s5 = pres.addSlide();
  s5.background = { color: BG };
  s5.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.04, fill: { color: TEAL } });

  s5.addText("Live Product", {
    x: 0.8, y: 0.3, w: 5, h: 0.7,
    fontSize: 36, fontFace: "Arial Black", color: WHITE, bold: true, margin: 0,
  });
  s5.addText("Updating every 30 seconds", {
    x: 0.8, y: 0.85, w: 5, h: 0.4,
    fontSize: 15, color: TEAL, margin: 0,
  });

  // Live features (left column)
  const liveFeatures = [
    { icon: icons.bolt, text: "Real-time BTC price from True Markets API" },
    { icon: icons.brain, text: "Auto-refreshing TCN predictions every 30s" },
    { icon: icons.exchange, text: "Polymarket mispricing detection" },
    { icon: icons.chart, text: "Clear BUY/SELL with signal-by-signal reasoning" },
  ];

  for (let i = 0; i < liveFeatures.length; i++) {
    const y = 1.5 + i * 0.7;
    s5.addImage({ data: liveFeatures[i].icon, x: 1.0, y: y, w: 0.35, h: 0.35 });
    s5.addText(liveFeatures[i].text, {
      x: 1.55, y: y - 0.02, w: 4, h: 0.4,
      fontSize: 13, color: WHITE, margin: 0,
    });
  }

  // What's Next (right column)
  s5.addShape(pres.shapes.RECTANGLE, {
    x: 5.8, y: 1.3, w: 3.8, h: 2.8,
    fill: { color: CARD },
  });
  s5.addText("What\u2019s Next", {
    x: 6.1, y: 1.45, w: 3.2, h: 0.4,
    fontSize: 18, fontFace: "Calibri", color: TEAL, bold: true, margin: 0,
  });

  const nextItems = [
    { icon: icons.coins, text: "Multi-coin (ETH, SOL configured)" },
    { icon: icons.rocket, text: "Trading via True Markets Gateway" },
    { icon: icons.mobile, text: "Mobile app" },
  ];

  for (let i = 0; i < nextItems.length; i++) {
    const y = 2.1 + i * 0.6;
    s5.addImage({ data: nextItems[i].icon, x: 6.2, y: y, w: 0.3, h: 0.3 });
    s5.addText(nextItems[i].text, {
      x: 6.7, y: y - 0.02, w: 2.7, h: 0.35,
      fontSize: 12, color: MUTED, margin: 0,
    });
  }

  // Footer
  s5.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 4.8, w: 10, h: 0.825,
    fill: { color: CARD },
  });
  s5.addText("All data exclusively from True Markets API. Zero external dependencies.", {
    x: 0.8, y: 4.95, w: 8.5, h: 0.4,
    fontSize: 14, fontFace: "Calibri", color: TEAL, align: "center", margin: 0,
  });
  s5.addText("truemarkets.co", {
    x: 0.8, y: 5.25, w: 8.5, h: 0.3,
    fontSize: 11, color: DIM, align: "center", margin: 0,
  });

  // ── Save ──
  const outputPath = "/Users/saieshagupta/Desktop/claude/truemarkets-predictions/TrueMarkets_YC_Deck.pptx";
  await pres.writeFile({ fileName: outputPath });
  console.log("Saved to: " + outputPath);
}

main().catch(console.error);
