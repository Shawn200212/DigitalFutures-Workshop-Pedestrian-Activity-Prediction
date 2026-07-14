const DATA = window.MODEL_VIZ_DATA;

const state = {
  view: "2d",
  layer: "pred",
  splits: new Set(["train", "val", "test"]),
  tile: "all",
  labeledOnly: false,
  hotspotOnly: false,
  pointSize: 2.4,
  heightScale: 1,
  colorMax: "p99",
  zoom: 1,
  panX: 0,
  panY: 0,
  yaw: -0.72,
  pitch: 0.82,
  autoOrbit: true,
  dragging: false,
  lastX: 0,
  lastY: 0,
  hoverPoint: null,
  lastFrame: 0,
};

const canvas = document.getElementById("mapCanvas");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");

const layerDefs = {
  pred: { title: "连续预测强度", field: "pred", palette: "viridis", requiresLabel: false, min: 0 },
  true: { title: "真实标签强度", field: "true", palette: "viridis", requiresLabel: true, min: 0 },
  residual: { title: "残差", field: "residual", palette: "diverging", requiresLabel: true, symmetric: true },
  absError: { title: "绝对误差", field: "absError", palette: "magma", requiresLabel: true, min: 0 },
  y5Score: { title: "y >= 5 热点分数", field: "y5Score", palette: "magma", requiresLabel: false, min: 0, max: 1 },
  y3Score: { title: "y >= 3 中高分数", field: "y3Score", palette: "magma", requiresLabel: false, min: 0, max: 1 },
  y5Class: { title: "y >= 5 分类", field: "y5Class", palette: "class", requiresLabel: false, min: 0, max: 1 },
  y3Class: { title: "y >= 3 分类", field: "y3Class", palette: "class", requiresLabel: false, min: 0, max: 1 },
};

const palettes = {
  viridis: [
    [0, "#482878"],
    [0.25, "#31688e"],
    [0.5, "#1f9e89"],
    [0.75, "#6ece58"],
    [1, "#fde725"],
  ],
  magma: [
    [0, "#1d1147"],
    [0.25, "#721f81"],
    [0.5, "#b73779"],
    [0.75, "#f1605d"],
    [1, "#fcfdbf"],
  ],
  diverging: [
    [0, "#2f6fb0"],
    [0.5, "#f7f7f7"],
    [1, "#c83f3f"],
  ],
};

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function hexToRgb(hex) {
  const clean = hex.replace("#", "");
  return {
    r: parseInt(clean.slice(0, 2), 16),
    g: parseInt(clean.slice(2, 4), 16),
    b: parseInt(clean.slice(4, 6), 16),
  };
}

function mix(a, b, t) {
  const ca = hexToRgb(a);
  const cb = hexToRgb(b);
  const r = Math.round(ca.r + (cb.r - ca.r) * t);
  const g = Math.round(ca.g + (cb.g - ca.g) * t);
  const bl = Math.round(ca.b + (cb.b - ca.b) * t);
  return `rgb(${r},${g},${bl})`;
}

function gradientColor(value, min, max, stops) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "rgba(148,163,184,0.35)";
  if (max <= min) return stops[stops.length - 1][1];
  const t = Math.max(0, Math.min(1, (value - min) / (max - min)));
  for (let i = 0; i < stops.length - 1; i += 1) {
    const [p0, c0] = stops[i];
    const [p1, c1] = stops[i + 1];
    if (t >= p0 && t <= p1) return mix(c0, c1, (t - p0) / Math.max(1e-9, p1 - p0));
  }
  return stops[stops.length - 1][1];
}

function layerValue(point) {
  const def = layerDefs[state.layer];
  if (def.requiresLabel && !point.hasLabel) return null;
  return point[def.field];
}

function visiblePoints() {
  return DATA.points.filter((point) => {
    if (!state.splits.has(point.split)) return false;
    if (state.tile !== "all" && point.tile !== state.tile) return false;
    if (state.labeledOnly && !point.hasLabel) return false;
    if (state.hotspotOnly && state.layer.includes("y5") && point.y5Class !== 1) return false;
    if (state.hotspotOnly && state.layer.includes("y3") && point.y3Class !== 1) return false;
    const value = layerValue(point);
    return value !== null && value !== undefined && !Number.isNaN(Number(value));
  });
}

function quantile(values, q) {
  if (!values.length) return 1;
  const sorted = [...values].sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sorted[base + 1] !== undefined) return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  return sorted[base];
}

function colorRange(points) {
  const def = layerDefs[state.layer];
  if (def.palette === "class") return { min: 0, max: 1 };
  if (def.max !== undefined) return { min: def.min ?? 0, max: def.max };
  const values = points.map(layerValue).filter((v) => v !== null && Number.isFinite(Number(v))).map(Number);
  if (!values.length) return { min: 0, max: 1 };
  if (def.symmetric) {
    const absMax = quantile(values.map((v) => Math.abs(v)), 0.99) || 1;
    return { min: -absMax, max: absMax };
  }
  let max = Math.max(...values);
  if (state.colorMax === "p95") max = quantile(values, 0.95);
  if (state.colorMax === "p99") max = quantile(values, 0.99);
  if (state.colorMax === "fixed5") max = 5;
  if (state.colorMax === "fixed10") max = 10;
  return { min: def.min ?? Math.min(...values), max: Math.max(max, 1e-6) };
}

function pointColor(point, range) {
  const def = layerDefs[state.layer];
  const value = layerValue(point);
  if (def.palette === "class") {
    return Number(value) === 1 ? "rgba(218,73,73,0.9)" : "rgba(120,135,156,0.42)";
  }
  return gradientColor(Number(value), range.min, range.max, palettes[def.palette]);
}

function withAlpha(color, alpha) {
  if (color.startsWith("rgba(")) {
    return color.replace(/rgba\(([^)]+),\s*[\d.]+\)/, `rgba($1,${alpha})`);
  }
  if (color.startsWith("rgb(")) {
    return color.replace("rgb(", "rgba(").replace(")", `,${alpha})`);
  }
  return color;
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(640, Math.floor(rect.width));
  const height = Math.max(540, Math.floor(rect.height));
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function baseProject(point) {
  const b = DATA.summary.bounds;
  const rect = canvas.getBoundingClientRect();
  const pad = 28;
  const dx = Math.max(1e-9, b.maxLon - b.minLon);
  const dy = Math.max(1e-9, b.maxLat - b.minLat);
  const scale = Math.min((rect.width - pad * 2) / dx, (rect.height - pad * 2) / dy);
  const mapW = dx * scale;
  const mapH = dy * scale;
  const extraX = (rect.width - pad * 2 - mapW) / 2;
  const extraY = (rect.height - pad * 2 - mapH) / 2;
  return {
    x: pad + extraX + (point.lon - b.minLon) * scale,
    y: pad + extraY + (b.maxLat - point.lat) * scale,
  };
}

function project2d(point) {
  const rect = canvas.getBoundingClientRect();
  const base = baseProject(point);
  const cx = rect.width / 2;
  const cy = rect.height / 2;
  return {
    x: cx + (base.x - cx) * state.zoom + state.panX,
    y: cy + (base.y - cy) * state.zoom + state.panY,
  };
}

function normalizedWorld(point) {
  const b = DATA.summary.bounds;
  const dx = Math.max(1e-9, b.maxLon - b.minLon);
  const dy = Math.max(1e-9, b.maxLat - b.minLat);
  const aspect = dx / dy;
  return {
    x: ((point.lon - (b.minLon + b.maxLon) / 2) / dx) * 2.2 * aspect,
    z: ((point.lat - (b.minLat + b.maxLat) / 2) / dy) * 2.2,
  };
}

function heightValue(point, range) {
  const value = Number(layerValue(point));
  if (!Number.isFinite(value)) return 0;
  if (layerDefs[state.layer].symmetric) {
    return Math.min(1.45, Math.abs(value) / Math.max(1e-6, Math.max(Math.abs(range.min), Math.abs(range.max)))) * state.heightScale;
  }
  return Math.max(0, Math.min(1.45, (value - range.min) / Math.max(1e-6, range.max - range.min))) * state.heightScale;
}

function project3dWorld(x, y, z) {
  const rect = canvas.getBoundingClientRect();
  const cy = Math.cos(state.yaw);
  const sy = Math.sin(state.yaw);
  const cp = Math.cos(state.pitch);
  const sp = Math.sin(state.pitch);
  const xr = x * cy - z * sy;
  const zr = x * sy + z * cy;
  const yr = y;
  const y2 = yr * cp - zr * sp;
  const z2 = yr * sp + zr * cp;
  const perspective = 3.7 / (3.7 + z2);
  const scale = Math.min(rect.width, rect.height) * 0.31 * state.zoom * perspective;
  return {
    x: rect.width / 2 + xr * scale + state.panX,
    y: rect.height * 0.58 - y2 * scale + state.panY,
    depth: z2,
    scale: perspective,
  };
}

function project3d(point, range) {
  const w = normalizedWorld(point);
  const h = heightValue(point, range);
  return {
    base: project3dWorld(w.x, 0, w.z),
    top: project3dWorld(w.x, h, w.z),
    height: h,
  };
}

function drawBackground2d() {
  const rect = canvas.getBoundingClientRect();
  const grd = ctx.createLinearGradient(0, 0, rect.width, rect.height);
  grd.addColorStop(0, "#f8fbff");
  grd.addColorStop(1, "#edf5ff");
  ctx.fillStyle = grd;
  ctx.fillRect(0, 0, rect.width, rect.height);
  ctx.strokeStyle = "rgba(148,163,184,0.22)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i += 1) {
    const x = 38 + (rect.width - 76) * (i / 5);
    ctx.beginPath();
    ctx.moveTo(x, 22);
    ctx.lineTo(x, rect.height - 22);
    ctx.stroke();
    const y = 38 + (rect.height - 76) * (i / 5);
    ctx.beginPath();
    ctx.moveTo(22, y);
    ctx.lineTo(rect.width - 22, y);
    ctx.stroke();
  }
}

function drawBackground3d() {
  const rect = canvas.getBoundingClientRect();
  const grd = ctx.createLinearGradient(0, 0, 0, rect.height);
  grd.addColorStop(0, "#071326");
  grd.addColorStop(0.52, "#12203a");
  grd.addColorStop(1, "#eff6ff");
  ctx.fillStyle = grd;
  ctx.fillRect(0, 0, rect.width, rect.height);
  ctx.strokeStyle = "rgba(203,213,225,0.28)";
  ctx.lineWidth = 1;
  const grid = [];
  for (let i = -5; i <= 5; i += 1) {
    grid.push([[-2.4, 0, i * 0.44], [2.4, 0, i * 0.44]]);
    grid.push([[i * 0.48, 0, -2.2], [i * 0.48, 0, 2.2]]);
  }
  for (const [a, b] of grid) {
    const pa = project3dWorld(a[0], a[1], a[2]);
    const pb = project3dWorld(b[0], b[1], b[2]);
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  }
}

function draw2d(points, range) {
  drawBackground2d();
  for (const point of points) {
    const p = project2d(point);
    if (!insideCanvas(p.x, p.y, 12)) continue;
    ctx.beginPath();
    ctx.fillStyle = pointColor(point, range);
    const radius = state.pointSize * (point.hasLabel ? 1.18 : 1);
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.fill();
  }
}

function draw3d(points, range) {
  drawBackground3d();
  const maxDraw = 24000;
  const step = Math.max(1, Math.ceil(points.length / maxDraw));
  const projected = [];
  for (let i = 0; i < points.length; i += step) {
    const point = points[i];
    const proj = project3d(point, range);
    projected.push({ point, ...proj });
  }
  projected.sort((a, b) => b.top.depth - a.top.depth);

  ctx.lineWidth = Math.max(1, state.pointSize * 0.35);
  for (const item of projected) {
    if (!insideCanvas(item.top.x, item.top.y, 40)) continue;
    const color = pointColor(item.point, range);
    ctx.strokeStyle = withAlpha(color, 0.28);
    if (item.height > 0.02) {
      ctx.beginPath();
      ctx.moveTo(item.base.x, item.base.y);
      ctx.lineTo(item.top.x, item.top.y);
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.fillStyle = color;
    const radius = Math.max(1.2, state.pointSize * 0.9 * item.top.scale);
    ctx.arc(item.top.x, item.top.y, radius, 0, Math.PI * 2);
    ctx.fill();
  }
}

function insideCanvas(x, y, pad) {
  const rect = canvas.getBoundingClientRect();
  return x > -pad && y > -pad && x < rect.width + pad && y < rect.height + pad;
}

function draw() {
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  const points = visiblePoints();
  const range = colorRange(points);
  if (state.view === "3d") draw3d(points, range);
  else draw2d(points, range);
  updateLegend(range);
  updateStats(points);
}

function updateLegend(range) {
  const def = layerDefs[state.layer];
  const legend = document.getElementById("legend");
  if (def.palette === "class") {
    legend.innerHTML = `<div class="legend-row"><span>低</span><span style="color:#d04b4b">高</span></div>`;
    return;
  }
  const gradient =
    def.palette === "diverging"
      ? "linear-gradient(90deg,#2f6fb0,#f7f7f7,#c83f3f)"
      : def.palette === "magma"
        ? "linear-gradient(90deg,#1d1147,#721f81,#b73779,#f1605d,#fcfdbf)"
        : "linear-gradient(90deg,#482878,#31688e,#1f9e89,#6ece58,#fde725)";
  legend.innerHTML = `
    <div class="legend-gradient" style="background:${gradient}"></div>
    <div class="legend-row"><span>${fmt(range.min, 2)}</span><span>${fmt(range.max, 2)}</span></div>
  `;
}

function updateStats(points) {
  const metrics = DATA.metrics;
  document.getElementById("metricR2").textContent = fmt(metrics.test_r2, 3);
  document.getElementById("metricRmse").textContent = fmt(metrics.test_rmse, 3);
  const y5 = DATA.posthoc.find((row) => row.task === "binary_y5");
  document.getElementById("metricY5").textContent = y5 ? fmt(y5.testAuc, 3) : "-";
  document.getElementById("visibleCount").textContent = points.length.toLocaleString();
  const def = layerDefs[state.layer];
  document.getElementById("mapTitle").textContent = state.view === "3d" ? `${def.title} · 3D 沉浸` : `${def.title} · 2D 俯视`;
  const splitText = [...state.splits].join(" / ") || "none";
  document.getElementById("mapSubtitle").textContent =
    `${splitText} · ${state.tile === "all" ? "全部 tile" : state.tile} · ${state.view === "3d" ? "高度表示当前图层强度" : "平面空间分布"}`;
}

function projectedPoint(point, range) {
  if (state.view === "3d") return project3d(point, range).top;
  return project2d(point);
}

function nearestPoint(mouseX, mouseY) {
  const points = visiblePoints();
  const range = colorRange(points);
  const maxCheck = 26000;
  const step = Math.max(1, Math.ceil(points.length / maxCheck));
  let best = null;
  let bestD = Infinity;
  const limit = Math.max(10, state.pointSize * 3.2);
  for (let i = 0; i < points.length; i += step) {
    const point = points[i];
    const p = projectedPoint(point, range);
    const d = Math.hypot(p.x - mouseX, p.y - mouseY);
    if (d < bestD && d <= limit) {
      bestD = d;
      best = point;
    }
  }
  return best;
}

function showPoint(point, x, y) {
  if (!point) {
    tooltip.style.display = "none";
    document.getElementById("selectionDetails").textContent = "移动鼠标查看节点";
    return;
  }
  const html = `
    <strong>${point.tile}</strong> · seg ${point.seg}<br>
    split: ${point.split}<br>
    pred: ${fmt(point.pred, 3)} · true: ${fmt(point.true, 3)}<br>
    residual: ${fmt(point.residual, 3)}<br>
    y5 score: ${fmt(point.y5Score, 3)} · pano: ${point.pano}
  `;
  tooltip.innerHTML = html;
  tooltip.style.left = `${x + 14}px`;
  tooltip.style.top = `${y + 14}px`;
  tooltip.style.display = "block";
  document.getElementById("selectionDetails").innerHTML = html;
}

function populateTileSelect() {
  const select = document.getElementById("tileSelect");
  const tiles = [...new Set(DATA.points.map((point) => point.tile))].sort();
  select.innerHTML = `<option value="all">全部 tile</option>` + tiles.map((tile) => `<option value="${tile}">${tile}</option>`).join("");
}

function renderBars(containerId, rows, valueKey, nameKey, maxRows) {
  const container = document.getElementById(containerId);
  const data = rows.slice(0, maxRows);
  const max = Math.max(1e-9, ...data.map((row) => Math.max(0, Number(row[valueKey]) || 0)));
  container.innerHTML = data
    .map((row) => {
      const value = Number(row[valueKey]) || 0;
      const width = Math.max(0, (value / max) * 100);
      return `
        <div class="bar-row" title="${row[nameKey]}">
          <div class="bar-label">${row[nameKey]}</div>
          <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
          <div class="bar-value">${fmt(value, 4)}</div>
        </div>
      `;
    })
    .join("");
}

function renderTileList() {
  const rows = DATA.tiles
    .filter((row) => state.splits.has(row.split))
    .sort((a, b) => (b.rmse || 0) - (a.rmse || 0))
    .slice(0, 12);
  document.getElementById("tileList").innerHTML = rows
    .map(
      (row) => `
      <div class="tile-row">
        <strong>${row.tile}</strong>
        <span>${row.split} · n=${row.n}</span>
        <span>${fmt(row.rmse, 2)}</span>
      </div>
    `,
    )
    .join("");
}

function resetView() {
  state.zoom = 1;
  state.panX = 0;
  state.panY = 0;
  state.yaw = -0.72;
  state.pitch = 0.82;
  draw();
}

function applyShot(shot) {
  if (shot === "overview") {
    state.view = "2d";
    state.layer = "pred";
    state.tile = "all";
    state.labeledOnly = false;
    state.hotspotOnly = false;
  } else if (shot === "immersive") {
    state.view = "3d";
    state.layer = "pred";
    state.tile = "all";
    state.labeledOnly = false;
    state.hotspotOnly = false;
    state.heightScale = 1.7;
    state.autoOrbit = true;
  } else if (shot === "hotspot") {
    state.view = "3d";
    state.layer = "y5Score";
    state.tile = "all";
    state.labeledOnly = false;
    state.hotspotOnly = false;
    state.heightScale = 2.1;
  } else if (shot === "residual") {
    state.view = "2d";
    state.layer = "residual";
    state.tile = "all";
    state.labeledOnly = true;
    state.hotspotOnly = false;
  } else if (shot === "importance") {
    state.view = "2d";
    state.layer = "pred";
    state.tile = "all";
    state.labeledOnly = false;
    state.hotspotOnly = false;
    document.getElementById("groupBars").scrollIntoView({ behavior: "smooth", block: "center" });
  }
  syncControls();
  resetView();
}

function applyInitialHash() {
  const hash = window.location.hash.replace("#", "").toLowerCase();
  if (hash === "3d" || hash === "immersive") {
    state.view = "3d";
    state.layer = "pred";
    state.tile = "all";
    state.labeledOnly = false;
    state.hotspotOnly = false;
    state.heightScale = 1.7;
    state.autoOrbit = true;
  } else if (hash === "hotspot") {
    state.view = "3d";
    state.layer = "y5Score";
    state.tile = "all";
    state.labeledOnly = false;
    state.hotspotOnly = false;
    state.heightScale = 2.1;
    state.autoOrbit = true;
  } else if (hash === "residual") {
    state.view = "2d";
    state.layer = "residual";
    state.tile = "all";
    state.labeledOnly = true;
    state.hotspotOnly = false;
  }
}

function syncControls() {
  document.getElementById("layerSelect").value = state.layer;
  document.getElementById("tileSelect").value = state.tile;
  document.getElementById("labeledOnly").checked = state.labeledOnly;
  document.getElementById("hotspotOnly").checked = state.hotspotOnly;
  document.getElementById("autoOrbit").checked = state.autoOrbit;
  document.getElementById("pointSize").value = state.pointSize;
  document.getElementById("pointSizeValue").textContent = state.pointSize.toFixed(1);
  document.getElementById("heightScale").value = state.heightScale;
  document.getElementById("heightScaleValue").textContent = state.heightScale.toFixed(1);
  document.getElementById("colorMax").value = state.colorMax;
  document.querySelectorAll("#splitButtons button").forEach((button) => {
    button.classList.toggle("active", state.splits.has(button.dataset.split));
  });
  document.querySelectorAll("#viewButtons button").forEach((button) => {
    button.classList.toggle("active", state.view === button.dataset.view);
  });
}

function bindControls() {
  document.getElementById("layerSelect").addEventListener("change", (event) => {
    state.layer = event.target.value;
    draw();
  });
  document.getElementById("tileSelect").addEventListener("change", (event) => {
    state.tile = event.target.value;
    draw();
  });
  document.getElementById("labeledOnly").addEventListener("change", (event) => {
    state.labeledOnly = event.target.checked;
    draw();
  });
  document.getElementById("hotspotOnly").addEventListener("change", (event) => {
    state.hotspotOnly = event.target.checked;
    draw();
  });
  document.getElementById("autoOrbit").addEventListener("change", (event) => {
    state.autoOrbit = event.target.checked;
  });
  document.getElementById("pointSize").addEventListener("input", (event) => {
    state.pointSize = Number(event.target.value);
    document.getElementById("pointSizeValue").textContent = state.pointSize.toFixed(1);
    draw();
  });
  document.getElementById("heightScale").addEventListener("input", (event) => {
    state.heightScale = Number(event.target.value);
    document.getElementById("heightScaleValue").textContent = state.heightScale.toFixed(1);
    draw();
  });
  document.getElementById("colorMax").addEventListener("change", (event) => {
    state.colorMax = event.target.value;
    draw();
  });
  document.getElementById("resetView").addEventListener("click", resetView);
  document.getElementById("exportPng").addEventListener("click", () => {
    const a = document.createElement("a");
    a.href = canvas.toDataURL("image/png");
    a.download = `graphsage_${state.view}_${state.layer}_${state.tile}.png`;
    a.click();
  });
  document.querySelectorAll("#splitButtons button").forEach((button) => {
    button.addEventListener("click", () => {
      const split = button.dataset.split;
      if (state.splits.has(split)) state.splits.delete(split);
      else state.splits.add(split);
      syncControls();
      renderTileList();
      draw();
    });
  });
  document.querySelectorAll("#viewButtons button").forEach((button) => {
    button.addEventListener("click", () => {
      state.view = button.dataset.view;
      syncControls();
      resetView();
    });
  });
  document.querySelectorAll("[data-shot]").forEach((button) => {
    button.addEventListener("click", () => applyShot(button.dataset.shot));
  });
}

function bindCanvas() {
  canvas.addEventListener("mousedown", (event) => {
    state.dragging = true;
    state.lastX = event.offsetX;
    state.lastY = event.offsetY;
    canvas.classList.add("dragging");
  });
  window.addEventListener("mouseup", () => {
    state.dragging = false;
    canvas.classList.remove("dragging");
  });
  canvas.addEventListener("mousemove", (event) => {
    if (state.dragging) {
      const dx = event.offsetX - state.lastX;
      const dy = event.offsetY - state.lastY;
      if (state.view === "3d") {
        state.yaw += dx * 0.008;
        state.pitch = Math.max(0.28, Math.min(1.18, state.pitch + dy * 0.004));
      } else {
        state.panX += dx;
        state.panY += dy;
      }
      state.lastX = event.offsetX;
      state.lastY = event.offsetY;
      tooltip.style.display = "none";
      draw();
      return;
    }
    const point = nearestPoint(event.offsetX, event.offsetY);
    showPoint(point, event.offsetX, event.offsetY);
  });
  canvas.addEventListener("mouseleave", () => showPoint(null));
  canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const oldZoom = state.zoom;
      const factor = event.deltaY < 0 ? 1.12 : 0.9;
      const newZoom = Math.max(0.55, Math.min(13, oldZoom * factor));
      if (state.view === "2d") {
        const rect = canvas.getBoundingClientRect();
        const cx = rect.width / 2;
        const cy = rect.height / 2;
        state.panX = event.offsetX - cx - ((event.offsetX - cx - state.panX) / oldZoom) * newZoom;
        state.panY = event.offsetY - cy - ((event.offsetY - cy - state.panY) / oldZoom) * newZoom;
      }
      state.zoom = newZoom;
      draw();
    },
    { passive: false },
  );
}

function animationLoop(time) {
  const dt = state.lastFrame ? time - state.lastFrame : 16;
  state.lastFrame = time;
  if (state.view === "3d" && state.autoOrbit && !state.dragging) {
    state.yaw += dt * 0.00016;
    draw();
  }
  requestAnimationFrame(animationLoop);
}

function init() {
  applyInitialHash();
  populateTileSelect();
  renderBars("groupBars", DATA.importance.groups, "rmseDelta", "name", 10);
  renderBars("featureBars", DATA.importance.features, "rmseDelta", "name", 12);
  renderTileList();
  bindControls();
  bindCanvas();
  syncControls();
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);
  requestAnimationFrame(animationLoop);
}

init();
