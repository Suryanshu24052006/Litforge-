const byId = (id) => document.getElementById(id),
  colors = { transformer: "#275fe6", gru: "#087f78" };
let studyData = null,
  healthData = null;
function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}
async function api(path, body) {
  const response = await fetch(
    path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : "Please check the prompt and settings.",
    );
  return data;
}
document.querySelectorAll("nav button").forEach(
  (button) =>
    (button.onclick = () => {
      document
        .querySelectorAll("nav button")
        .forEach((item) =>
          item.setAttribute("aria-selected", String(item === button)),
        );
      ["results", "write", "inspect"].forEach(
        (id) => (byId(id).hidden = id !== button.dataset.tab),
      );
    }),
);
function tableRow(table, values) {
  const row = el("tr");
  values.forEach((value) => row.append(el("td", String(value))));
  byId(table).querySelector("tbody").append(row);
}
function drawCurve(data) {
  const canvas = byId("curve"),
    ctx = canvas.getContext("2d"),
    w = canvas.width,
    h = canvas.height,
    pad = 48;
  ctx.clearRect(0, 0, w, h);
  const series = [];
  for (const name of Object.keys(data.protocol.models)) {
    const grouped = new Map();
    for (const [run, history] of Object.entries(data.histories)) {
      if (!run.startsWith(name + "-")) continue;
      for (const row of history) {
        const values = grouped.get(row.step) || [];
        values.push(row.validation.bits_per_byte);
        grouped.set(row.step, values);
      }
    }
    series.push({
      name,
      points: [...grouped.entries()]
        .sort((a, b) => a[0] - b[0])
        .map(([step, values]) => [
          step,
          values.reduce((a, b) => a + b, 0) / values.length,
        ]),
    });
  }
  const values = series.flatMap((s) => s.points.map((p) => p[1]));
  if (!values.length) return;
  const low = Math.floor(Math.min(...values)),
    high = Math.ceil(Math.max(...values)),
    span = Math.max(0.1, high - low),
    maxStep = data.protocol.steps;
  ctx.font = "12px system-ui";
  for (let line = 0; line <= 4; line++) {
    const y = pad + ((h - 2 * pad) * line) / 4;
    ctx.strokeStyle = "#e4e9f0";
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(w - pad, y);
    ctx.stroke();
    ctx.fillStyle = "#607089";
    ctx.fillText((high - (span * line) / 4).toFixed(1), 8, y + 4);
  }
  for (const item of series) {
    ctx.strokeStyle = colors[item.name];
    ctx.lineWidth = 3;
    ctx.beginPath();
    item.points.forEach(([step, value], index) => {
      const x = pad + (step / maxStep) * (w - 2 * pad),
        y = pad + ((high - value) / span) * (h - 2 * pad);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
  ctx.fillStyle = "#607089";
  ctx.fillText("0", pad, h - 14);
  ctx.fillText("Training steps", w / 2 - 35, h - 14);
  ctx.fillText(String(maxStep), w - pad - 32, h - 14);
}
function renderStudy(data) {
  studyData = data;
  const cards = byId("score-cards");
  cards.replaceChildren();
  for (const row of data.aggregate) {
    const card = el("div", undefined, "card"),
      heading = el("div", undefined, "row");
    heading.append(
      el(
        "h2",
        row.architecture === "gru"
          ? "GRU · recurrent memory"
          : "Transformer · attention",
      ),
      el("span", row.seeds.length + " seeds", "pill"),
    );
    card.append(
      heading,
      el("div", row.mean_test_bpb.toFixed(3), "big"),
      el(
        "div",
        "test bits / byte · SD " +
          (row.sample_std_test_bpb === null
            ? "not estimated"
            : row.sample_std_test_bpb.toFixed(4)),
        "label",
      ),
    );
    const run = data.runs.find((r) => r.architecture === row.architecture);
    card.append(
      el(
        "p",
        run.parameters.toLocaleString() +
          " parameters · " +
          data.protocol.steps.toLocaleString() +
          " steps per seed",
        "subtle",
      ),
    );
    cards.append(card);
  }
  drawCurve(data);
  const contextHeader = byId("context-table").querySelector("thead tr");
  contextHeader.replaceChildren(el("th", "Model"));
  data.protocol.context_probe.lengths.forEach((length) =>
    contextHeader.append(el("th", length + " bytes")),
  );
  byId("context-table").querySelector("tbody").replaceChildren();
  for (const name of Object.keys(data.protocol.models)) {
    const values = data.protocol.context_probe.lengths.map((length) => {
      const entries = data.runs
        .filter((r) => r.architecture === name)
        .flatMap((r) =>
          r.context_probe
            .filter((p) => p.context_bytes === length)
            .map((p) => p.bits_per_byte),
        );
      return (entries.reduce((a, b) => a + b, 0) / entries.length).toFixed(3);
    });
    tableRow("context-table", [name, ...values]);
  }
  byId("baselines").querySelector("tbody").replaceChildren();
  data.ngram_baselines.forEach((row) =>
    tableRow("baselines", [row.order + "-gram", row.bits_per_byte.toFixed(3)]),
  );
  const split = byId("splits");
  split.replaceChildren();
  for (const key of ["train", "validation", "test"]) {
    const box = el("div");
    box.append(el("div", key.toUpperCase(), "eyebrow"));
    const list = el("ul");
    data.dataset.books
      .filter((b) => b.split === key)
      .forEach((book) => list.append(el("li", book.title)));
    box.append(list);
    split.append(box);
  }
  const context = Object.values(data.protocol.models)[0].context;
  byId("protocol-note").textContent =
    (
      data.protocol.steps *
      data.protocol.batch_size *
      context
    ).toLocaleString() +
    " training bytes per neural run. Checkpoints are chosen on " +
    data.dataset.books
      .filter((b) => b.split === "validation")
      .map((b) => b.title)
      .join(", ") +
    "; " +
    data.dataset.books
      .filter((b) => b.split === "test")
      .map((b) => b.title)
      .join(", ") +
    " is scored after all " +
    data.protocol.seeds.length * Object.keys(data.protocol.models).length +
    " planned runs finish.";
}
api("/health")
  .then((data) => {
    healthData = data;
    byId("health").textContent =
      data.models.length +
      " model" +
      (data.models.length === 1 ? "" : "s") +
      " ready · CPU";
    document.querySelectorAll(".model-select").forEach((select) => {
      select.replaceChildren();
      data.models.forEach((model) => {
        const option = el(
          "option",
          model.architecture === "gru" ? "GRU" : "Transformer",
        );
        option.value = model.id;
        select.append(option);
      });
    });
    updateInspector();
  })
  .catch((error) => {
    byId("health").textContent = error.message;
    byId("health").classList.add("error");
  });
api("/study")
  .then(renderStudy)
  .catch((error) => {
    byId("score-cards").replaceChildren(el("div", error.message, "card empty"));
  });
byId("temperature").oninput = (event) =>
  (byId("temperature-value").textContent = event.target.value);
byId("write-form").onsubmit = async (event) => {
  event.preventDefault();
  const button = byId("write-button");
  button.disabled = true;
  byId("write-status").textContent = "Generating…";
  try {
    const data = await api("/generate", {
      model: byId("writer-model").value,
      prompt: byId("prompt").value,
      max_new_tokens: Number(byId("length").value),
      temperature: Number(byId("temperature").value),
      seed: Number(byId("seed").value),
      top_k: Number(byId("topk").value),
    });
    byId("output").textContent = data.text;
    byId("write-status").textContent =
      data.generated_tokens +
      " " +
      data.token_unit +
      " tokens · " +
      data.generated_characters +
      " visible characters · " +
      data.seconds.toFixed(2) +
      " seconds";
  } catch (error) {
    byId("write-status").textContent = error.message;
  } finally {
    button.disabled = false;
  }
};
function updateInspector() {
  if (!healthData) return;
  const model = healthData.models.find(
    (m) => m.id === byId("inspector-model").value,
  );
  if (!model) return;
  byId("layer").max = model.layers - 1;
  byId("head").max = Math.max(0, model.heads - 1);
  byId("layer").value = "0";
  byId("head").value = "0";
  byId("head").disabled = model.heads === 0;
  byId("layer").disabled = model.heads === 0;
}
byId("inspector-model").onchange = updateInspector;
function drawAttention(data) {
  const canvas = byId("attention"),
    ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, 720, 720);
  if (!data.attention) {
    ctx.fillStyle = "#607089";
    ctx.font = "17px system-ui";
    ctx.fillText("Recurrent memory has no attention matrix.", 35, 70);
    return;
  }
  const start = Math.max(0, data.token_count - 48),
    n = data.token_count - start,
    pad = 55,
    size = 620 / n;
  ctx.font = Math.min(12, size * 0.8) + "px monospace";
  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      const weight = data.attention[start + row][start + col];
      ctx.fillStyle =
        weight === 0
          ? "#f4f6fa"
          : `rgba(39,95,230,${Math.max(0.07, Math.sqrt(weight))})`;
      ctx.fillRect(pad + col * size, pad + row * size, size - 0.6, size - 0.6);
    }
    ctx.fillStyle = "#607089";
    ctx.fillText(data.tokens[start + row], 10, pad + (row + 0.7) * size);
    ctx.save();
    ctx.translate(pad + (row + 0.65) * size, 45);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(data.tokens[start + row], 0, 0);
    ctx.restore();
  }
  ctx.fillStyle = "#607089";
  ctx.font = "12px system-ui";
  ctx.fillText(
    "Rows: queries · columns: keys · last " + n + " positions",
    pad,
    705,
  );
}
byId("inspect-form").onsubmit = async (event) => {
  event.preventDefault();
  const button = byId("inspect-button");
  button.disabled = true;
  byId("inspect-status").textContent = "Inspecting…";
  try {
    const data = await api("/inspect", {
      model: byId("inspector-model").value,
      prompt: byId("inspect-prompt").value,
      layer: Number(byId("layer").value),
      head: Number(byId("head").value),
    });
    const panel = byId("probabilities");
    panel.replaceChildren();
    for (const token of data.next_tokens) {
      const row = el("div", undefined, "token-row"),
        track = el("div", undefined, "track"),
        fill = el("div", undefined, "fill");
      fill.style.width = token.probability * 100 + "%";
      track.append(fill);
      row.append(
        el("span", token.label, "token-name"),
        track,
        el("span", (token.probability * 100).toFixed(1) + "%", "prob"),
      );
      panel.append(row);
    }
    drawAttention(data);
    byId("attention-note").textContent =
      data.note +
      " Showing at most the last 48 positions; earlier attention can be outside the displayed area.";
    byId("inspect-status").textContent =
      data.token_count +
      " tokens inspected" +
      (data.clipped ? " · prompt clipped to the context window" : "");
  } catch (error) {
    byId("inspect-status").textContent = error.message;
  } finally {
    button.disabled = false;
  }
};
