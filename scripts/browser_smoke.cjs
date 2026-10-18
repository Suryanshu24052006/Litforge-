const { chromium } = require("playwright");
const assert = require("node:assert/strict");
const { resolve } = require("node:path");
const url = process.argv[2] || "http://127.0.0.1:8000";
const screenshotRoot = process.env.LITFORGE_SCREENSHOTS;
(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_PATH || undefined,
    headless: true,
    args: ["--no-sandbox", "--disable-dev-shm-usage"],
  });
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1080 },
    deviceScaleFactor: 1,
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(url, { waitUntil: "networkidle" });
  const health = await (
    await page.request.get(new URL("/health", url).href)
  ).json();
  await page.getByText(health.models.length + " models ready · CPU").waitFor();
  const modelIds = health.models.map((model) => model.id);
  assert.ok(modelIds.includes("primary") && modelIds.includes("gru"));
  if (modelIds.includes("extended")) {
    assert.equal(health.default_model, "extended");
    assert.equal(await page.locator("#writer-model").inputValue(), "extended");
    assert.equal(await page.locator("#extended-note").isVisible(), true);
  }
  assert.equal(await page.locator("#score-cards .card").count(), 2);
  assert.equal(await page.locator("#baselines tbody tr").count(), 4);
  assert.equal(await page.locator("#splits li").count(), 5);
  if (screenshotRoot)
    await page.screenshot({
      path: resolve(screenshotRoot, "dashboard.png"),
      fullPage: true,
    });
  await page.getByRole("tab", { name: "Try the models" }).click();
  await page.locator("#length").selectOption("100");
  for (const name of modelIds) {
    await page.locator("#writer-model").selectOption(name);
    await page.locator("#prompt").fill("A café — <script>alert(1)</script> ");
    await page.getByRole("button", { name: "Continue passage" }).click();
    await page.waitForFunction(() =>
      document
        .querySelector("#write-status")
        .textContent.includes("100 byte tokens"),
    );
    assert.ok(
      (await page.locator("#output").textContent()).startsWith(
        "A café — <script>alert(1)</script> ",
      ),
    );
    assert.equal(await page.locator("#output script").count(), 0);
  }
  await page.getByRole("tab", { name: "Inside the model" }).click();
  await page.getByRole("button", { name: "Inspect this prompt" }).click();
  await page.waitForFunction(
    () => document.querySelector("#probabilities").children.length === 8,
  );
  assert.ok(
    (await page.locator("#inspect-status").textContent()).includes(
      "tokens inspected",
    ),
  );
  if (screenshotRoot)
    await page.screenshot({
      path: resolve(screenshotRoot, "inspector.png"),
      fullPage: true,
    });
  await page.locator("#inspector-model").selectOption("gru");
  await page.getByRole("button", { name: "Inspect this prompt" }).click();
  await page.waitForFunction(() =>
    document
      .querySelector("#attention-note")
      .textContent.includes("no attention matrix"),
  );
  assert.equal(await page.locator("#head").isDisabled(), true);
  await page.setViewportSize({ width: 390, height: 844 });
  for (const name of ["Experiment", "Try the models", "Inside the model"]) {
    await page.getByRole("tab", { name, exact: true }).click();
    assert.equal(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
      true,
      "Mobile page overflows: " + name,
    );
  }
  assert.deepEqual(errors, []);
  console.log(
    "PASS: all tabs and loaded byte models, default selection, Unicode/HTML text, attention and GRU views, mobile overflow, no browser errors.",
  );
  await browser.close();
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
