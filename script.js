const byId = (id) => document.getElementById(id);
const LOCAL_API_ORIGIN = "http://127.0.0.1:8000";
const servedFromApiOrigin =
  (window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost") &&
  window.location.port === "8000";
const apiBase = servedFromApiOrigin ? "" : LOCAL_API_ORIGIN;

let currentUser = null;
let requireAuth = false;
let deepfakeBusy = false;
let newsBusy = false;

const year = byId("year");
if (year) {
  year.textContent = new Date().getFullYear();
}

function appPath(path8000, path5500) {
  return servedFromApiOrigin ? path8000 : path5500;
}

function apiUrl(path) {
  return `${apiBase}${path}`;
}

async function fetchApi(path, options = {}) {
  return fetch(apiUrl(path), {
    credentials: "include",
    ...options,
  });
}

async function parseApiResponse(response) {
  const text = await response.text();
  let payload = {};
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch (_err) {
      payload = { detail: text.slice(0, 250) };
    }
  }
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed (${response.status})`);
  }
  return payload;
}

function normalizeError(error) {
  const message = String(error?.message || "Request failed.");
  if (message.toLowerCase().includes("failed to fetch")) {
    return `Cannot reach API at ${apiBase || window.location.origin}. Start backend on :8000.`;
  }
  return message;
}

function setupRevealAnimations() {
  const nodes = document.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window)) {
    nodes.forEach((node) => node.classList.add("visible"));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.2 }
  );

  nodes.forEach((node) => observer.observe(node));
}

function resolveVerdict(score) {
  if (score >= 75) {
    return { label: "High Risk", className: "is-danger" };
  }
  if (score >= 45) {
    return { label: "Review Needed", className: "is-warning" };
  }
  return { label: "Likely Authentic", className: "is-safe" };
}

function setVerdictText(element, text, className) {
  element.classList.remove("is-danger", "is-warning", "is-safe");
  element.classList.add(className);
  element.textContent = text;
}

function setBackendStatus(text, isError = false) {
  const status = byId("backendStatus");
  if (!status) return;
  status.textContent = text;
  status.style.color = isError ? "var(--danger)" : "var(--text-muted)";
}

function updateAuthUi() {
  const currentModeMetric = byId("currentModeMetric");
  const authActionBtn = byId("authActionBtn");
  const signupNavLink = byId("signupNavLink");
  const loginNavLink = byId("loginNavLink");
  const brandLink = byId("brandLink");
  const dashboardLink = byId("dashboardLink");

  if (brandLink) brandLink.href = appPath("/", "/syntHeye.html");
  if (dashboardLink) dashboardLink.href = appPath("/", "/syntHeye.html");
  if (signupNavLink) signupNavLink.href = appPath("/signup", "/signup.html");
  if (loginNavLink) loginNavLink.href = appPath("/login", "/login.html");

  if (currentModeMetric) {
    currentModeMetric.textContent = currentUser ? "Authenticated" : "Guest Mode";
  }

  if (!authActionBtn) return;

  if (currentUser) {
    authActionBtn.textContent = `Logout (${currentUser.email})`;
    authActionBtn.onclick = async () => {
      try {
        await parseApiResponse(await fetchApi("/api/auth/logout", { method: "POST" }));
        currentUser = null;
        updateAuthUi();
        setBackendStatus("Logged out successfully.");
      } catch (error) {
        setBackendStatus(normalizeError(error), true);
      }
    };
    return;
  }

  authActionBtn.textContent = "Login / Signup";
  authActionBtn.onclick = () => {
    window.location.href = appPath("/login", "/login.html");
  };
}

async function loadSession() {
  try {
    const me = await parseApiResponse(await fetchApi("/api/me"));
    currentUser = me.user || null;
    requireAuth = Boolean(me.require_auth);
    updateAuthUi();

    if (requireAuth && !currentUser) {
      setBackendStatus("Auth required. Sign in to run live analysis.", true);
    } else {
      setBackendStatus("Connected. Live analysis endpoints are ready.");
    }
  } catch (error) {
    setBackendStatus(normalizeError(error), true);
  }
}

function requireSignInForAction() {
  if (requireAuth && !currentUser) {
    setBackendStatus("Please sign in to run analysis.", true);
    window.location.href = appPath("/login", "/login.html");
    return true;
  }
  return false;
}

function toPercentage(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(100, n * 100));
}

function setupDeepfakeModule() {
  const deepfakeForm = byId("deepfakeForm");
  const assetFile = byId("assetFile");
  const deepfakeStatus = byId("deepfakeStatus");
  const manipulationProbability = byId("manipulationProbability");
  const anomalyScore = byId("anomalyScore");
  const deepfakeVerdict = byId("deepfakeVerdict");
  const deepfakeSubmitBtn = byId("deepfakeSubmitBtn");

  deepfakeForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (deepfakeBusy) return;
    if (requireSignInForAction()) return;

    if (!assetFile.files || !assetFile.files[0]) {
      deepfakeStatus.textContent = "Upload an image or video to begin forensic analysis.";
      return;
    }

    const formData = new FormData();
    formData.append("file", assetFile.files[0]);
    const notes = String(byId("assetNotes")?.value || "").trim();
    if (notes) {
      formData.append("notes", notes);
    }

    deepfakeBusy = true;
    if (deepfakeSubmitBtn) deepfakeSubmitBtn.disabled = true;
    deepfakeStatus.textContent = "Running live forensic scan...";

    try {
      const payload = await parseApiResponse(
        await fetchApi("/api/analyze/file", {
          method: "POST",
          body: formData,
        })
      );

      const fakeProbability =
        typeof payload.real_score === "number"
          ? toPercentage(1 - payload.real_score)
          : payload.prediction === "fake"
            ? toPercentage(payload.confidence)
            : toPercentage(1 - Number(payload.confidence || 0));
      const anomaly = payload.frames_sampled
        ? `${Math.min(100, 30 + payload.frames_sampled)}/100`
        : `${Math.round(Math.min(100, Math.abs(50 - fakeProbability) + 50))}/100`;
      const verdict = resolveVerdict(fakeProbability);

      deepfakeStatus.textContent = `${payload.task || "analysis"} complete.`;
      manipulationProbability.textContent = `${fakeProbability.toFixed(1)}%`;
      anomalyScore.textContent = anomaly;
      setVerdictText(deepfakeVerdict, `${verdict.label} (${String(payload.prediction || "").toUpperCase()})`, verdict.className);
    } catch (error) {
      const message = normalizeError(error);
      deepfakeStatus.textContent = message;
      if (message.toLowerCase().includes("authentication required")) {
        window.location.href = appPath("/login", "/login.html");
      }
    } finally {
      deepfakeBusy = false;
      if (deepfakeSubmitBtn) deepfakeSubmitBtn.disabled = false;
    }
  });
}

function setupNewsModule() {
  const newsForm = byId("newsForm");
  const newsText = byId("newsText");
  const sourceUrl = byId("sourceUrl");
  const newsStatus = byId("newsStatus");
  const misinfoRisk = byId("misinfoRisk");
  const volatilityScore = byId("volatilityScore");
  const newsVerdict = byId("newsVerdict");
  const newsSubmitBtn = byId("newsSubmitBtn");

  newsForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (newsBusy) return;
    if (requireSignInForAction()) return;

    const text = String(newsText.value || "").trim();
    const url = String(sourceUrl.value || "").trim();
    if (!text && !url) {
      newsStatus.textContent = "Paste article text or provide a source URL to analyze.";
      return;
    }

    const payloadText = text || `Source URL: ${url}`;
    newsBusy = true;
    if (newsSubmitBtn) newsSubmitBtn.disabled = true;
    newsStatus.textContent = "Running live intelligence scan...";

    try {
      const payload = await parseApiResponse(
        await fetchApi("/api/analyze/text", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: payloadText }),
        })
      );

      const risk = toPercentage(payload.fake_probability);
      const volatility = Math.round(toPercentage(payload.confidence));
      const verdict = resolveVerdict(risk);

      newsStatus.textContent = `${payload.task || "analysis"} complete.`;
      misinfoRisk.textContent = `${risk.toFixed(1)}%`;
      volatilityScore.textContent = `${volatility}/100`;
      setVerdictText(newsVerdict, `${verdict.label} (${String(payload.prediction || "").toUpperCase()})`, verdict.className);
    } catch (error) {
      const message = normalizeError(error);
      newsStatus.textContent = message;
      if (message.toLowerCase().includes("authentication required")) {
        window.location.href = appPath("/login", "/login.html");
      }
    } finally {
      newsBusy = false;
      if (newsSubmitBtn) newsSubmitBtn.disabled = false;
    }
  });
}

function setupNavigationActions() {
  byId("startScanning")?.addEventListener("click", () => {
    byId("modules")?.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  byId("watchScene")?.addEventListener("click", () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
}

function setupThreeBackground() {
  if (!window.THREE) {
    return;
  }

  const canvas = byId("bg-canvas");
  if (!canvas) {
    return;
  }

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x03060f, 8, 24);

  const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
  camera.position.set(0, 0, 8);

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);

  const ambient = new THREE.AmbientLight(0x6699ff, 0.7);
  scene.add(ambient);

  const point = new THREE.PointLight(0x67ffd8, 1.6, 40);
  point.position.set(2.8, 2.4, 5.2);
  scene.add(point);

  const coreGeometry = new THREE.IcosahedronGeometry(1.7, 1);
  const coreMaterial = new THREE.MeshStandardMaterial({
    color: 0x78a2ff,
    wireframe: true,
    transparent: true,
    opacity: 0.4,
    emissive: 0x001733,
    roughness: 0.22,
    metalness: 0.8,
  });
  const coreMesh = new THREE.Mesh(coreGeometry, coreMaterial);
  scene.add(coreMesh);

  const particleCount = 1000;
  const pointsGeometry = new THREE.BufferGeometry();
  const positions = new Float32Array(particleCount * 3);
  for (let i = 0; i < particleCount * 3; i += 3) {
    positions[i] = (Math.random() - 0.5) * 40;
    positions[i + 1] = (Math.random() - 0.5) * 40;
    positions[i + 2] = (Math.random() - 0.5) * 40;
  }
  pointsGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));

  const pointsMaterial = new THREE.PointsMaterial({
    size: 0.03,
    color: 0x9fd9ff,
    transparent: true,
    opacity: 0.65,
  });
  const stars = new THREE.Points(pointsGeometry, pointsMaterial);
  scene.add(stars);

  const mouse = { x: 0, y: 0 };
  window.addEventListener("mousemove", (event) => {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = (event.clientY / window.innerHeight) * 2 - 1;
  });

  const clock = new THREE.Clock();
  function animate() {
    const elapsed = clock.getElapsedTime();

    coreMesh.rotation.x = elapsed * 0.24;
    coreMesh.rotation.y = elapsed * 0.32;
    coreMesh.position.y = Math.sin(elapsed * 0.8) * 0.2;

    stars.rotation.y = elapsed * 0.01;
    stars.rotation.x = elapsed * 0.005;

    camera.position.x += ((mouse.x * 1.8) - camera.position.x) * 0.02;
    camera.position.y += ((-mouse.y * 1.5) - camera.position.y) * 0.02;
    camera.lookAt(scene.position);

    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }
  animate();

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  });
}

setupRevealAnimations();
setupDeepfakeModule();
setupNewsModule();
setupNavigationActions();
setupThreeBackground();
loadSession();
