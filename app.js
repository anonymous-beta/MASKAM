
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const $ = (selector) => document.querySelector(selector);

const video = $("#camera");
const canvas = $("#overlay");
const ctx = canvas?.getContext("2d", { alpha: true });
const stage = $("#stage");
const startBtn = $("#startBtn");
const stopBtn = $("#stopBtn");
const flipBtn = $("#flipBtn");
const startBtnText = $("#startBtnText");
const statusMessage = $("#statusMessage");
const trackingBadge = $("#trackingBadge");
const cameraLabel = $("#cameraLabel");
const fpsLabel = $("#fpsLabel");
const opacitySlider = $("#opacitySlider");
const opacityValue = $("#opacityValue");
const mirrorToggle = $("#mirrorToggle");
const settingsDialog = $("#settingsDialog");
const stageEmpty = $("#stageEmpty");

const required = {
  video,
  canvas,
  ctx,
  stage,
  startBtn,
  stopBtn,
  flipBtn,
  startBtnText,
  statusMessage,
  trackingBadge,
  cameraLabel,
  fpsLabel,
  opacitySlider,
  opacityValue,
  mirrorToggle
};

for (const [name, element] of Object.entries(required)) {
  if (!element) {
    throw new Error(`MASKAM initialization failed: missing ${name}.`);
  }
}

/*
 * Keep this version aligned with package.json:
 * "@mediapipe/tasks-vision": "0.10.35"
 */
const MODEL_VERSION = "0.10.35";

const WASM_URL =
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MODEL_VERSION}/wasm`;

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const TARGET_INTERVAL = 1000 / 24;
const MAX_DPR = 2;
const SMOOTHING = 0.38;

const MASK_STYLES = {
  classic: {
    fill: "#e8e6df",
    stroke: "#202127",
    features: "#202127",
    accent: "#bcb9b1"
  },
  noir: {
    fill: "#22252c",
    stroke: "#aeb3bd",
    features: "#e6e8eb",
    accent: "#555b67"
  },
  crimson: {
    fill: "#e7e3dc",
    stroke: "#33181d",
    features: "#32171c",
    accent: "#bd343d"
  }
};

let stream = null;
let landmarker = null;
let modelPromise = null;
let operation = Promise.resolve();
let animationId = null;
let facingMode = "user";
let maskStyle = "classic";
let maskOpacity = Number(opacitySlider.value) / 100;
let lastStatus = "";
let lastVideoTime = -1;
let lastInferenceTime = 0;
let lastFpsTime = performance.now();
let frameCount = 0;
let disposed = false;
let generation = 0;
let smoothedLandmarks = null;
let resizeObserver = null;

function setStatus(message) {
  if (message === lastStatus) return;

  lastStatus = message;
  statusMessage.textContent = message;
}

function setTrackingState(active, found = false) {
  trackingBadge.classList.toggle("active", active && found);

  const label = !active
    ? "STANDBY"
    : found
      ? "FACE TRACKED"
      : "SEARCHING";

  const dot = document.createElement("span");
  dot.className = "state-dot";

  trackingBadge.replaceChildren(
    dot,
    document.createTextNode(` ${label}`)
  );
}

function setControls(starting = false) {
  const active = Boolean(stream);

  startBtn.disabled = starting || active || disposed;
  stopBtn.disabled = !active;
  flipBtn.disabled = !active || starting || disposed;

  startBtnText.textContent = starting
    ? "Starting…"
    : active
      ? "Camera enabled"
      : "Enable camera";
}

function resizeCanvas() {
  if (disposed) return;

  const rect = stage.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);

  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  clearCanvas();
}

function clearCanvas() {
  const rect = stage.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
}

function resetSmoothing() {
  smoothedLandmarks = null;
  lastVideoTime = -1;
  lastInferenceTime = 0;
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function smoothLandmarks(points) {
  if (
    !smoothedLandmarks ||
    smoothedLandmarks.length !== points.length
  ) {
    smoothedLandmarks = points.map((p) => ({
      x: p.x,
      y: p.y,
      z: p.z || 0
    }));

    return smoothedLandmarks;
  }

  smoothedLandmarks = points.map((p, i) => {
    const previous = smoothedLandmarks[i];

    return {
      x: previous.x + (p.x - previous.x) * SMOOTHING,
      y: previous.y + (p.y - previous.y) * SMOOTHING,
      z: previous.z + ((p.z || 0) - previous.z) * SMOOTHING
    };
  });

  return smoothedLandmarks;
}

function drawMask(landmarks) {
  const rect = stage.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  const vw = video.videoWidth;
  const vh = video.videoHeight;

  if (!vw || !vh || !width || !height) return;

  // Match the usual CSS object-fit: cover camera presentation.
  const scale = Math.max(width / vw, height / vh);
  const dw = vw * scale;
  const dh = vh * scale;
  const ox = (width - dw) / 2;
  const oy = (height - dh) / 2;
  const mirrored = mirrorToggle.checked;

  function point(index) {
    const p = landmarks[index];
    let x = ox + p.x * dw;

    if (mirrored) {
      x = width - x;
    }

    return {
      x,
      y: oy + p.y * dh
    };
  }

  const forehead = point(10);
  const chin = point(152);
  const leftFace = point(234);
  const rightFace = point(454);
  const leftEye = point(33);
  const rightEye = point(263);
  const nose = point(1);
  const mouthLeft = point(61);
  const mouthRight = point(291);
  const mouthBottom = point(14);

  const faceWidth = distance(leftFace, rightFace);

  if (!Number.isFinite(faceWidth) || faceWidth < 25) return;

  const style = MASK_STYLES[maskStyle];
  const eyeX = (leftEye.x + rightEye.x) / 2;
  const eyeY = (leftEye.y + rightEye.y) / 2;
  const eyeDistance = distance(leftEye, rightEye);
  const mouthX = (mouthLeft.x + mouthRight.x) / 2;
  const mouthY = (mouthLeft.y + mouthRight.y) / 2;

  const halfWidth = faceWidth * 0.53;
  const templeY = forehead.y + faceWidth * 0.28;
  const lowerY = chin.y - faceWidth * 0.02;
  const cheekY = eyeY + faceWidth * 0.19;

  ctx.save();
  ctx.globalAlpha = maskOpacity;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  // Main sculpted mask silhouette.
  ctx.beginPath();
  ctx.moveTo(eyeX - halfWidth * 0.91, templeY);

  ctx.quadraticCurveTo(
    eyeX - halfWidth * 0.74,
    eyeY - faceWidth * 0.05,
    eyeX - eyeDistance * 0.48,
    eyeY - faceWidth * 0.02
  );

  ctx.lineTo(
    nose.x - faceWidth * 0.055,
    nose.y + faceWidth * 0.08
  );

  ctx.quadraticCurveTo(
    nose.x,
    nose.y + faceWidth * 0.15,
    nose.x + faceWidth * 0.055,
    nose.y + faceWidth * 0.08
  );

  ctx.lineTo(
    eyeX + eyeDistance * 0.48,
    eyeY - faceWidth * 0.02
  );

  ctx.quadraticCurveTo(
    eyeX + halfWidth * 0.74,
    eyeY - faceWidth * 0.05,
    eyeX + halfWidth * 0.91,
    templeY
  );

  ctx.quadraticCurveTo(
    rightFace.x + faceWidth * 0.02,
    cheekY,
    eyeX + halfWidth * 0.66,
    lowerY - faceWidth * 0.16
  );

  ctx.quadraticCurveTo(
    eyeX,
    lowerY + faceWidth * 0.08,
    eyeX - halfWidth * 0.66,
    lowerY - faceWidth * 0.16
  );

  ctx.quadraticCurveTo(
    leftFace.x - faceWidth * 0.02,
    cheekY,
    eyeX - halfWidth * 0.91,
    templeY
  );

  ctx.closePath();
  ctx.fillStyle = style.fill;
  ctx.fill();

  ctx.strokeStyle = style.stroke;
  ctx.lineWidth = Math.max(1.2, faceWidth * 0.012);
  ctx.stroke();

  // Gentle shading for depth.
  const gradient = ctx.createLinearGradient(
    eyeX,
    eyeY,
    eyeX,
    lowerY
  );

  gradient.addColorStop(0, "#ffffff18");
  gradient.addColorStop(1, "#00000020");

  ctx.fillStyle = gradient;
  ctx.fill();

  // Eye openings follow the detected eye-line angle.
  const eyeWidth = Math.max(12, eyeDistance * 0.39);
  const eyeHeight = Math.max(7, faceWidth * 0.073);

  const eyeAngle = Math.atan2(
    rightEye.y - leftEye.y,
    rightEye.x - leftEye.x
  );

  function drawEye(center) {
    ctx.save();

    ctx.translate(
      center.x,
      center.y + faceWidth * 0.025
    );

    ctx.rotate(eyeAngle);

    ctx.beginPath();
    ctx.moveTo(-eyeWidth * 0.58, -eyeHeight * 0.12);

    ctx.quadraticCurveTo(
      -eyeWidth * 0.2,
      -eyeHeight * 0.75,
      0,
      -eyeHeight * 0.2
    );

    ctx.quadraticCurveTo(
      eyeWidth * 0.28,
      -eyeHeight * 0.55,
      eyeWidth * 0.58,
      0
    );

    ctx.quadraticCurveTo(
      eyeWidth * 0.22,
      eyeHeight * 0.7,
      0,
      eyeHeight * 0.18
    );

    ctx.quadraticCurveTo(
      -eyeWidth * 0.28,
      eyeHeight * 0.6,
      -eyeWidth * 0.58,
      -eyeHeight * 0.12
    );

    ctx.closePath();
    ctx.fillStyle = "#08090c";
    ctx.fill();

    ctx.strokeStyle = style.features;
    ctx.lineWidth = Math.max(1, faceWidth * 0.009);
    ctx.stroke();

    ctx.restore();
  }

  drawEye(leftEye);
  drawEye(rightEye);

  // Nose detail.
  ctx.beginPath();

  ctx.moveTo(
    nose.x - faceWidth * 0.075,
    nose.y + faceWidth * 0.13
  );

  ctx.quadraticCurveTo(
    nose.x,
    nose.y + faceWidth * 0.18,
    nose.x + faceWidth * 0.075,
    nose.y + faceWidth * 0.13
  );

  ctx.strokeStyle = style.accent;
  ctx.lineWidth = Math.max(1, faceWidth * 0.01);
  ctx.stroke();

  // Signature moustache.
  const moustacheY = mouthY - faceWidth * 0.015;
  const moustacheHalf = faceWidth * 0.20;

  ctx.fillStyle = style.features;

  for (const direction of [-1, 1]) {
    ctx.beginPath();
    ctx.moveTo(mouthX, moustacheY);

    ctx.bezierCurveTo(
      mouthX + direction * moustacheHalf * 0.55,
      moustacheY - faceWidth * 0.10,
      mouthX + direction * moustacheHalf * 1.05,
      moustacheY - faceWidth * 0.02,
      mouthX + direction * moustacheHalf * 1.10,
      moustacheY + faceWidth * 0.055
    );

    ctx.bezierCurveTo(
      mouthX + direction * moustacheHalf * 0.68,
      moustacheY + faceWidth * 0.035,
      mouthX + direction * moustacheHalf * 0.27,
      moustacheY + faceWidth * 0.11,
      mouthX,
      moustacheY + faceWidth * 0.075
    );

    ctx.closePath();
    ctx.fill();
  }

  // Mouth contour.
  ctx.beginPath();
  ctx.moveTo(
    mouthLeft.x,
    mouthBottom.y + faceWidth * 0.025
  );

  ctx.quadraticCurveTo(
    mouthX,
    mouthBottom.y + faceWidth * 0.09,
    mouthRight.x,
    mouthBottom.y + faceWidth * 0.025
  );

  ctx.strokeStyle = style.stroke;
  ctx.lineWidth = Math.max(1.2, faceWidth * 0.014);
  ctx.stroke();

  // Lower decorative ridge.
  ctx.beginPath();

  ctx.moveTo(
    eyeX - halfWidth * 0.5,
    lowerY - faceWidth * 0.08
  );

  ctx.quadraticCurveTo(
    eyeX,
    lowerY + faceWidth * 0.01,
    eyeX + halfWidth * 0.5,
    lowerY - faceWidth * 0.08
  );

  ctx.strokeStyle = style.accent;
  ctx.lineWidth = Math.max(1, faceWidth * 0.008);
  ctx.stroke();

  ctx.restore();
}

async function loadFaceLandmarker() {
  if (landmarker) return landmarker;
  if (modelPromise) return modelPromise;

  modelPromise = (async () => {
    setStatus("Loading the on-device face-tracking model…");

    const vision = await FilesetResolver.forVisionTasks(WASM_URL);

    const tracker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: "CPU"
      },
      runningMode: "VIDEO",
      numFaces: 1,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false
    });

    landmarker = tracker;
    return tracker;
  })();

  try {
    return await modelPromise;
  } catch (error) {
    modelPromise = null;
    throw error;
  }
}

function friendlyError(error) {
  const name = error?.name || "";
  const message = String(error?.message || "");

  if (
    name === "NotAllowedError" ||
    name === "PermissionDeniedError"
  ) {
    return "Camera permission denied. Allow camera access in your browser or device settings, then retry.";
  }

  if (
    name === "NotFoundError" ||
    name === "DevicesNotFoundError"
  ) {
    return "No compatible camera was found.";
  }

  if (
    name === "NotReadableError" ||
    name === "TrackStartError"
  ) {
    return "The camera is busy. Close other apps using the camera and retry.";
  }

  if (name === "OverconstrainedError") {
    return "The selected camera does not support the requested settings.";
  }

  if (name === "SecurityError") {
    return "Camera access is blocked by the current security context. Use HTTPS or the installed app.";
  }

  if (/fetch|network|wasm|model|http/i.test(message)) {
    return "The tracking model could not load. Check your internet connection and retry.";
  }

  return `MASKAM error: ${message || name || "Unknown error"}.`;
}

function enqueue(task) {
  const result = operation.then(task, task);
  operation = result.catch(() => {});
  return result;
}

function waitForVideo() {
  if (
    video.readyState >= HTMLMediaElement.HAVE_METADATA &&
    video.videoWidth > 0
  ) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    let settled = false;

    const cleanup = () => {
      video.removeEventListener("loadedmetadata", onReady);
      video.removeEventListener("error", onError);
    };

    const finish = (callback, value) => {
      if (settled) return;

      settled = true;
      cleanup();
      callback(value);
    };

    const onReady = () => {
      if (video.videoWidth > 0) {
        finish(resolve);
      }
    };

    const onError = () => {
      finish(
        reject,
        new Error("The camera video could not be initialized.")
      );
    };

    video.addEventListener("loadedmetadata", onReady);
    video.addEventListener("error", onError);

    // Avoid a race if metadata arrived just before listener registration.
    onReady();
  });
}

async function startCamera() {
  if (disposed) return;

  return enqueue(async () => {
    if (disposed || stream) return;

    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus(
        "Camera access requires HTTPS or a supported installed app."
      );
      return;
    }

    const myGeneration = ++generation;

    setControls(true);
    setStatus("Requesting camera access…");

    let candidateStream = null;

    try {
      candidateStream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: facingMode },
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30, max: 30 }
        }
      });

      if (disposed || myGeneration !== generation) {
        candidateStream.getTracks().forEach((track) => track.stop());
        return;
      }

      stream = candidateStream;

      for (const track of stream.getVideoTracks()) {
        track.addEventListener(
          "ended",
          handleTrackEnded,
          { once: true }
        );
      }

      video.srcObject = stream;
      video.muted = true;
      video.autoplay = true;
      video.playsInline = true;

      await waitForVideo();

      if (disposed || myGeneration !== generation || !stream) return;

      await video.play();

      if (disposed || myGeneration !== generation || !stream) return;

      stage.classList.add("has-camera");

      if (stageEmpty) {
        stageEmpty.hidden = true;
      }

      cameraLabel.textContent =
        facingMode === "user"
          ? "FRONT CAMERA"
          : "REAR CAMERA";

      resizeCanvas();
      resetSmoothing();

      setStatus("Camera ready. Loading face tracking…");

      await loadFaceLandmarker();

      if (disposed || myGeneration !== generation || !stream) return;

      setStatus("Looking for a face. Move into view.");
      setControls();
      scheduleFrame();
    } catch (error) {
      if (candidateStream && candidateStream !== stream) {
        candidateStream.getTracks().forEach((track) => track.stop());
      }

      if (myGeneration === generation) {
        cleanupCamera(false);
        setStatus(friendlyError(error));
      }

      console.error("MASKAM camera startup failed:", error);
    } finally {
      setControls();
    }
  });
}

function scheduleFrame() {
  if (disposed || !stream || animationId !== null) return;

  animationId = requestAnimationFrame(trackFrame);
}

function trackFrame(timestamp) {
  animationId = null;

  if (disposed || !stream || !landmarker) return;

  if (document.hidden) {
    scheduleFrame();
    return;
  }

  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    scheduleFrame();
    return;
  }

  if (
    video.currentTime === lastVideoTime ||
    timestamp - lastInferenceTime < TARGET_INTERVAL
  ) {
    scheduleFrame();
    return;
  }

  lastVideoTime = video.currentTime;
  lastInferenceTime = timestamp;

  try {
    const result = landmarker.detectForVideo(video, timestamp);

    clearCanvas();

    const detected = result.faceLandmarks?.[0];

    if (detected) {
      drawMask(smoothLandmarks(detected));
      setTrackingState(true, true);
      setStatus("Face detected. The mask is following your face.");
    } else {
      smoothedLandmarks = null;
      setTrackingState(true, false);
      setStatus(
        "Searching for a face. Move into view and face the camera."
      );
    }

    frameCount++;

    if (timestamp - lastFpsTime >= 1000) {
      const elapsed = timestamp - lastFpsTime;

      fpsLabel.textContent =
        `${Math.round((frameCount * 1000) / elapsed)} FPS`;

      frameCount = 0;
      lastFpsTime = timestamp;
    }
  } catch (error) {
    console.error("MASKAM tracking failed:", error);

    clearCanvas();
    setTrackingState(true, false);

    setStatus(
      "Tracking encountered an error. Stop and restart the camera."
    );

    // Do not repeatedly throw errors every animation frame.
    return;
  }

  scheduleFrame();
}

function cleanupCamera(updateStatus = true) {
  generation++;

  if (animationId !== null) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }

  const oldStream = stream;
  stream = null;

  if (oldStream) {
    oldStream.getTracks().forEach((track) => {
      track.removeEventListener("ended", handleTrackEnded);
      track.stop();
    });
  }

  video.pause();
  video.srcObject = null;

  video.style.transform = mirrorToggle.checked
    ? "scaleX(-1)"
    : "none";

  stage.classList.remove("has-camera");

  if (stageEmpty) {
    stageEmpty.hidden = false;
  }

  clearCanvas();
  resetSmoothing();

  cameraLabel.textContent = "CAMERA OFFLINE";
  fpsLabel.textContent = "-- FPS";

  setTrackingState(false, false);
  setControls();

  if (updateStatus) {
    setStatus("Camera is off. Nothing is being captured.");
  }
}

function stopCamera() {
  return enqueue(async () => {
    cleanupCamera(true);
  });
}

function handleTrackEnded() {
  if (disposed || !stream) return;

  cleanupCamera(false);

  setStatus(
    "The camera disconnected. Enable the camera to try again."
  );
}

async function flipCamera() {
  return enqueue(async () => {
    if (!stream || disposed) return;

    const nextFacing =
      facingMode === "user" ? "environment" : "user";

    cleanupCamera(false);
    facingMode = nextFacing;

    setStatus("Switching camera…");

    // Start the replacement camera in this queued operation.
    const myGeneration = ++generation;

    setControls(true);

    let candidateStream = null;

    try {
      candidateStream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: facingMode },
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30, max: 30 }
        }
      });

      if (disposed || myGeneration !== generation) {
        candidateStream.getTracks().forEach((track) => track.stop());
        return;
      }

      stream = candidateStream;

      for (const track of stream.getVideoTracks()) {
        track.addEventListener(
          "ended",
          handleTrackEnded,
          { once: true }
        );
      }

      video.srcObject = stream;
      video.muted = true;
      video.autoplay = true;
      video.playsInline = true;

      await waitForVideo();

      if (disposed || myGeneration !== generation || !stream) return;

      await video.play();

      if (disposed || myGeneration !== generation || !stream) return;

      stage.classList.add("has-camera");

      if (stageEmpty) {
        stageEmpty.hidden = true;
      }

      cameraLabel.textContent =
        facingMode === "user"
          ? "FRONT CAMERA"
          : "REAR CAMERA";

      resizeCanvas();
      resetSmoothing();

      await loadFaceLandmarker();

      if (disposed || myGeneration !== generation || !stream) return;

      setStatus("Looking for a face. Move into view.");
      scheduleFrame();
    } catch (error) {
      if (candidateStream && candidateStream !== stream) {
        candidateStream.getTracks().forEach((track) => track.stop());
      }

      if (myGeneration === generation) {
        cleanupCamera(false);
        setStatus(friendlyError(error));
      }

      console.error("MASKAM camera switch failed:", error);
    } finally {
      setControls();
    }
  });
}

function setMaskStyle(style) {
  if (!MASK_STYLES[style]) return;

  maskStyle = style;

  document.querySelectorAll(".mask-option").forEach((button) => {
    const selected = button.dataset.mask === style;

    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
}

function openSettings() {
  if (!settingsDialog) return;

  if (typeof settingsDialog.showModal === "function") {
    if (!settingsDialog.open) {
      settingsDialog.showModal();
    }
  } else {
    settingsDialog.setAttribute("open", "");
  }
}

function closeSettings() {
  if (!settingsDialog) return;

  if (typeof settingsDialog.close === "function") {
    if (settingsDialog.open) {
      settingsDialog.close();
    }
  } else {
    settingsDialog.removeAttribute("open");
  }
}

// Camera controls.
startBtn.addEventListener("click", () => {
  startCamera().catch((error) => {
    console.error(error);
    setStatus(friendlyError(error));
  });
});

stopBtn.addEventListener("click", () => {
  stopCamera().catch((error) => {
    console.error(error);
  });
});

flipBtn.addEventListener("click", () => {
  flipCamera().catch((error) => {
    console.error(error);
    setStatus(friendlyError(error));
  });
});

// Mask style controls.
document.querySelectorAll(".mask-option").forEach((button) => {
  button.addEventListener("click", () => {
    setMaskStyle(button.dataset.mask);
  });
});

// Mask opacity.
opacitySlider.addEventListener("input", () => {
  const value = Number(opacitySlider.value);

  maskOpacity = Math.min(100, Math.max(0, value)) / 100;

  opacityValue.value = `${value}%`;
  opacityValue.textContent = `${value}%`;
});

// Mirror preview and mask coordinates together.
mirrorToggle.addEventListener("change", () => {
  video.style.transform = mirrorToggle.checked
    ? "scaleX(-1)"
    : "none";

  clearCanvas();
});

// Optional settings controls.
$("#settingsBtn")?.addEventListener("click", openSettings);
$("#closeSettingsBtn")?.addEventListener("click", closeSettings);
$("#doneSettingsBtn")?.addEventListener("click", closeSettings);

// Responsive canvas.
window.addEventListener("resize", resizeCanvas);

if ("ResizeObserver" in window) {
  resizeObserver = new ResizeObserver(resizeCanvas);
  resizeObserver.observe(stage);
}

// Resume cleanly after returning to the app.
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && stream) {
    resetSmoothing();
    resizeCanvas();
  }
});

// Release camera and model resources when leaving the page.
window.addEventListener("pagehide", () => {
  disposed = true;

  cleanupCamera(false);
  resizeObserver?.disconnect();

  landmarker?.close();
  landmarker = null;
  modelPromise = null;
});

// Initial UI state.
video.style.transform = mirrorToggle.checked
  ? "scaleX(-1)"
  : "none";

document.querySelectorAll(".mask-option").forEach((button) => {
  button.setAttribute(
    "aria-pressed",
    String(button.dataset.mask === "classic")
  );
});

setControls();
setTrackingState(false, false);
resizeCanvas();
