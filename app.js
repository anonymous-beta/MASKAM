
import {
  FaceLandmarker,
  FilesetResolver
} from "@mediapipe/tasks-vision";

/* =========================================================
   MASKAM — camera, tracking and mask renderer
   MediaPipe Tasks Vision: 0.10.35
   ========================================================= */

const $ = (selector) => document.querySelector(selector);

const ui = {
  video: $("#camera"),
  canvas: $("#overlay"),
  stage: $("#stage"),
  start: $("#startBtn"),
  stop: $("#stopBtn"),
  flip: $("#flipBtn"),
  startText: $("#startBtnText"),
  status: $("#statusMessage"),
  badge: $("#trackingBadge"),
  cameraLabel: $("#cameraLabel"),
  fps: $("#fpsLabel"),
  opacity: $("#opacitySlider"),
  opacityValue: $("#opacityValue"),
  mirror: $("#mirrorToggle"),
  empty: $("#stageEmpty"),
  settings: $("#settingsDialog")
};

const ctx = ui.canvas?.getContext("2d", {
  alpha: true,
  desynchronized: true
});

for (const [key, element] of Object.entries(ui)) {
  if (
    key !== "empty" &&
    key !== "settings" &&
    !element
  ) {
    throw new Error(`MASKAM: required HTML element missing: ${key}`);
  }
}

if (!ctx) {
  throw new Error("MASKAM could not initialize the overlay canvas.");
}

/* -------------------- Configuration -------------------- */

const VERSION = "0.10.35";

const WASM_URL =
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/wasm`;

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const MAX_DPR = 2;
const TARGET_FPS = 24;
const FRAME_INTERVAL = 1000 / TARGET_FPS;
const LANDMARK_SMOOTHING = 0.42;
const MIN_FACE_WIDTH = 30;

const MASK_STYLES = {
  classic: {
    fill: "#e8e5dd",
    shadow: "#c5c0b5",
    stroke: "#332125",
    features: "#321b21",
    accent: "#a33c48",
    eye: "#0d0d11"
  },

  noir: {
    fill: "#303139",
    shadow: "#191a20",
    stroke: "#b8bbc4",
    features: "#e1e1e6",
    accent: "#777b89",
    eye: "#08090c"
  },

  crimson: {
    fill: "#e9e2dc",
    shadow: "#c7b7b1",
    stroke: "#471b24",
    features: "#481a24",
    accent: "#dc5261",
    eye: "#170b10"
  }
};

/* -------------------- Runtime state -------------------- */

let cameraStream = null;
let landmarker = null;
let modelPromise = null;

let cameraFacing = "user";
let currentStyle = "classic";
let opacity = clamp(Number(ui.opacity.value) / 100, 0, 1);

let rafId = null;
let resizeObserver = null;
let lastVideoTime = -1;
let lastInferenceAt = 0;
let lastFpsAt = performance.now();
let frameCount = 0;

let smoothed = null;
let operationQueue = Promise.resolve();
let generation = 0;
let disposed = false;
let lastStatus = "";
let lastTrackingState = "";

/* -------------------- Small utilities -------------------- */

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function setStatus(message) {
  if (lastStatus === message) return;

  lastStatus = message;
  ui.status.textContent = message;
}

function setTrackingState(state) {
  if (lastTrackingState === state) return;

  lastTrackingState = state;

  const found = state === "tracked";
  const active = state !== "standby";

  ui.badge.classList.toggle("active", found);

  const labels = {
    standby: "STANDBY",
    searching: "SEARCHING",
    tracked: "FACE TRACKED"
  };

  const dot = document.createElement("span");
  dot.className = "state-dot";

  ui.badge.replaceChildren(
    dot,
    document.createTextNode(` ${labels[state] || "STANDBY"}`)
  );

  ui.badge.setAttribute("aria-live", "polite");
  ui.badge.dataset.state = active ? state : "standby";
}

function updateControls(starting = false) {
  const active = Boolean(cameraStream);

  ui.start.disabled = starting || active || disposed;
  ui.stop.disabled = !active || starting || disposed;
  ui.flip.disabled = !active || starting || disposed;

  ui.startText.textContent = starting
    ? "Starting camera…"
    : active
      ? "Camera enabled"
      : "Enable camera";
}

function enqueue(task) {
  const result = operationQueue.then(task, task);

  operationQueue = result.catch((error) => {
    console.error("MASKAM operation failed:", error);
  });

  return result;
}

function clearCanvas() {
  const rect = ui.stage.getBoundingClientRect();

  ctx.clearRect(0, 0, rect.width, rect.height);
}

function resetTracking() {
  smoothed = null;
  lastVideoTime = -1;
  lastInferenceAt = 0;
  frameCount = 0;
  lastFpsAt = performance.now();
  ui.fps.textContent = "-- FPS";
}

function resizeCanvas() {
  if (disposed) return;

  const rect = ui.stage.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);

  const pixelWidth = Math.max(1, Math.round(rect.width * dpr));
  const pixelHeight = Math.max(1, Math.round(rect.height * dpr));

  if (
    ui.canvas.width !== pixelWidth ||
    ui.canvas.height !== pixelHeight
  ) {
    ui.canvas.width = pixelWidth;
    ui.canvas.height = pixelHeight;
  }

  ui.canvas.style.width = `${rect.width}px`;
  ui.canvas.style.height = `${rect.height}px`;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  clearCanvas();
}

/* -------------------- Face smoothing -------------------- */

function smoothLandmarks(points) {
  if (!smoothed || smoothed.length !== points.length) {
    smoothed = points.map((point) => ({
      x: point.x,
      y: point.y,
      z: point.z || 0
    }));

    return smoothed;
  }

  smoothed = points.map((point, index) => {
    const previous = smoothed[index];

    return {
      x: previous.x +
        (point.x - previous.x) * LANDMARK_SMOOTHING,

      y: previous.y +
        (point.y - previous.y) * LANDMARK_SMOOTHING,

      z: previous.z +
        ((point.z || 0) - previous.z) * LANDMARK_SMOOTHING
    };
  });

  return smoothed;
}

function distance(a, b) {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

/* =========================================================
   SOLID MASK RENDERER
   ========================================================= */

function drawMask(landmarks) {
  if (!landmarks || landmarks.length < 300) return;

  const rect = ui.stage.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  const videoWidth = ui.video.videoWidth;
  const videoHeight = ui.video.videoHeight;

  if (!width || !height || !videoWidth || !videoHeight) return;

  /*
   * Convert camera coordinates to stage coordinates.
   * This matches object-fit: cover for the video preview.
   */
  const scale = Math.max(
    width / videoWidth,
    height / videoHeight
  );

  const renderedWidth = videoWidth * scale;
  const renderedHeight = videoHeight * scale;

  const offsetX = (width - renderedWidth) / 2;
  const offsetY = (height - renderedHeight) / 2;

  const mirrored = ui.mirror.checked;

  function point(index) {
    const landmark = landmarks[index];

    let x = offsetX + landmark.x * renderedWidth;

    if (mirrored) {
      x = width - x;
    }

    return {
      x,
      y: offsetY + landmark.y * renderedHeight
    };
  }

  // MediaPipe face landmarks.
  const forehead = point(10);
  const chin = point(152);
  const leftCheek = point(234);
  const rightCheek = point(454);

  const leftEye = point(33);
  const rightEye = point(263);

  const noseBridge = point(168);
  const noseTip = point(1);
  const noseBottom = point(2);

  const mouthLeft = point(61);
  const mouthRight = point(291);
  const lowerLip = point(17);

  const faceWidth = distance(leftCheek, rightCheek);
  const faceHeight = distance(forehead, chin);

  if (
    faceWidth < MIN_FACE_WIDTH ||
    faceHeight < MIN_FACE_WIDTH ||
    !Number.isFinite(faceWidth) ||
    !Number.isFinite(faceHeight)
  ) {
    return;
  }

  const centerX = (leftCheek.x + rightCheek.x) / 2;
  const centerY = (leftEye.y + rightEye.y) / 2;
  const eyeDistance = distance(leftEye, rightEye);

  if (eyeDistance < 5) return;

  const style = MASK_STYLES[currentStyle];

  const halfWidth = faceWidth * 0.57;
  const topY = forehead.y - faceHeight * 0.025;
  const bottomY = chin.y + faceHeight * 0.015;

  const eyeLineAngle = Math.atan2(
    rightEye.y - leftEye.y,
    rightEye.x - leftEye.x
  );

  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  /*
   * A continuous, rounded mask silhouette.
   * The face is covered by a filled shape, not crossing lines.
   */
  ctx.beginPath();

  ctx.moveTo(
    centerX - halfWidth * 0.77,
    topY + faceHeight * 0.09
  );

  // Left forehead.
  ctx.bezierCurveTo(
    centerX - halfWidth * 0.92,
    topY - faceHeight * 0.02,
    centerX - halfWidth * 0.68,
    topY,
    centerX - halfWidth * 0.36,
    topY + faceHeight * 0.035
  );

  // Rounded upper forehead.
  ctx.quadraticCurveTo(
    centerX,
    topY + faceHeight * 0.075,
    centerX + halfWidth * 0.36,
    topY + faceHeight * 0.035
  );

  // Right forehead.
  ctx.bezierCurveTo(
    centerX + halfWidth * 0.68,
    topY,
    centerX + halfWidth * 0.92,
    topY - faceHeight * 0.02,
    centerX + halfWidth * 0.77,
    topY + faceHeight * 0.09
  );

  // Right temple and cheek.
  ctx.bezierCurveTo(
    centerX + halfWidth * 0.90,
    centerY + faceHeight * 0.10,
    centerX + halfWidth * 0.86,
    bottomY - faceHeight * 0.20,
    centerX + halfWidth * 0.62,
    bottomY - faceHeight * 0.075
  );

  // Chin, with a soft rounded bottom.
  ctx.bezierCurveTo(
    centerX + halfWidth * 0.43,
    bottomY + faceHeight * 0.025,
    centerX + halfWidth * 0.18,
    bottomY + faceHeight * 0.035,
    centerX,
    bottomY + faceHeight * 0.025
  );

  ctx.bezierCurveTo(
    centerX - halfWidth * 0.18,
    bottomY + faceHeight * 0.035,
    centerX - halfWidth * 0.43,
    bottomY + faceHeight * 0.025,
    centerX - halfWidth * 0.62,
    bottomY - faceHeight * 0.075
  );

  // Left cheek and return to forehead.
  ctx.bezierCurveTo(
    centerX - halfWidth * 0.86,
    bottomY - faceHeight * 0.20,
    centerX - halfWidth * 0.90,
    centerY + faceHeight * 0.10,
    centerX - halfWidth * 0.77,
    topY + faceHeight * 0.09
  );

  ctx.closePath();

  const bodyGradient = ctx.createLinearGradient(
    centerX - halfWidth,
    topY,
    centerX + halfWidth * 0.45,
    bottomY
  );

  bodyGradient.addColorStop(0, style.fill);
  bodyGradient.addColorStop(0.55, style.fill);
  bodyGradient.addColorStop(1, style.shadow);

  ctx.fillStyle = bodyGradient;
  ctx.fill();

  ctx.strokeStyle = style.stroke;
  ctx.lineWidth = Math.max(1.4, faceWidth * 0.011);
  ctx.stroke();

  /*
   * Eye openings.
   * They remain separate from the outer mask silhouette.
   */
  function drawEye(center) {
    const eyeWidth = eyeDistance * 0.40;
    const eyeHeight = faceHeight * 0.027;

    ctx.save();
    ctx.translate(
      center.x,
      center.y + faceHeight * 0.012
    );
    ctx.rotate(eyeLineAngle);

    ctx.beginPath();
    ctx.moveTo(-eyeWidth * 0.5, 0);

    ctx.quadraticCurveTo(
      -eyeWidth * 0.18,
      -eyeHeight * 1.45,
      eyeWidth * 0.5,
      0
    );

    ctx.quadraticCurveTo(
      eyeWidth * 0.15,
      eyeHeight * 1.3,
      -eyeWidth * 0.5,
      0
    );

    ctx.closePath();
    ctx.fillStyle = style.eye;
    ctx.fill();

    ctx.strokeStyle = style.stroke;
    ctx.lineWidth = Math.max(0.8, faceWidth * 0.005);
    ctx.stroke();

    ctx.restore();
  }

  drawEye(leftEye);
  drawEye(rightEye);

  /*
   * Nose: compact filled shape, following the real nose.
   */
  const noseWidth = faceWidth * 0.07;

  ctx.beginPath();

  ctx.moveTo(
    noseBridge.x - noseWidth * 0.40,
    noseBridge.y + faceHeight * 0.045
  );

  ctx.quadraticCurveTo(
    noseTip.x - noseWidth,
    noseTip.y + faceHeight * 0.02,
    noseTip.x - noseWidth * 0.68,
    noseBottom.y
  );

  ctx.quadraticCurveTo(
    noseTip.x,
    noseBottom.y + faceHeight * 0.035,
    noseTip.x + noseWidth * 0.68,
    noseBottom.y
  );

  ctx.quadraticCurveTo(
    noseTip.x + noseWidth,
    noseTip.y + faceHeight * 0.02,
    noseBridge.x + noseWidth * 0.40,
    noseBridge.y + faceHeight * 0.045
  );

  ctx.closePath();
  ctx.fillStyle = style.shadow;
  ctx.fill();

  ctx.strokeStyle = style.stroke;
  ctx.lineWidth = Math.max(0.8, faceWidth * 0.005);
  ctx.stroke();

  /*
   * Filled handlebar moustache.
   */
  const mouthCenterX = (mouthLeft.x + mouthRight.x) / 2;
  const mouthCenterY = (mouthLeft.y + mouthRight.y) / 2;

  const moustacheWidth =
    Math.abs(mouthRight.x - mouthLeft.x) * 0.43;

  const moustacheHeight = faceHeight * 0.045;

  function drawMoustacheLobe(side) {
    ctx.beginPath();

    ctx.moveTo(
      mouthCenterX,
      mouthCenterY - moustacheHeight * 0.05
    );

    ctx.bezierCurveTo(
      mouthCenterX + side * moustacheWidth * 0.42,
      mouthCenterY - moustacheHeight * 1.35,
      mouthCenterX + side * moustacheWidth * 0.96,
      mouthCenterY - moustacheHeight * 0.82,
      mouthCenterX + side * moustacheWidth,
      mouthCenterY - moustacheHeight * 0.12
    );

    ctx.bezierCurveTo(
      mouthCenterX + side * moustacheWidth * 0.80,
      mouthCenterY + moustacheHeight * 0.52,
      mouthCenterX + side * moustacheWidth * 0.28,
      mouthCenterY + moustacheHeight * 0.50,
      mouthCenterX,
      mouthCenterY + moustacheHeight * 0.22
    );

    ctx.closePath();
    ctx.fillStyle = style.features;
    ctx.fill();

    ctx.strokeStyle = style.stroke;
    ctx.lineWidth = Math.max(0.8, faceWidth * 0.005);
    ctx.stroke();
  }

  drawMoustacheLobe(-1);
  drawMoustacheLobe(1);

  /*
   * Mouth and smile.
   */
  const smileWidth =
    Math.abs(mouthRight.x - mouthLeft.x) * 0.68;

  const smileY = Math.max(
    mouthCenterY + faceHeight * 0.075,
    lowerLip.y + faceHeight * 0.025
  );

  ctx.beginPath();

  ctx.moveTo(
    mouthCenterX - smileWidth / 2,
    smileY
  );

  ctx.quadraticCurveTo(
    mouthCenterX,
    smileY + faceHeight * 0.045,
    mouthCenterX + smileWidth / 2,
    smileY
  );

  ctx.strokeStyle = style.features;
  ctx.lineWidth = Math.max(2, faceWidth * 0.013);
  ctx.stroke();

  // Restrained lower lip highlight.
  ctx.beginPath();

  ctx.moveTo(
    mouthCenterX - smileWidth * 0.33,
    smileY + faceHeight * 0.018
  );

  ctx.quadraticCurveTo(
    mouthCenterX,
    smileY + faceHeight * 0.050,
    mouthCenterX + smileWidth * 0.33,
    smileY + faceHeight * 0.018
  );

  ctx.strokeStyle = style.accent;
  ctx.lineWidth = Math.max(1, faceWidth * 0.005);
  ctx.stroke();

  /*
   * Very subtle cheek details.
   * These are short curves, not long lines across the face.
   */
  ctx.globalAlpha = opacity * 0.35;
  ctx.strokeStyle = style.accent;
  ctx.lineWidth = Math.max(1, faceWidth * 0.005);

  for (const side of [-1, 1]) {
    ctx.beginPath();

    ctx.moveTo(
      centerX + side * halfWidth * 0.72,
      centerY + faceHeight * 0.18
    );

    ctx.quadraticCurveTo(
      centerX + side * halfWidth * 0.57,
      centerY + faceHeight * 0.24,
      centerX + side * halfWidth * 0.43,
      centerY + faceHeight * 0.25
    );

    ctx.stroke();
  }

  ctx.restore();
}

/* -------------------- MediaPipe model -------------------- */

async function loadLandmarker() {
  if (landmarker) return landmarker;
  if (modelPromise) return modelPromise;

  modelPromise = (async () => {
    setStatus("Loading face-tracking model…");

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

    if (disposed) {
      tracker.close();
      throw new Error("MASKAM was closed while loading the model.");
    }

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

/* -------------------- Camera error messages -------------------- */

function explainError(error) {
  const name = error?.name || "";
  const message = String(error?.message || "");

  if (
    name === "NotAllowedError" ||
    name === "PermissionDeniedError"
  ) {
    return "Camera permission was denied. Allow camera access and retry.";
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
    return "The camera is busy. Close other apps using it and retry.";
  }

  if (name === "OverconstrainedError") {
    return "This camera cannot satisfy the requested video settings.";
  }

  if (
    /wasm|model|fetch|network|failed to load|http/i.test(message)
  ) {
    return "The face-tracking model could not load. Check your internet connection.";
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    return "Camera access requires HTTPS or a supported installed app.";
  }

  return `Camera or tracking error: ${message || name || "Unknown error"}`;
}

/* -------------------- Video readiness -------------------- */

function waitForVideo() {
  if (
    ui.video.readyState >= HTMLMediaElement.HAVE_METADATA &&
    ui.video.videoWidth > 0
  ) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    let settled = false;

    const cleanup = () => {
      ui.video.removeEventListener("loadedmetadata", onReady);
      ui.video.removeEventListener("error", onError);
    };

    const finish = (callback, value) => {
      if (settled) return;

      settled = true;
      cleanup();
      callback(value);
    };

    const onReady = () => {
      if (ui.video.videoWidth > 0) {
        finish(resolve);
      }
    };

    const onError = () => {
      finish(
        reject,
        new Error("Camera video failed to initialize.")
      );
    };

    ui.video.addEventListener("loadedmetadata", onReady);
    ui.video.addEventListener("error", onError);

    // Prevent a metadata-listener timing race.
    onReady();
  });
}

/* -------------------- Camera lifecycle -------------------- */

async function startCamera() {
  return enqueue(async () => {
    if (disposed || cameraStream) return;

    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus(
        "Camera access requires HTTPS or a supported installed app."
      );
      return;
    }

    const myGeneration = ++generation;
    let candidate = null;

    updateControls(true);
    setStatus("Requesting camera access…");

    try {
      candidate = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: cameraFacing },
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30, max: 30 }
        }
      });

      if (disposed || myGeneration !== generation) {
        candidate.getTracks().forEach((track) => track.stop());
        return;
      }

      cameraStream = candidate;

      for (const track of cameraStream.getVideoTracks()) {
        track.addEventListener("ended", handleCameraEnded, {
          once: true
        });
      }

      ui.video.srcObject = cameraStream;
      ui.video.muted = true;
      ui.video.playsInline = true;
      ui.video.autoplay = true;

      await waitForVideo();

      if (disposed || myGeneration !== generation || !cameraStream) {
        return;
      }

      await ui.video.play();

      if (disposed || myGeneration !== generation || !cameraStream) {
        return;
      }

      ui.stage.classList.add("has-camera");

      if (ui.empty) {
        ui.empty.hidden = true;
      }

      ui.cameraLabel.textContent =
        cameraFacing === "user" ? "FRONT CAMERA" : "REAR CAMERA";

      resizeCanvas();
      resetTracking();

      await loadLandmarker();

      if (disposed || myGeneration !== generation || !cameraStream) {
        return;
      }

      setTrackingState("searching");
      setStatus("Looking for a face. Move into view.");
      updateControls();

      scheduleFrame();
    } catch (error) {
      if (candidate && candidate !== cameraStream) {
        candidate.getTracks().forEach((track) => track.stop());
      }

      if (myGeneration === generation) {
        cleanupCamera(false);
        setStatus(explainError(error));
      }

      console.error("MASKAM camera startup error:", error);
    } finally {
      updateControls();
    }
  });
}

function scheduleFrame() {
  if (disposed || !cameraStream || rafId !== null) return;

  rafId = requestAnimationFrame(trackFrame);
}

function trackFrame(timestamp) {
  rafId = null;

  if (disposed || !cameraStream || !landmarker) return;

  if (document.hidden) {
    scheduleFrame();
    return;
  }

  if (ui.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    scheduleFrame();
    return;
  }

  if (
    ui.video.currentTime === lastVideoTime ||
    timestamp - lastInferenceAt < FRAME_INTERVAL
  ) {
    scheduleFrame();
    return;
  }

  lastVideoTime = ui.video.currentTime;
  lastInferenceAt = timestamp;

  try {
    const result = landmarker.detectForVideo(ui.video, timestamp);
    clearCanvas();

    const detected = result.faceLandmarks?.[0];

    if (detected) {
      drawMask(smoothLandmarks(detected));
      setTrackingState("tracked");
      setStatus("Face detected. The mask is following your face.");
    } else {
      smoothed = null;
      setTrackingState("searching");
      setStatus("Searching for a face. Move into view.");
    }

    frameCount++;

    if (timestamp - lastFpsAt >= 1000) {
      const elapsed = timestamp - lastFpsAt;

      ui.fps.textContent =
        `${Math.round(frameCount * 1000 / elapsed)} FPS`;

      frameCount = 0;
      lastFpsAt = timestamp;
    }
  } catch (error) {
    console.error("MASKAM face tracking error:", error);

    clearCanvas();
    setTrackingState("searching");
    setStatus("Tracking failed. Stop and restart the camera.");

    return;
  }

  scheduleFrame();
}

function cleanupCamera(showStatus = true) {
  generation++;

  if (rafId !== null) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }

  const oldStream = cameraStream;
  cameraStream = null;

  if (oldStream) {
    for (const track of oldStream.getTracks()) {
      track.removeEventListener("ended", handleCameraEnded);
      track.stop();
    }
  }

  ui.video.pause();
  ui.video.srcObject = null;

  ui.stage.classList.remove("has-camera");

  if (ui.empty) {
    ui.empty.hidden = false;
  }

  clearCanvas();
  resetTracking();

  ui.cameraLabel.textContent = "CAMERA OFFLINE";

  setTrackingState("standby");
  updateControls();

  if (showStatus) {
    setStatus("Camera is off. No camera frames are being processed.");
  }
}

function stopCamera() {
  return enqueue(async () => {
    cleanupCamera(true);
  });
}

function handleCameraEnded() {
  if (disposed || !cameraStream) return;

  cleanupCamera(false);

  setStatus("Camera disconnected. Enable the camera to reconnect.");
}

async function flipCamera() {
  return enqueue(async () => {
    if (disposed || !cameraStream) return;

    const nextFacing =
      cameraFacing === "user" ? "environment" : "user";

    cleanupCamera(false);
    cameraFacing = nextFacing;

    setStatus("Switching camera…");

    // This operation already owns the queue, so start the replacement
    // camera directly rather than enqueueing startCamera again.
    const myGeneration = ++generation;
    let candidate = null;

    updateControls(true);

    try {
      candidate = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: cameraFacing },
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30, max: 30 }
        }
      });

      if (disposed || myGeneration !== generation) {
        candidate.getTracks().forEach((track) => track.stop());
        return;
      }

      cameraStream = candidate;

      for (const track of cameraStream.getVideoTracks()) {
        track.addEventListener("ended", handleCameraEnded, {
          once: true
        });
      }

      ui.video.srcObject = cameraStream;
      ui.video.muted = true;
      ui.video.playsInline = true;
      ui.video.autoplay = true;

      await waitForVideo();

      if (disposed || myGeneration !== generation || !cameraStream) {
        return;
      }

      await ui.video.play();

      if (disposed || myGeneration !== generation || !cameraStream) {
        return;
      }

      ui.stage.classList.add("has-camera");

      if (ui.empty) {
        ui.empty.hidden = true;
      }

      ui.cameraLabel.textContent =
        cameraFacing === "user" ? "FRONT CAMERA" : "REAR CAMERA";

      resizeCanvas();
      resetTracking();

      await loadLandmarker();

      if (disposed || myGeneration !== generation || !cameraStream) {
        return;
      }

      setTrackingState("searching");
      setStatus("Looking for a face. Move into view.");

      scheduleFrame();
    } catch (error) {
      if (candidate && candidate !== cameraStream) {
        candidate.getTracks().forEach((track) => track.stop());
      }

      if (myGeneration === generation) {
        cleanupCamera(false);
        setStatus(explainError(error));
      }

      console.error("MASKAM camera switch error:", error);
    } finally {
      updateControls();
    }
  });
}

/* -------------------- Appearance controls -------------------- */

function selectMaskStyle(style) {
  if (!MASK_STYLES[style]) return;

  currentStyle = style;

  document.querySelectorAll(".mask-option").forEach((button) => {
    const selected = button.dataset.mask === style;

    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
}

function updateOpacity() {
  const value = clamp(Number(ui.opacity.value), 0, 100);

  opacity = value / 100;

  if ("value" in ui.opacityValue) {
    ui.opacityValue.value = `${value}%`;
  }

  ui.opacityValue.textContent = `${value}%`;
}

function updateMirror() {
  /*
   * Video and overlay are mirrored independently.
   * drawMask() mirrors its X coordinates to match the preview.
   */
  ui.video.style.transform = ui.mirror.checked
    ? "scaleX(-1)"
    : "none";

  clearCanvas();
}

/* -------------------- Optional settings dialog -------------------- */

function openSettings() {
  if (!ui.settings) return;

  if (typeof ui.settings.showModal === "function") {
    if (!ui.settings.open) ui.settings.showModal();
  } else {
    ui.settings.setAttribute("open", "");
  }
}

function closeSettings() {
  if (!ui.settings) return;

  if (typeof ui.settings.close === "function") {
    if (ui.settings.open) ui.settings.close();
  } else {
    ui.settings.removeAttribute("open");
  }
}

/* -------------------- Event listeners -------------------- */

ui.start.addEventListener("click", () => {
  startCamera().catch((error) => {
    console.error(error);
    setStatus(explainError(error));
  });
});

ui.stop.addEventListener("click", () => {
  stopCamera().catch(console.error);
});

ui.flip.addEventListener("click", () => {
  flipCamera().catch((error) => {
    console.error(error);
    setStatus(explainError(error));
  });
});

ui.opacity.addEventListener("input", updateOpacity);
ui.mirror.addEventListener("change", updateMirror);

document.querySelectorAll(".mask-option").forEach((button) => {
  button.addEventListener("click", () => {
    selectMaskStyle(button.dataset.mask);
  });
});

$("#settingsBtn")?.addEventListener("click", openSettings);
$("#closeSettingsBtn")?.addEventListener("click", closeSettings);
$("#doneSettingsBtn")?.addEventListener("click", closeSettings);

window.addEventListener("resize", resizeCanvas);

if ("ResizeObserver" in window) {
  resizeObserver = new ResizeObserver(resizeCanvas);
  resizeObserver.observe(ui.stage);
}

document.addEventListener("visibilitychange", () => {
  if (!document.hidden && cameraStream) {
    resetTracking();
    resizeCanvas();
  }
});

window.addEventListener("pagehide", () => {
  disposed = true;

  cleanupCamera(false);
  resizeObserver?.disconnect();

  if (landmarker) {
    landmarker.close();
    landmarker = null;
  }

  modelPromise = null;
});

/* -------------------- Initial state -------------------- */

ui.video.style.transform = ui.mirror.checked
  ? "scaleX(-1)"
  : "none";

selectMaskStyle("classic");
updateOpacity();
updateControls();
setTrackingState("standby");
resizeCanvas();
