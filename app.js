
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const $ = (selector) => document.querySelector(selector);

const video = $("#camera");
const canvas = $("#overlay");
const ctx = canvas?.getContext("2d");
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

if (
  !video || !canvas || !ctx || !stage || !startBtn || !stopBtn ||
  !flipBtn || !startBtnText || !statusMessage || !trackingBadge ||
  !cameraLabel || !fpsLabel || !opacitySlider || !opacityValue ||
  !mirrorToggle
) {
  throw new Error("MASKAM could not initialize: a required HTML element is missing.");
}

let stream = null;
let landmarker = null;
let modelPromise = null;
let startPromise = null;
let animationId = null;
let lastVideoTime = -1;
let facingMode = "user";
let maskStyle = "classic";
let maskOpacity = Number(opacitySlider.value) / 100;
let disposed = false;
let trackingError = false;
let lastStatus = "";
let lastInferenceTime = 0;
let lastFpsTime = 0;
let frames = 0;
let fps = 0;

const TARGET_INTERVAL = 1000 / 20;

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

function setStatus(message) {
  // Avoid repeated DOM updates on every video frame.
  if (message === lastStatus) return;
  lastStatus = message;
  statusMessage.textContent = message;
}

function setTrackingState(active, found = false) {
  trackingBadge.classList.toggle("active", active && found);
  trackingBadge.innerHTML =
    `<span class="state-dot"></span> ${
      !active ? "STANDBY" : found ? "FACE TRACKED" : "SEARCHING"
    }`;
}

function setControls() {
  const starting = Boolean(startPromise);

  startBtn.disabled = starting || Boolean(stream);
  stopBtn.disabled = !stream;
  flipBtn.disabled = !stream || starting;

  startBtnText.textContent = starting
    ? "Starting…"
    : stream
      ? "Camera enabled"
      : "Enable camera";
}

function resizeCanvas() {
  const rect = stage.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);

  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function clearCanvas() {
  const rect = stage.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function drawMask(landmarks) {
  const rect = stage.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  const vw = video.videoWidth;
  const vh = video.videoHeight;

  if (!vw || !vh || !width || !height) return;

  // Align landmarks with the video's object-fit: cover display.
  const scale = Math.max(width / vw, height / vh);
  const dw = vw * scale;
  const dh = vh * scale;
  const ox = (width - dw) / 2;
  const oy = (height - dh) / 2;

  function point(index) {
    const p = landmarks[index];
    let x = ox + p.x * dw;
    if (mirrorToggle.checked) x = width - x;
    return { x, y: oy + p.y * dh };
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
  const cheekY = eyeY + faceWidth * 0.19;
  const lowerY = chin.y - faceWidth * 0.02;
  const halfWidth = faceWidth * 0.53;
  const templeY = forehead.y + faceWidth * 0.28;

  ctx.save();
  ctx.globalAlpha = maskOpacity;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  // Classic sculpted mask silhouette.
  ctx.beginPath();
  ctx.moveTo(eyeX - halfWidth * 0.91, templeY);
  ctx.quadraticCurveTo(
    eyeX - halfWidth * 0.74, eyeY - faceWidth * 0.05,
    eyeX - eyeDistance * 0.48, eyeY - faceWidth * 0.02
  );
  ctx.lineTo(nose.x - faceWidth * 0.055, nose.y + faceWidth * 0.08);
  ctx.quadraticCurveTo(
    nose.x, nose.y + faceWidth * 0.15,
    nose.x + faceWidth * 0.055, nose.y + faceWidth * 0.08
  );
  ctx.lineTo(eyeX + eyeDistance * 0.48, eyeY - faceWidth * 0.02);
  ctx.quadraticCurveTo(
    eyeX + halfWidth * 0.74, eyeY - faceWidth * 0.05,
    eyeX + halfWidth * 0.91, templeY
  );
  ctx.quadraticCurveTo(
    rightFace.x + faceWidth * 0.02, cheekY,
    eyeX + halfWidth * 0.66, lowerY - faceWidth * 0.16
  );
  ctx.quadraticCurveTo(
    eyeX, lowerY + faceWidth * 0.08,
    eyeX - halfWidth * 0.66, lowerY - faceWidth * 0.16
  );
  ctx.quadraticCurveTo(
    leftFace.x - faceWidth * 0.02, cheekY,
    eyeX - halfWidth * 0.91, templeY
  );
  ctx.closePath();

  ctx.fillStyle = style.fill;
  ctx.fill();
  ctx.strokeStyle = style.stroke;
  ctx.lineWidth = Math.max(1.2, faceWidth * 0.012);
  ctx.stroke();

  const gradient = ctx.createLinearGradient(eyeX, eyeY, eyeX, lowerY);
  gradient.addColorStop(0, "#ffffff18");
  gradient.addColorStop(1, "#00000020");
  ctx.fillStyle = gradient;
  ctx.fill();

  const eyeWidth = Math.max(12, eyeDistance * 0.39);
  const eyeHeight = Math.max(7, faceWidth * 0.073);

  function drawEye(center, angle) {
    ctx.save();
    ctx.translate(center.x, center.y + faceWidth * 0.025);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.moveTo(-eyeWidth * 0.58, -eyeHeight * 0.12);
    ctx.quadraticCurveTo(
      -eyeWidth * 0.2, -eyeHeight * 0.75, 0, -eyeHeight * 0.2
    );
    ctx.quadraticCurveTo(
      eyeWidth * 0.28, -eyeHeight * 0.55,
      eyeWidth * 0.58, 0
    );
    ctx.quadraticCurveTo(
      eyeWidth * 0.22, eyeHeight * 0.7,
      0, eyeHeight * 0.18
    );
    ctx.quadraticCurveTo(
      -eyeWidth * 0.28, eyeHeight * 0.6,
      -eyeWidth * 0.58, -eyeHeight * 0.12
    );
    ctx.closePath();
    ctx.fillStyle = "#08090c";
    ctx.fill();
    ctx.strokeStyle = style.features;
    ctx.lineWidth = Math.max(1, faceWidth * 0.009);
    ctx.stroke();
    ctx.restore();
  }

  const eyeAngle = Math.atan2(
    rightEye.y - leftEye.y,
    rightEye.x - leftEye.x
  );

  drawEye(leftEye, eyeAngle);
  drawEye(rightEye, eyeAngle);

  // Nose detail.
  ctx.beginPath();
  ctx.moveTo(nose.x - faceWidth * 0.075, nose.y + faceWidth * 0.13);
  ctx.quadraticCurveTo(
    nose.x, nose.y + faceWidth * 0.18,
    nose.x + faceWidth * 0.075, nose.y + faceWidth * 0.13
  );
  ctx.strokeStyle = style.accent;
  ctx.lineWidth = Math.max(1, faceWidth * 0.01);
  ctx.stroke();

  // Two halves of the signature moustache.
  const mouth = {
    x: (mouthLeft.x + mouthRight.x) / 2,
    y: (mouthLeft.y + mouthRight.y) / 2
  };
  const moustacheY = mouth.y - faceWidth * 0.015;
  const moustacheHalf = faceWidth * 0.20;

  ctx.fillStyle = style.features;

  for (const direction of [-1, 1]) {
    ctx.beginPath();
    ctx.moveTo(mouth.x, moustacheY);
    ctx.bezierCurveTo(
      mouth.x + direction * moustacheHalf * 0.55,
      moustacheY - faceWidth * 0.10,
      mouth.x + direction * moustacheHalf * 1.05,
      moustacheY - faceWidth * 0.02,
      mouth.x + direction * moustacheHalf * 1.10,
      moustacheY + faceWidth * 0.055
    );
    ctx.bezierCurveTo(
      mouth.x + direction * moustacheHalf * 0.68,
      moustacheY + faceWidth * 0.035,
      mouth.x + direction * moustacheHalf * 0.27,
      moustacheY + faceWidth * 0.11,
      mouth.x,
      moustacheY + faceWidth * 0.075
    );
    ctx.closePath();
    ctx.fill();
  }

  // Smile and lower accent.
  ctx.beginPath();
  ctx.moveTo(mouthLeft.x, mouthBottom.y + faceWidth * 0.025);
  ctx.quadraticCurveTo(
    mouth.x, mouthBottom.y + faceWidth * 0.09,
    mouthRight.x, mouthBottom.y + faceWidth * 0.025
  );
  ctx.strokeStyle = style.stroke;
  ctx.lineWidth = Math.max(1.2, faceWidth * 0.014);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(eyeX - halfWidth * 0.5, lowerY - faceWidth * 0.08);
  ctx.quadraticCurveTo(
    eyeX, lowerY + faceWidth * 0.01,
    eyeX + halfWidth * 0.5, lowerY - faceWidth * 0.08
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
    setStatus("Loading the face-tracking model. Internet is required for the first load.");

    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm"
    );

    const tracker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
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
    // Permit a later retry instead of caching a rejected promise forever.
    modelPromise = null;
    throw error;
  }
}

function friendlyError(error) {
  const name = error?.name || "";
  const message = String(error?.message || "");

  if (name === "NotAllowedError" || name === "PermissionDeniedError") {
    return "Camera permission was denied. Allow camera access in Android settings and retry.";
  }

  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return "No compatible camera was found.";
  }

  if (name === "NotReadableError" || name === "TrackStartError") {
    return "The camera is busy. Close other camera apps and try again.";
  }

  if (name === "OverconstrainedError") {
    return "This camera does not support the requested settings. Try again.";
  }

  if (/fetch|network|wasm|model|http/i.test(message)) {
    return "The camera may work, but the tracking model failed to load. Check your internet connection and retry.";
  }

  return `Unable to start MASKAM: ${message || name || "unknown error"}.`;
}

async function startCamera() {
  if (startPromise || stream) return startPromise;

  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("Camera access needs the installed app or a secure HTTPS website.");
    return;
  }

  const attempt = (async () => {
    startBtn.disabled = true;
    stopBtn.disabled = true;
    flipBtn.disabled = true;
    startBtnText.textContent = "Starting…";
    setStatus("Requesting camera access…");

    let candidate = null;

    try {
      // Ask for the camera first. Do not make camera permission depend
      // on whether the tracking model loads.
      candidate = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: facingMode },
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 24, max: 30 }
        }
      });

      stream = candidate;
      video.srcObject = candidate;
      video.muted = true;
      video.playsInline = true;

      await video.play();

      if (!video.videoWidth) {
        await new Promise((resolve, reject) => {
          const timeout = setTimeout(() => {
            cleanup();
            reject(new Error("Camera video did not start in time."));
          }, 10000);

          function onReady() {
            cleanup();
            resolve();
          }

          function cleanup() {
            clearTimeout(timeout);
            video.removeEventListener("loadedmetadata", onReady);
          }

          video.addEventListener("loadedmetadata", onReady, { once: true });
        });
      }

      if (!video.videoWidth || !video.videoHeight) {
        throw new Error("The camera opened without producing video.");
      }

      stage.classList.add("has-camera");
      resizeCanvas();
      cameraLabel.textContent = "CAMERA ONLINE";
      setStatus("Camera connected. Preparing local face tracking…");

      // A failed model load should not silently leave dead controls.
      // The catch below stops the stream and provides a useful message.
      await loadFaceLandmarker();

      if (stream !== candidate) return;

      disposed = false;
      trackingError = false;
      lastVideoTime = -1;
      lastInferenceTime = 0;
      lastFpsTime = performance.now();
      frames = 0;

      setTrackingState(true, false);
      setStatus("Ready. Move into the frame to align your mask.");
      trackFrame();
    } catch (error) {
      console.error("MASKAM startup error:", error);

      if (candidate) {
        candidate.getTracks().forEach((track) => track.stop());
      }

      if (stream === candidate) stream = null;

      video.pause();
      video.srcObject = null;
      stage.classList.remove("has-camera");
      clearCanvas();

      cameraLabel.textContent = "CAMERA OFFLINE";
      fpsLabel.textContent = "-- FPS";
      setTrackingState(false, false);
      setStatus(friendlyError(error));
    } finally {
      setControls();
    }
  })();

  startPromise = attempt;

  try {
    await attempt;
  } finally {
    startPromise = null;
    setControls();
  }
}

function trackFrame() {
  if (disposed || !stream || !landmarker) return;

  animationId = requestAnimationFrame(trackFrame);

  if (document.hidden) return;
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  if (video.currentTime === lastVideoTime) return;

  const now = performance.now();

  // Limit inference frequency so the UI has time to process taps and
  // control updates on mid-range Android phones.
  if (now - lastInferenceTime < TARGET_INTERVAL) return;

  lastInferenceTime = now;
  lastVideoTime = video.currentTime;

  try {
    const result = landmarker.detectForVideo(video, now);
    clearCanvas();

    const landmarks = result.faceLandmarks?.[0];

    if (landmarks) {
      drawMask(landmarks);
      setTrackingState(true, true);
      setStatus("Face detected. The mask is following your face.");
    } else {
      setTrackingState(true, false);
      setStatus("Searching for a face. Move into view and face the camera.");
    }

    trackingError = false;
  } catch (error) {
    console.error("MASKAM tracking error:", error);
    trackingError = true;
    setTrackingState(true, false);
    setStatus("Tracking stopped after an error. Stop the camera and try again.");
    cancelAnimationFrame(animationId);
    animationId = null;
    return;
  }

  frames++;

  if (now - lastFpsTime >= 1000) {
    fps = Math.round((frames * 1000) / (now - lastFpsTime));
    fpsLabel.textContent = `${fps} FPS`;
    frames = 0;
    lastFpsTime = now;
  }
}

function stopCamera() {
  disposed = true;

  if (animationId !== null) {
    cancelAnimationFrame(animationId);
  }

  animationId = null;

  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }

  stream = null;
  video.pause();
  video.srcObject = null;
  stage.classList.remove("has-camera");
  clearCanvas();

  cameraLabel.textContent = "CAMERA OFFLINE";
  fpsLabel.textContent = "-- FPS";
  setTrackingState(false, false);
  setStatus("Camera is off. Nothing is being captured.");
  setControls();
}

async function flipCamera() {
  if (!stream || startPromise) return;

  const nextFacing = facingMode === "user" ? "environment" : "user";
  stopCamera();
  facingMode = nextFacing;
  await startCamera();
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

startBtn.addEventListener("click", () => {
  startCamera().catch((error) => {
    console.error("MASKAM start button error:", error);
    setStatus(friendlyError(error));
  });
});

stopBtn.addEventListener("click", stopCamera);
flipBtn.addEventListener("click", () => {
  flipCamera().catch((error) => {
    console.error("MASKAM camera switch error:", error);
    setStatus(friendlyError(error));
  });
});

document.querySelectorAll(".mask-option").forEach((button) => {
  button.addEventListener("click", () => setMaskStyle(button.dataset.mask));
});

opacitySlider.addEventListener("input", () => {
  maskOpacity = Number(opacitySlider.value) / 100;
  opacityValue.value = `${opacitySlider.value}%`;
  opacityValue.textContent = `${opacitySlider.value}%`;
});

mirrorToggle.addEventListener("change", () => {
  video.style.transform = mirrorToggle.checked ? "scaleX(-1)" : "none";
  clearCanvas();
});

function openSettings() {
  if (!settingsDialog) {
    setStatus("Settings are unavailable in this version.");
    return;
  }

  if (typeof settingsDialog.showModal === "function") {
    if (!settingsDialog.open) settingsDialog.showModal();
  } else {
    // Fallback for WebViews without the native dialog API.
    settingsDialog.setAttribute("open", "");
  }
}

function closeSettings() {
  if (!settingsDialog) return;

  if (typeof settingsDialog.close === "function") {
    if (settingsDialog.open) settingsDialog.close();
  } else {
    settingsDialog.removeAttribute("open");
  }
}

$("#settingsBtn")?.addEventListener("click", openSettings);
$("#closeSettingsBtn")?.addEventListener("click", closeSettings);
$("#doneSettingsBtn")?.addEventListener("click", closeSettings);

window.addEventListener("resize", resizeCanvas);

// Pause inference when the app is backgrounded, but don't automatically
// stop the camera merely because the browser temporarily hides the page.
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && stream) {
    lastVideoTime = -1;
    lastInferenceTime = 0;
    resizeCanvas();
  }
});

window.addEventListener("pagehide", stopCamera);

video.style.transform = mirrorToggle.checked ? "scaleX(-1)" : "none";

document.querySelectorAll(".mask-option").forEach((button) => {
  button.setAttribute(
    "aria-pressed",
    String(button.dataset.mask === "classic")
  );
});

setControls();
setTrackingState(false, false);
resizeCanvas();
