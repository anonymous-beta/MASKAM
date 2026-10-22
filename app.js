
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const $ = (selector) => document.querySelector(selector);

const video = $("#camera");
const canvas = $("#overlay");
const ctx = canvas.getContext("2d");
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

let stream = null;
let landmarker = null;
let animationId = null;
let lastVideoTime = -1;
let facingMode = "user";
let maskStyle = "classic";
let maskOpacity = 0.96;
let disposed = false;
let starting = false;
let frameCount = 0;
let lastFpsUpdate = 0;

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
  statusMessage.textContent = message;
}

function setTrackingState(active, found = false) {
  trackingBadge.classList.toggle("active", active && found);
  trackingBadge.innerHTML = `
    <span class="state-dot"></span>
    ${!active ? "STANDBY" : found ? "FACE TRACKED" : "SEARCHING"}
  `;
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
  const videoWidth = video.videoWidth;
  const videoHeight = video.videoHeight;

  if (!videoWidth || !videoHeight || !width || !height) return;

  // Match the video element's object-fit: cover geometry.
  const scale = Math.max(width / videoWidth, height / videoHeight);
  const drawWidth = videoWidth * scale;
  const drawHeight = videoHeight * scale;
  const offsetX = (width - drawWidth) / 2;
  const offsetY = (height - drawHeight) / 2;

  function mapped(index) {
    const point = landmarks[index];
    let x = offsetX + point.x * drawWidth;
    const y = offsetY + point.y * drawHeight;

    // Mirror the mask in the same way as the preview.
    if (mirrorToggle.checked) x = width - x;

    return { x, y, z: point.z || 0 };
  }

  const forehead = mapped(10);
  const chin = mapped(152);
  const leftFace = mapped(234);
  const rightFace = mapped(454);
  const leftEye = mapped(33);
  const rightEye = mapped(263);
  const nose = mapped(1);
  const mouthLeft = mapped(61);
  const mouthRight = mapped(291);
  const mouthBottom = mapped(14);

  const faceWidth = distance(leftFace, rightFace);
  if (!Number.isFinite(faceWidth) || faceWidth < 25) return;

  const style = MASK_STYLES[maskStyle];
  const eyeMidX = (leftEye.x + rightEye.x) / 2;
  const eyeMidY = (leftEye.y + rightEye.y) / 2;
  const eyeDistance = distance(leftEye, rightEye);
  const cheekY = eyeMidY + faceWidth * 0.19;
  const lowerY = chin.y - faceWidth * 0.02;
  const halfWidth = faceWidth * 0.53;
  const templeY = forehead.y + faceWidth * 0.28;

  ctx.save();
  ctx.globalAlpha = maskOpacity;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  // Sculpted Anonymous-inspired mask silhouette.
  ctx.beginPath();
  ctx.moveTo(eyeMidX - halfWidth * 0.91, templeY);
  ctx.quadraticCurveTo(
    eyeMidX - halfWidth * 0.74,
    eyeMidY - faceWidth * 0.05,
    eyeMidX - eyeDistance * 0.48,
    eyeMidY - faceWidth * 0.02
  );
  ctx.lineTo(nose.x - faceWidth * 0.055, nose.y + faceWidth * 0.08);
  ctx.quadraticCurveTo(
    nose.x,
    nose.y + faceWidth * 0.15,
    nose.x + faceWidth * 0.055,
    nose.y + faceWidth * 0.08
  );
  ctx.lineTo(eyeMidX + eyeDistance * 0.48, eyeMidY - faceWidth * 0.02);
  ctx.quadraticCurveTo(
    eyeMidX + halfWidth * 0.74,
    eyeMidY - faceWidth * 0.05,
    eyeMidX + halfWidth * 0.91,
    templeY
  );
  ctx.quadraticCurveTo(
    rightFace.x + faceWidth * 0.02,
    cheekY,
    eyeMidX + halfWidth * 0.66,
    lowerY - faceWidth * 0.16
  );
  ctx.quadraticCurveTo(
    eyeMidX,
    lowerY + faceWidth * 0.08,
    eyeMidX - halfWidth * 0.66,
    lowerY - faceWidth * 0.16
  );
  ctx.quadraticCurveTo(
    leftFace.x - faceWidth * 0.02,
    cheekY,
    eyeMidX - halfWidth * 0.91,
    templeY
  );
  ctx.closePath();

  ctx.fillStyle = style.fill;
  ctx.fill();
  ctx.lineWidth = Math.max(1.2, faceWidth * 0.012);
  ctx.strokeStyle = style.stroke;
  ctx.stroke();

  const gradient = ctx.createLinearGradient(
    eyeMidX, eyeMidY, eyeMidX, lowerY
  );
  gradient.addColorStop(0, "#ffffff18");
  gradient.addColorStop(1, "#00000020");
  ctx.fillStyle = gradient;
  ctx.fill();

  // Eye openings.
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
      eyeWidth * 0.28, -eyeHeight * 0.55, eyeWidth * 0.58, 0
    );
    ctx.quadraticCurveTo(
      eyeWidth * 0.22, eyeHeight * 0.7, 0, eyeHeight * 0.18
    );
    ctx.quadraticCurveTo(
      -eyeWidth * 0.28, eyeHeight * 0.6,
      -eyeWidth * 0.58, -eyeHeight * 0.12
    );
    ctx.closePath();
    ctx.fillStyle = "#08090c";
    ctx.fill();
    ctx.lineWidth = Math.max(1, faceWidth * 0.009);
    ctx.strokeStyle = style.features;
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

  // Characteristic moustache.
  const mouthCenter = {
    x: (mouthLeft.x + mouthRight.x) / 2,
    y: (mouthLeft.y + mouthRight.y) / 2
  };

  const moustacheY = mouthCenter.y - faceWidth * 0.015;
  const moustacheHalf = faceWidth * 0.20;

  ctx.fillStyle = style.features;

  for (const direction of [-1, 1]) {
    ctx.beginPath();
    ctx.moveTo(mouthCenter.x, moustacheY);
    ctx.bezierCurveTo(
      mouthCenter.x + direction * moustacheHalf * 0.55,
      moustacheY - faceWidth * 0.10,
      mouthCenter.x + direction * moustacheHalf * 1.05,
      moustacheY - faceWidth * 0.02,
      mouthCenter.x + direction * moustacheHalf * 1.10,
      moustacheY + faceWidth * 0.055
    );
    ctx.bezierCurveTo(
      mouthCenter.x + direction * moustacheHalf * 0.68,
      moustacheY + faceWidth * 0.035,
      mouthCenter.x + direction * moustacheHalf * 0.27,
      moustacheY + faceWidth * 0.11,
      mouthCenter.x,
      moustacheY + faceWidth * 0.075
    );
    ctx.closePath();
    ctx.fill();
  }

  // Smile detail.
  ctx.beginPath();
  ctx.moveTo(mouthLeft.x, mouthBottom.y + faceWidth * 0.025);
  ctx.quadraticCurveTo(
    mouthCenter.x,
    mouthBottom.y + faceWidth * 0.09,
    mouthRight.x,
    mouthBottom.y + faceWidth * 0.025
  );
  ctx.strokeStyle = style.stroke;
  ctx.lineWidth = Math.max(1.2, faceWidth * 0.014);
  ctx.stroke();

  // Lower accent line.
  ctx.beginPath();
  ctx.moveTo(eyeMidX - halfWidth * 0.5, lowerY - faceWidth * 0.08);
  ctx.quadraticCurveTo(
    eyeMidX, lowerY + faceWidth * 0.01,
    eyeMidX + halfWidth * 0.5, lowerY - faceWidth * 0.08
  );
  ctx.strokeStyle = style.accent;
  ctx.lineWidth = Math.max(1, faceWidth * 0.008);
  ctx.stroke();

  ctx.restore();
}

async function loadFaceLandmarker() {
  if (landmarker) return landmarker;

  setStatus(
    "Camera connected. Loading the on-device tracking model; first load requires internet."
  );

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm"
  );

  landmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      // CPU is more compatible with Android WebView devices.
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

  return landmarker;
}

function getCameraError(error) {
  const name = error?.name || "";
  const message = String(error?.message || "");

  if (name === "NotAllowedError" || name === "PermissionDeniedError") {
    return "Camera permission was denied. Allow camera access for MASKAM in Android app settings, then try again.";
  }

  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return "No compatible camera was found on this device.";
  }

  if (name === "NotReadableError" || name === "TrackStartError") {
    return "The camera is busy or unavailable. Close other camera apps and retry.";
  }

  if (name === "OverconstrainedError") {
    return "The requested camera mode is unsupported. Try again or switch cameras.";
  }

  if (name === "SecurityError") {
    return "The app or browser blocked camera access. Use the installed app or an HTTPS website.";
  }

  if (/fetch|network|wasm|model|http/i.test(message)) {
    return "The camera may be available, but the tracking model could not load. Check your internet connection and retry.";
  }

  return `MASKAM could not start: ${message || name || "unknown error"}. Check camera permission and internet access, then retry.`;
}

async function startCamera() {
  if (starting || stream) return;

  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus(
      "Camera access is unavailable in this context. Open MASKAM through the installed app or a secure HTTPS website."
    );
    return;
  }

  starting = true;
  startBtn.disabled = true;
  stopBtn.disabled = true;
  flipBtn.disabled = true;
  startBtnText.textContent = "Starting…";
  setStatus("Requesting camera access…");

  let candidateStream = null;

  try {
    // Request the camera FIRST, so model download failures do not hide
    // camera permission errors.
    candidateStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: facingMode },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 }
      }
    });

    stream = candidateStream;
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;

    await video.play();

    if (video.videoWidth === 0 || video.videoHeight === 0) {
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          cleanup();
          reject(new Error("Timed out waiting for camera video."));
        }, 10000);

        function onMetadata() {
          cleanup();
          resolve();
        }

        function cleanup() {
          clearTimeout(timeout);
          video.removeEventListener("loadedmetadata", onMetadata);
        }

        video.addEventListener("loadedmetadata", onMetadata, { once: true });
      });
    }

    if (!video.videoWidth || !video.videoHeight) {
      throw new Error("The camera opened but returned no video frames.");
    }

    stage.classList.add("has-camera");
    resizeCanvas();

    cameraLabel.textContent = "CAMERA ONLINE";
    setStatus("Camera connected. Initializing local face tracking…");

    // A camera can work even when the model fails. We now report that
    // failure accurately and cleanly release the camera.
    await loadFaceLandmarker();

    disposed = false;
    lastVideoTime = -1;
    frameCount = 0;
    lastFpsUpdate = performance.now();

    stopBtn.disabled = false;
    flipBtn.disabled = false;
    startBtnText.textContent = "Camera enabled";
    setTrackingState(true, false);
    setStatus("Face tracker ready. Move into the frame to align your mask.");

    trackFrame();
  } catch (error) {
    console.error("MASKAM camera/tracking startup error:", error);

    // Stop any stream opened during this attempt.
    if (candidateStream) {
      candidateStream.getTracks().forEach((track) => track.stop());
    }

    if (stream === candidateStream) stream = null;

    video.pause();
    video.srcObject = null;
    stage.classList.remove("has-camera");
    clearCanvas();

    cameraLabel.textContent = "CAMERA OFFLINE";
    fpsLabel.textContent = "-- FPS";
    setTrackingState(false, false);
    setStatus(getCameraError(error));
  } finally {
    starting = false;
    startBtn.disabled = false;
    startBtnText.textContent = stream ? "Camera enabled" : "Enable camera";
    stopBtn.disabled = !stream;
    flipBtn.disabled = !stream;
  }
}

function trackFrame() {
  if (!stream || disposed || !landmarker) return;

  animationId = requestAnimationFrame(trackFrame);

  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  if (video.currentTime === lastVideoTime) return;

  lastVideoTime = video.currentTime;
  const now = performance.now();

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
  } catch (error) {
    console.error("MASKAM face-tracking error:", error);
    setStatus("Face tracking encountered an error. Stop and restart the camera.");
  }

  frameCount++;

  if (now - lastFpsUpdate >= 1000) {
    const fps = Math.round(
      (frameCount * 1000) / (now - lastFpsUpdate)
    );

    fpsLabel.textContent = `${fps} FPS`;
    frameCount = 0;
    lastFpsUpdate = now;
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

  startBtn.disabled = false;
  stopBtn.disabled = true;
  flipBtn.disabled = true;
  startBtnText.textContent = "Enable camera";
  cameraLabel.textContent = "CAMERA OFFLINE";
  fpsLabel.textContent = "-- FPS";

  setTrackingState(false, false);
  setStatus("Camera is off. Nothing is being captured.");
}

async function flipCamera() {
  if (!stream || starting) return;

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

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
flipBtn.addEventListener("click", flipCamera);

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
  if (stream) resizeCanvas();
});

$("#settingsBtn").addEventListener("click", () => {
  if (typeof settingsDialog.showModal === "function") {
    settingsDialog.showModal();
  }
});

$("#closeSettingsBtn").addEventListener("click", () => {
  settingsDialog.close();
});

$("#doneSettingsBtn").addEventListener("click", () => {
  settingsDialog.close();
});

window.addEventListener("resize", () => {
  resizeCanvas();
});

window.addEventListener("pagehide", stopCamera);

video.style.transform = mirrorToggle.checked ? "scaleX(-1)" : "none";
setMaskStyle("classic");
resizeCanvas();
