
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const $ = (selector) => document.querySelector(selector);

const video = $("#camera");
const canvas = $("#overlay");
const ctx = canvas.getContext("2d");
const stage = $("#stage");
const stageEmpty = $("#stageEmpty");
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
let faceFound = false;
let lastFrameTimestamp = 0;
let frameCount = 0;
let lastFpsUpdate = 0;
let fps = 0;
let disposed = false;

const MASK_STYLES = {
  classic: { fill: "#e8e6df", stroke: "#202127", features: "#202127", accent: "#bcb9b1" },
  noir: { fill: "#22252c", stroke: "#aeb3bd", features: "#e6e8eb", accent: "#555b67" },
  crimson: { fill: "#e7e3dc", stroke: "#33181d", features: "#32171c", accent: "#bd343d" }
};

function setStatus(message) {
  statusMessage.textContent = message;
}

function setTrackingState(active, found = false) {
  trackingBadge.classList.toggle("active", active && found);
  trackingBadge.innerHTML = `<span class="state-dot"></span> ${
    !active ? "STANDBY" : found ? "FACE TRACKED" : "SEARCHING"
  }`;
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

function getPoint(landmarks, index, width, height) {
  const point = landmarks[index];
  return { x: point.x * width, y: point.y * height, z: point.z || 0 };
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function drawMask(landmarks) {
  const rect = stage.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;

  // The camera video and overlay use the same cover scaling and mirror transform.
  const videoWidth = video.videoWidth;
  const videoHeight = video.videoHeight;
  if (!videoWidth || !videoHeight) return;

  const scale = Math.max(width / videoWidth, height / videoHeight);
  const drawWidth = videoWidth * scale;
  const drawHeight = videoHeight * scale;
  const offsetX = (width - drawWidth) / 2;
  const offsetY = (height - drawHeight) / 2;

  function mapped(index) {
    const p = landmarks[index];
    let x = offsetX + p.x * drawWidth;
    const y = offsetY + p.y * drawHeight;
    if (mirrorToggle.checked) x = width - x;
    return { x, y, z: p.z || 0 };
  }

  // MediaPipe Face Landmarker landmark indices.
  const forehead = mapped(10);
  const chin = mapped(152);
  const leftFace = mapped(234);
  const rightFace = mapped(454);
  const leftEye = mapped(33);
  const rightEye = mapped(263);
  const nose = mapped(1);
  const mouthLeft = mapped(61);
  const mouthRight = mapped(291);
  const mouthTop = mapped(13);
  const mouthBottom = mapped(14);

  const faceWidth = distance(leftFace, rightFace);
  if (!Number.isFinite(faceWidth) || faceWidth < 35) return;

  // Mask silhouette from cheekbones, nose bridge and chin.
  const style = MASK_STYLES[maskStyle];
  const eyeMidY = (leftEye.y + rightEye.y) / 2;
  const eyeMidX = (leftEye.x + rightEye.x) / 2;
  const eyeDistance = distance(leftEye, rightEye);
  const cheekY = eyeMidY + faceWidth * 0.19;
  const lowerY = chin.y - faceWidth * 0.02;
  const halfWidth = faceWidth * 0.53;
  const templeY = forehead.y + faceWidth * 0.28;

  ctx.save();
  ctx.globalAlpha = maskOpacity;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  // Mask shape.
  ctx.beginPath();
  ctx.moveTo(eyeMidX - halfWidth * 0.91, templeY);
  ctx.quadraticCurveTo(eyeMidX - halfWidth * 0.74, eyeMidY - faceWidth * 0.05,
                       eyeMidX - eyeDistance * 0.48, eyeMidY - faceWidth * 0.02);
  ctx.lineTo(nose.x - faceWidth * 0.055, nose.y + faceWidth * 0.08);
  ctx.quadraticCurveTo(nose.x, nose.y + faceWidth * 0.15,
                       nose.x + faceWidth * 0.055, nose.y + faceWidth * 0.08);
  ctx.lineTo(eyeMidX + eyeDistance * 0.48, eyeMidY - faceWidth * 0.02);
  ctx.quadraticCurveTo(eyeMidX + halfWidth * 0.74, eyeMidY - faceWidth * 0.05,
                       eyeMidX + halfWidth * 0.91, templeY);
  ctx.quadraticCurveTo(rightFace.x + faceWidth * 0.02, cheekY,
                       eyeMidX + halfWidth * 0.66, lowerY - faceWidth * 0.16);
  ctx.quadraticCurveTo(eyeMidX, lowerY + faceWidth * 0.08,
                       eyeMidX - halfWidth * 0.66, lowerY - faceWidth * 0.16);
  ctx.quadraticCurveTo(leftFace.x - faceWidth * 0.02, cheekY,
                       eyeMidX - halfWidth * 0.91, templeY);
  ctx.closePath();

  ctx.fillStyle = style.fill;
  ctx.fill();
  ctx.lineWidth = Math.max(1.2, faceWidth * 0.012);
  ctx.strokeStyle = style.stroke;
  ctx.stroke();

  // Soft center shading adds depth without claiming to be a 3D model.
  const gradient = ctx.createLinearGradient(eyeMidX, eyeMidY, eyeMidX, lowerY);
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
    ctx.quadraticCurveTo(-eyeWidth * 0.2, -eyeHeight * 0.75, 0, -eyeHeight * 0.2);
    ctx.quadraticCurveTo(eyeWidth * 0.28, -eyeHeight * 0.55, eyeWidth * 0.58, 0);
    ctx.quadraticCurveTo(eyeWidth * 0.22, eyeHeight * 0.7, 0, eyeHeight * 0.18);
    ctx.quadraticCurveTo(-eyeWidth * 0.28, eyeHeight * 0.6, -eyeWidth * 0.58, -eyeHeight * 0.12);
    ctx.closePath();
    ctx.fillStyle = "#08090c";
    ctx.fill();
    ctx.lineWidth = Math.max(1, faceWidth * 0.009);
    ctx.strokeStyle = style.features;
    ctx.stroke();
    ctx.restore();
  }

  const eyeAngle = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);
  drawEye(leftEye, eyeAngle);
  drawEye(rightEye, eyeAngle);

  // Nose detail.
  ctx.beginPath();
  ctx.moveTo(nose.x - faceWidth * 0.075, nose.y + faceWidth * 0.13);
  ctx.quadraticCurveTo(nose.x, nose.y + faceWidth * 0.18,
                       nose.x + faceWidth * 0.075, nose.y + faceWidth * 0.13);
  ctx.strokeStyle = style.accent;
  ctx.lineWidth = Math.max(1, faceWidth * 0.01);
  ctx.stroke();

  // Characteristic moustache: two curved halves.
  const mouthCenter = {
    x: (mouthLeft.x + mouthRight.x) / 2,
    y: (mouthLeft.y + mouthRight.y) / 2
  };
  const moustacheY = mouthCenter.y - faceWidth * 0.015;
  const moustacheHalf = faceWidth * 0.20;

  ctx.fillStyle = style.features;
  ctx.beginPath();
  ctx.moveTo(mouthCenter.x, moustacheY);
  ctx.bezierCurveTo(
    mouthCenter.x - moustacheHalf * 0.55, moustacheY - faceWidth * 0.10,
    mouthCenter.x - moustacheHalf * 1.05, moustacheY - faceWidth * 0.02,
    mouthCenter.x - moustacheHalf * 1.10, moustacheY + faceWidth * 0.055
  );
  ctx.bezierCurveTo(
    mouthCenter.x - moustacheHalf * 0.68, moustacheY + faceWidth * 0.035,
    mouthCenter.x - moustacheHalf * 0.27, moustacheY + faceWidth * 0.11,
    mouthCenter.x, moustacheY + faceWidth * 0.075
  );
  ctx.closePath();
  ctx.fill();

  ctx.beginPath();
  ctx.moveTo(mouthCenter.x, moustacheY);
  ctx.bezierCurveTo(
    mouthCenter.x + moustacheHalf * 0.55, moustacheY - faceWidth * 0.10,
    mouthCenter.x + moustacheHalf * 1.05, moustacheY - faceWidth * 0.02,
    mouthCenter.x + moustacheHalf * 1.10, moustacheY + faceWidth * 0.055
  );
  ctx.bezierCurveTo(
    mouthCenter.x + moustacheHalf * 0.68, moustacheY + faceWidth * 0.035,
    mouthCenter.x + moustacheHalf * 0.27, moustacheY + faceWidth * 0.11,
    mouthCenter.x, moustacheY + faceWidth * 0.075
  );
  ctx.closePath();
  ctx.fill();

  // Small stylized smile below the moustache.
  ctx.beginPath();
  ctx.moveTo(mouthLeft.x, mouthBottom.y + faceWidth * 0.025);
  ctx.quadraticCurveTo(mouthCenter.x, mouthBottom.y + faceWidth * 0.09,
                       mouthRight.x, mouthBottom.y + faceWidth * 0.025);
  ctx.strokeStyle = style.stroke;
  ctx.lineWidth = Math.max(1.2, faceWidth * 0.014);
  ctx.stroke();

  // A restrained accent line along the lower mask.
  ctx.beginPath();
  ctx.moveTo(eyeMidX - halfWidth * 0.5, lowerY - faceWidth * 0.08);
  ctx.quadraticCurveTo(eyeMidX, lowerY + faceWidth * 0.01,
                       eyeMidX + halfWidth * 0.5, lowerY - faceWidth * 0.08);
  ctx.strokeStyle = style.accent;
  ctx.lineWidth = Math.max(1, faceWidth * 0.008);
  ctx.stroke();

  ctx.restore();
}

async function loadFaceLandmarker() {
  if (landmarker) return landmarker;

  setStatus("Loading the on-device face-tracking model. The first load needs an internet connection.");

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm"
  );

  landmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU"
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

async function startCamera() {
  if (stream) return;

  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("Camera access requires HTTPS or a supported secure app context. Open the deployed site over HTTPS and try again.");
    return;
  }

  startBtn.disabled = true;
  startBtnText.textContent = "Starting…";

  try {
    const tracker = await loadFaceLandmarker();

    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: facingMode },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30, max: 30 }
      }
    });

    video.srcObject = stream;
    await video.play();

    await new Promise((resolve, reject) => {
      if (video.videoWidth > 0) return resolve();
      const timer = setTimeout(() => reject(new Error("Camera startup timed out.")), 10000);
      video.addEventListener("loadedmetadata", () => {
        clearTimeout(timer);
        resolve();
      }, { once: true });
    });

    stage.classList.add("has-camera");
    resizeCanvas();
    stopBtn.disabled = false;
    flipBtn.disabled = false;
    startBtnText.textContent = "Camera enabled";
    cameraLabel.textContent = "CAMERA ONLINE";
    setTrackingState(true, false);
    setStatus("Camera is active. Face tracking runs locally on this device.");
    disposed = false;
    lastVideoTime = -1;
    lastFrameTimestamp = 0;
    frameCount = 0;
    lastFpsUpdate = performance.now();
    trackFrame();
  } catch (error) {
    stopCamera();
    const message = error?.name === "NotAllowedError"
      ? "Camera permission was denied. Allow camera access in your browser or app settings and try again."
      : error?.name === "NotFoundError"
        ? "No compatible camera was found on this device."
        : `Unable to start the camera: ${error?.message || "unknown error"}`;
    setStatus(message);
    console.error("MASKAM camera error:", error);
  } finally {
    startBtn.disabled = false;
    startBtnText.textContent = stream ? "Camera enabled" : "Enable camera";
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
    faceFound = Boolean(landmarks);

    if (landmarks) {
      drawMask(landmarks);
      setTrackingState(true, true);
      setStatus("Face detected. The mask is following your face.");
    } else {
      setTrackingState(true, false);
      setStatus("Looking for a face. Move into the camera frame and face the camera.");
    }
  } catch (error) {
    console.error("MASKAM tracking error:", error);
    setStatus("Face tracking encountered an error. Stop and restart the camera to try again.");
  }

  frameCount++;
  if (now - lastFpsUpdate >= 1000) {
    fps = Math.round(frameCount * 1000 / (now - lastFpsUpdate));
    fpsLabel.textContent = `${fps} FPS`;
    frameCount = 0;
    lastFpsUpdate = now;
  }
}

function stopCamera() {
  disposed = true;
  if (animationId !== null) cancelAnimationFrame(animationId);
  animationId = null;

  if (stream) {
    for (const track of stream.getTracks()) track.stop();
    stream = null;
  }

  video.pause();
  video.srcObject = null;
  stage.classList.remove("has-camera");
  clearCanvas();
  faceFound = false;
  lastVideoTime = -1;

  stopBtn.disabled = true;
  flipBtn.disabled = true;
  startBtn.disabled = false;
  startBtnText.textContent = "Enable camera";
  cameraLabel.textContent = "CAMERA OFFLINE";
  fpsLabel.textContent = "-- FPS";
  setTrackingState(false, false);
  setStatus("Camera is off. Nothing is being captured.");
}

async function flipCamera() {
  if (!stream) return;
  const nextFacing = facingMode === "user" ? "environment" : "user";
  stopCamera();
  facingMode = nextFacing;
  await startCamera();
}

function setMaskStyle(style) {
  if (!MASK_STYLES[style]) return;
  maskStyle = style;
  document.querySelectorAll(".mask-option").forEach((button) => {
    button.classList.toggle("selected", button.dataset.mask === style);
    button.setAttribute("aria-pressed", String(button.dataset.mask === style));
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
});

$("#settingsBtn").addEventListener("click", () => {
  if (typeof settingsDialog.showModal === "function") settingsDialog.showModal();
});
$("#closeSettingsBtn").addEventListener("click", () => settingsDialog.close());
$("#doneSettingsBtn").addEventListener("click", () => settingsDialog.close());

window.addEventListener("resize", () => {
  if (stream) resizeCanvas();
});
window.addEventListener("pagehide", stopCamera);

video.style.transform = mirrorToggle.checked ? "scaleX(-1)" : "none";
setMaskStyle("classic");
resizeCanvas();
