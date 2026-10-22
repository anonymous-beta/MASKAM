
# MASKAM

**A privacy-first, face-tracked virtual mask.**

Created by **Anonymous-beta (Chinedu)**.

MASKAM is an experimental camera application with a classic Anonymous-inspired mask overlay, real-time face landmark tracking, and a mobile-friendly interface.

## Features

- Real-time face landmark tracking using MediaPipe Tasks Vision.
- Classic, Noir, and Crimson mask appearance options.
- Front/rear camera switching where supported.
- Mirror-preview toggle and mask-opacity control.
- Local camera processing; the application does not send camera frames to a MASKAM backend.
- Responsive web interface.
- Android debug APK build through GitHub Actions.

## Requirements

- Node.js 20 or newer.
- npm.
- A modern browser with camera support for web testing.
- HTTPS for browser camera access, except supported local development contexts.

The first face-tracking model load requires an internet connection. MediaPipe's model and WASM assets are downloaded from their configured CDN URLs.

## Run locally

```bash
npm install
npm run dev
```

Open the local development URL printed by Vite.

To create a production build:

```bash
npm run build
```

## Android APK

1. Push the project to the `main` branch.
2. Open the repository's **Actions** tab.
3. Select **Build MASKAM Android APK**.
4. Run the workflow or wait for the push-triggered workflow.
5. Open the completed workflow run.
6. Download the `MASKAM-debug-apk` artifact.
7. Extract the artifact ZIP to obtain `app-debug.apk`.

The APK produced by this workflow is a **debug APK**, not a Play Store release.

## Permissions and privacy

Camera access is requested only when the user taps Enable camera. The app does not request microphone access and does not intentionally upload camera frames. Face tracking is performed locally using MediaPipe.

The face-tracking model and WebAssembly runtime are currently loaded from external URLs. Therefore, the first model load requires network access. For a more strictly offline deployment, bundle and verify those assets locally before release.

## Current limitations

- The mask is a 2D canvas overlay positioned from face landmarks; it is not a full 3D face mesh or physically accurate virtual mask.
- Virtual webcam output is not implemented.
- Tracking quality depends on lighting, camera quality, pose, and device performance.
- This is an early-access project and has not been validated across every Android device.

## Project structure

```text
MASKAM/
├── assets/
│   └── maskam-mark.svg
├── .github/
│   └── workflows/
│       └── android-apk.yml
├── app.js
├── index.html
├── styles.css
├── package.json
├── vite.config.js
├── capacitor.config.json
├── manifest.webmanifest
└── README.md
```

## Attribution

MASKAM — Created by Anonymous-beta (Chinedu).
