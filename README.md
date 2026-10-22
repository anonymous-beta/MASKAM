
# MASKAM

Privacy-first face-tracked mask application by Anonymous-beta.

## Features

- Browser-based camera access.
- On-device face landmark detection with MediaPipe.
- Classic, Noir, and Crimson mask appearances.
- Adjustable mask opacity.
- Mirrored selfie preview.
- Front and rear camera selection where supported.
- Android packaging with Capacitor.
- Local processing of camera frames by the application.

## Requirements

- Node.js 20 or a compatible supported release.
- npm.
- A modern browser with camera support, or the installed Android application.
- An internet connection for the initial MediaPipe model and runtime download.

## Install

```bash
npm install
```

## Run locally

```bash
npm run dev
```

Open the URL printed by Vite.

Camera access normally requires HTTPS or localhost. Opening the HTML
file directly using `file://` is not a supported deployment method.

## Production web build

```bash
npm run build
npm run preview
```

The production files are written to `dist/`.

## Android

Install dependencies and build the web application:

```bash
npm install
npm run build
npx cap add android
npx cap sync android
npx cap open android
```

The Android camera permission must be present in
`android/app/src/main/AndroidManifest.xml`.

To build a debug APK from the Android project:

```bash
cd android
./gradlew assembleDebug
```

The resulting APK is normally located at:

`android/app/build/outputs/apk/debug/app-debug.apk`

Alternatively, push the project to GitHub and run the
Build MASKAM Android APK workflow. Download its APK artifact
from the completed workflow run.

## Privacy

Camera frames are processed locally by MediaPipe while tracking is active.
This application does not intentionally upload camera frames to a MASKAM
backend. The first model load downloads runtime/model assets from external
CDN and Google-hosted endpoints.

## Current limitations

- The mask is a 2D canvas overlay, not a fully rigged 3D model.
- Virtual webcam output is not implemented.
- Tracking quality depends on camera quality, lighting, device performance,
  and face visibility.
- The model requires a successful initial download unless its assets are
  separately bundled and configured.

## Troubleshooting

### Camera permission denied

Allow camera access in browser or Android app settings and retry.

### Tracking model fails to load

Check network access to the MediaPipe runtime and model URLs, then reload
the application.

### Camera already in use

Close other applications that may be using the camera.

### Android build fails

Check the GitHub Actions logs for the first failing step. A successful web
build does not guarantee a successful Android build.

## License

Add the license that applies to your project before public distribution.
