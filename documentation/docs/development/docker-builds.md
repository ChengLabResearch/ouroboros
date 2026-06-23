# Docker Build Readiness

This note captures the current build hot spots and the next cleanup targets for keeping Docker startup fast.

## Current behavior

- Development startup intentionally runs the main server compose file with `--build` so Python source changes are picked up while working locally.
- Production startup now uses plain compose startup. Docker will build the bundled server image the first time it is missing, but repeated app launches should not force a rebuild.
- Local plugin installation still runs an explicit compose build after copying a plugin into the app data folder. That is the correct time to build a packaged plugin backend because the app has just installed new backend files.

## Remaining follow-ups

- Publish the production server image to a registry during releases, then ship a compose file that references the immutable image tag instead of building from the bundled wheel at first launch.
- Add a separate development compose path with bind-mounted Python source and reload-friendly commands so main-server development does not need a full image rebuild for ordinary source edits.
- Add a plugin-template backend development compose file that bind-mounts `backend/app` and avoids `--build` on every `npm run dev` start once the image exists.
- Keep GPU-specific containers in plugins or optional compose overrides rather than making the core app image GPU-first. The core server image does not currently require CUDA, while segmentation plugins such as SAM-backed plugins are the better place for CUDA runtime bases.
- Preserve the current wheel-based release path until registry-published server images exist, because it keeps releases reproducible without depending on a mutable external image name.
