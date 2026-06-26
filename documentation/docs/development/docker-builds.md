# Docker Build Readiness

This note captures the current build hot spots and the next cleanup targets for keeping Docker startup fast.

## Current behavior

- Development startup uses `python/compose.dev.yml`, bind-mounts the Python source tree, and runs the Docker server through Uvicorn reload. Ordinary Python source edits should not require a compose rebuild.
- Production startup now uses plain compose startup. Docker will build the bundled server image the first time it is missing, but repeated app launches should not force a rebuild.
- Local plugin installation still runs an explicit compose build after copying a plugin into the app data folder. That is the correct time to build a packaged plugin backend because the app has just installed new backend files.
- The plugin template uses `backend/compose.dev.yml` during `npm run dev-backend`, bind-mounts `backend/app`, and runs the backend in FastAPI development mode.

## Development compose files

- `python/compose.dev.yml` is for main-server development. It still uses the normal Python Dockerfile for dependencies, but mounts `python/ouroboros` into the container and reloads the server when that package changes.
- `plugins/plugin-template/backend/compose.dev.yml` is for template plugin backend development. It keeps the packaged `backend/compose.yml` unchanged while mounting `backend/app` for reload-oriented local edits.
- Run `docker compose -f <compose.dev.yml> build` after dependency or Dockerfile changes. Routine source edits can use plain compose startup.

## Remaining follow-ups

- Publish the production server image to a registry during releases, then ship a compose file that references the immutable image tag instead of building from the bundled wheel at first launch.
- Keep GPU-specific containers in plugins or optional compose overrides rather than making the core app image GPU-first. The core server image does not currently require CUDA, while segmentation plugins such as SAM-backed plugins are the better place for CUDA runtime bases.
- Preserve the current wheel-based release path until registry-published server images exist, because it keeps releases reproducible without depending on a mutable external image name.
