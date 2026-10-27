# Development Docker and WSL troubleshooting

Start with the [README's Development Quick Start](https://github.com/ChengLabResearch/ouroboros/blob/main/README.md#development-quick-start). It owns the installation, launch, and interactive verification sequence. This page covers backend verification and conditional troubleshooting; run commands from the repository root unless stated otherwise.

## Verify and rebuild the development backend

Close the installed Ouroboros app and other development instances before launching. Development and installed apps share the `ouroboros-server` container name, ports, and `ouroboros-volume` volume. Do not use a running released container as evidence that the current branch is being tested.

`npm run dev` starts both the Docker backend (port 8000) and the host volume-transfer service (port 3001). Starting Compose alone does not start that host service, which is needed to transfer inputs and deliver results.

After launching, inspect the running container:

```bash
docker inspect ouroboros-server --format '{{json .Mounts}}'
docker logs --tail 50 ouroboros-server
```

In the mounts, the source for destination `/app/ouroboros` must correspond to **this checkout's** `python/ouroboros` directory (Docker Desktop may translate the host path). The logs should show Uvicorn watching `/app/ouroboros` and completing application startup. If the mount is absent or points to another checkout, stop the app that owns that server and restart development from the intended checkout.

`python/compose.dev.yml` uses the normal Python Dockerfile for dependencies, binds the checked-out package at `/app/ouroboros`, and enables Uvicorn reload. Source edits in that package do not require an image rebuild. After dependency or Dockerfile changes, close the development app, rebuild, then relaunch using the README sequence:

```bash
docker compose -f python/compose.dev.yml build ouroboros-server
```

## Docker Desktop is running but WSL cannot connect

A missing socket error such as this means the development shell cannot reach Docker:

```text
failed to connect to the docker API at unix:///var/run/docker.sock
connect: no such file or directory
```

In Docker Desktop, enable the WSL 2 engine and **Settings → Resources → WSL Integration** for the specific distribution running development, then apply the changes. If integration settings are absent, check that Docker Desktop is using Linux containers. Retry `docker info` in the same WSL shell you use for `npm run dev`; it must return server information. Docker Desktop being open, or a container running elsewhere, does not verify this connection. See [Docker's WSL integration instructions](https://docs.docker.com/desktop/features/wsl/#enable-docker-in-a-wsl-2-distribution).

## Electron uninstall after npm ci

Electron's binary must be installed before electron-vite reads Electron's `path.txt`. An otherwise successful dependency installation can end with `Error: Electron uninstall` at launch if that file is missing.

Run the explicit `install-electron` step in the README. It uses [Electron's bundled installer](https://www.electronjs.org/docs/latest/tutorial/installation#binary-download-step); no custom downloader is needed. Check the installed executable with:

```bash
npm exec --no -- electron --version
```

This verifies executable installation, not the GUI or Docker connection.

## JSON paths and folder selection

Paths must be valid for the operating system running Electron. **Linux Electron inside WSL** needs `/mnt/c/...` or `/mnt/wsl/...`, not `C:\...` or `\\wsl$\Ubuntu\...`. Native Windows Electron uses Windows paths.

Use **File → Open Folder**, choose the containing folder, then drag the JSON from the app's file explorer into **Neuroglancer JSON**. The slicing page currently ignores JSON changes before a working folder has been selected, even when the typed absolute path is valid. In the GTK folder picker, **Ctrl+L** lets you enter an absolute folder path.

### Conditional X11 file-picker workaround

A forwarded X11 session can leave the desktop portal without a usable display. Check the portal logs for `Failed to open X11 display` and failure to associate the dialog with its parent. A nonfunctional window's title alone does not establish the cause.

For this portal/X11 mismatch, close the development app and launch with:

```bash
GDK_BACKEND=x11 npm run dev -- -- --xdg-portal-required-version=999
```

Both `--` separators are intentional: npm consumes the first, and electron-vite forwards the arguments after the second to Electron. The [portal-version switch](https://www.electronjs.org/docs/latest/api/command-line-switches#--xdg-portal-required-versionversion) selects the native GTK dialog when the portal does not meet that version requirement; `GDK_BACKEND=x11` uses the app's X11 display. This workaround applies to a portal/X11 mismatch; verify the file picker and WebGL preview separately.

## WebGL preview failures

The slicing page's React Three Fiber `Canvas` in `VisualizeSlicing.tsx` draws the 3D slice/bounding-box preview. These repeated messages indicate that its WebGL context could not be created:

```text
THREE.WebGLRenderer: A WebGL context could not be created.
GL_VENDOR = Disabled, GL_RENDERER = Disabled, Sandboxed = yes
ErrorMessage = BindToCurrentSequence failed
```

The app does not explicitly disable hardware acceleration and currently provides no preview fallback for context creation failure. A functioning file picker and successful backend processing are separate from a functioning preview.

### Capture the failing launch

- Record the launch command, Electron version, and `DISPLAY`, `WAYLAND_DISPLAY`, and `XDG_SESSION_TYPE` from its shell. Distinguish a WSLg session from forwarded X11. Check [Microsoft's WSL GUI/OpenGL prerequisites](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps) for WSLg.
- Capture the preview error from the renderer's DevTools console (**F12** in development) and GPU messages from the launch terminal. For Chromium logging and a main-process debugger, add `--inspect` before electron-vite's `--` and `--enable-logging` after it: `npm run dev -- --inspect -- --enable-logging`. Retain any display settings and other Electron switches from the failing launch. Attach a Node debugger to the inspector address printed in the terminal (electron-vite 5 defaults to port 5858).
- In that **main-process** debugger, collect Electron's [`app.getGPUFeatureStatus()`](https://www.electronjs.org/docs/latest/api/app#appgetgpufeaturestatus) after GPU initialization (`gpu-info-update`), and [`app.getGPUInfo('complete')`](https://www.electronjs.org/docs/latest/api/app#appgetgpuinfoinfotype), including any rejection. These APIs belong to `app` from `electron`, not the renderer's console. Use diagnostics from the actual failing app; an isolated probe alone does not establish its GPU state.

### Software-rendering launch

For the affected environment, close the development app and retain the working file-picker configuration while selecting the bundled SwiftShader driver:

```bash
GDK_BACKEND=x11 npm run dev -- -- \
  --xdg-portal-required-version=999 \
  --use-gl=angle \
  --use-angle=swiftshader
```

These [Chromium driver switches](https://chromium.googlesource.com/chromium/src/+/main/docs/gpu/swiftshader.md) render on the CPU and may be slower than hardware acceleration. They apply to this launch and require no source changes. Do not use `--disable-gpu` or `--no-sandbox` as the standard fix.

Verify the preview in the development app and retain any graphics warnings in the diagnostic report. Do not add `--enable-unsafe-swiftshader` merely to suppress a fallback warning.

## Other build paths

- Installed apps use the server image pinned by digest in the [release lock](https://github.com/ChengLabResearch/ouroboros/blob/main/release/release-lock.json). Docker pulls a missing image; the installed app does not build a server wheel or image.
- Local plugin installation explicitly builds Compose after copying the plugin into the app data folder, when its new backend files become available.
- The plugin template uses `plugins/plugin-template/backend/compose.dev.yml` during `npm run dev-backend`, mounts `backend/app`, and runs FastAPI development mode. Rebuild that Compose service after dependency or Dockerfile changes; ordinary source edits use reload. The packaged `backend/compose.yml` stays unchanged.
- Keep GPU-specific containers in plugins or optional Compose overrides. The core server does not require CUDA; segmentation plugins such as SAM-backed plugins are the appropriate place for CUDA runtime bases.
