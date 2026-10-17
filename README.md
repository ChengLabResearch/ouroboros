# Ouroboros

`or-uh-bore-us`

Quickly extract ROIs from cloud-hosted medical scans.

This repository is a monorepo containing the code for the Ouroboros app and the Ouroboros Python package (`python` folder). 

For a **usage guide**, check out the [Documentation](https://chenglabresearch.github.io/ouroboros/).

# Quick Start

**[Docker](https://www.docker.com/products/docker-desktop/) is REQUIRED to run Ouroboros.**

There are prebuilt applications available in [Releases](https://github.com/ChengLabResearch/ouroboros/releases).

- Windows: `*-setup.exe`
- Mac: `*.dmg`
- Linux: Multiple options available

When you open the app, the GUI will open immediately, but **the local server that runs the processing may take a while to start the first time** (Docker needs to pull the server image).

There is an indicator in the GUI that shows if the server is connected.

Currently, none of the apps are notarized. 

For Mac, if an error occurs when you try to run the app, find the app installation (should be called `Ouroboros.app`) and run `xattr -d com.apple.quarantine Ouroboros.app`. 

# Plugins

### Recommended Plugins

- [Neuroglancer Plugin](https://github.com/ChengLabResearch/neuroglancer-plugin)
    - Embeds Neuroglancer as a page in the app.
    - Supports loading from and saving to JSON configuration files in Ouroboros's File Explorer.
    - Has additional features like fullscreen mode and screenshots.

- Assisted Automatic Segmentation Plugin
    - Coming Soon!
    - Based on Segment Anything, it attempts to automatically segment the ROI.
        - Benefits from the assumption that the desired structure is centered in each slice.
    - Supports human input via bounding boxes and positive or negative annotations.

### Installing a Plugin

1. Open the Plugin Manager: `File > Manage Plugins`.
2. Press the Plus Icon
3. Paste the GitHub URL of the plugin.
4. Press Download (Ouroboros downloads and installs the plugin for you)

_Where are plugins installed? In the [appData](https://github.com/electron/electron/blob/main/docs/api/app.md#appgetpathname)/ouroboros folder. This folder is different on each OS._

### Creating a Plugin

See the [template README](https://github.com/ChengLabResearch/ouroboros/blob/main/plugins/plugin-template/README.md) for more information.

# Development Quick Start

Clone this repository and run the following commands from its root. Install [Node.js 24 and npm](https://nodejs.org/en/download/) (the Node version used by CI), and Docker with Compose. In WSL, run Node/npm and Docker commands in the same Linux shell; Docker Desktop must have integration enabled for that distribution.

For VS Code, the recommended extensions are [ESLint](https://marketplace.visualstudio.com/items?itemName=dbaeumer.vscode-eslint) and [Prettier](https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode).

## 1. Check Docker and close other instances

Close any installed Ouroboros app or other development instance first. They share the `ouroboros-server` container name, ports, and `ouroboros-volume` Docker volume.

```bash
docker info
```

This must succeed in the shell you will use for development. A running Docker Desktop window or an existing server container does not establish that this shell can connect. For a missing `/var/run/docker.sock` or other connection failure, see [Docker/WSL troubleshooting](documentation/docs/development/docker-builds.md#docker-desktop-is-running-but-wsl-cannot-connect).

## 2. Install dependencies and the Electron executable

```bash
npm ci
npm exec --no -- install-electron
```

The second command uses Electron's bundled binary installer. With the Electron 43.7.8 / electron-vite 5 combination tested in [#141](https://github.com/ChengLabResearch/ouroboros/issues/141), skipping it left electron-vite unable to find the executable (`Error: Electron uninstall`). See [Electron installation troubleshooting](documentation/docs/development/docker-builds.md#electron-uninstall-after-npm-ci).

## 3. Start the development app

```bash
npm run dev
```

This starts the Electron GUI, Docker backend, and host volume-transfer service. No host Python installation is required for this workflow. The first launch may need to build the backend image; wait for the GUI's server connection indicator.

The development Compose configuration mounts this checkout's `python/ouroboros` at `/app/ouroboros` and enables Uvicorn reload. Ordinary Python edits reload automatically. For dependency or Dockerfile changes, see [rebuilding the development backend](documentation/docs/development/docker-builds.md#verify-and-rebuild-the-development-backend).

## 4. Verify the GUI, backend, and output

1. [Confirm the running backend's source mount](documentation/docs/development/docker-builds.md#verify-and-rebuild-the-development-backend) points to this checkout. A connected released server does not verify your branch.
2. Use **File → Open Folder** to open the folder containing a Neuroglancer JSON input. Drag the JSON from the app's file explorer into **Neuroglancer JSON** on the slicing page. Opening a folder is required before entering a JSON path. In Linux Electron under WSL, use `/mnt/c/...` or `/mnt/wsl/...` paths; see [file selection troubleshooting](documentation/docs/development/docker-builds.md#json-paths-and-folder-selection).
3. Select the image and annotation layers, set a writable host output folder and output name, and use a short annotation path for a small slicing run. Follow the [slicing guide](documentation/docs/guide/slicing.md) for the remaining options.
4. Check that the slice/bounding-box preview visibly renders and responds to interaction. A working window or file picker alone does not verify WebGL. If it fails, collect the [graphics diagnostics](documentation/docs/development/docker-builds.md#webgl-preview-failures) and use the conditional workaround there if applicable.
5. Run slicing and confirm the TIFF output and formatted statistics JSON reach the chosen **host** folder. Server tasks report statistics by default; see [run statistics](documentation/docs/reference/run-statistics.md). Successful backend processing alone does not verify host delivery or the visual preview.

These are interactive development checks. Unit tests and headless backend tests do not exercise the Electron file picker, preview, or complete GUI-to-host workflow.

## Optional: native Python development

To work on the Python package outside Docker or run its tests, follow the [Python development instructions](python/README.md#development) and [testing instructions](python/README.md#testing). That guide covers Python 3.11 installation and pyenv selection, Poetry installation, VS Code environment setup, CLI/server commands, and shared-library prerequisites for local PyInstaller builds.
