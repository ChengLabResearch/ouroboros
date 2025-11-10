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

When you open the app, the GUI will open immediately, but **the local server that runs the processing could a while to start the first time** (the server needs to build). 

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

**[Docker](https://www.docker.com/products/docker-desktop/) is REQUIRED to run Ouroboros.**

Ouroboros has two main components:

- Python package (`python` folder within repository)
- Electron app (main repository)

The Python package and the Electron app have separate setup steps which are listed below. Before you begin, **it is recommended that you open the Python package (`python` folder) in a separate VSCode window whenever you are running or writing Python code** (one window for the Electron app and one window for Python).

**Before you begin, `git clone` the project locally.**

## Electron Setup

VScode Setup: [ESLint](https://marketplace.visualstudio.com/items?itemName=dbaeumer.vscode-eslint) & [Prettier](https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode)

### 1. Install Node.js and NPM

If you don't already have them installed, follow the instructions at the following link: https://nodejs.org/en/download/

### 2. Install Dependencies

```bash
$ npm install
```

### 3. Run Electron App In Development Mode

```bash
$ npm run dev
```

## Python Setup

The following steps are here to streamline the setup process for Python. For more advanced development or if you encounter any issues, you may need to follow the instructions available in the [Python README](./python/README.md).

**Easier Python Option (No Python Setup)**

If you only plan on developing the Electron app and not the Python package, there is no need to setup Python.

Running `npm run dev` in the Electron app will automatically build and start the Python server inside of Docker.

This may take up to a minute the first time, but after that, it will be almost instantaneous.

### 1. Install Python

Due to some Python dependencies, it is highly recommended that the default Python installation for the system is `3.11`.

**Easy Installation**

Download and install Python `3.11` from their website: https://www.python.org/downloads/.

**Advanced Installation**

If you use [pyenv](https://github.com/pyenv/pyenv) or a similar Python version manager, install `3.11` and set it to be the global default.

If you don't make it the global version, you'll need to use `poetry env use 3.11` later to set the Python version before running `poetry install`. 

_Example:_

```
pyenv install 3.11
pyenv global 3.11
```

I recommend following the instructions below first if you want to be able to compile the Python server using PyInstaller:

https://stackoverflow.com/questions/60917013/how-to-build-python-with-enable-framework-enable-shared-on-macos

### 2. Install Poetry and Dependencies

[Poetry](https://python-poetry.org/) is the Python project and dependency manager used for the Ouroboros Python package.

Follow the instructions on the following page to install Poetry on your system: https://python-poetry.org/docs/#installing-with-the-official-installer

**Before you continue, it is recommended that you open the Python folder (`python`) of the project in a separate window in VSCode.**

**It is also usually easier to run the setup commands in the VSCode integrated terminal, which can be opened with `CMD/CTRL + J`.**

Configure Poetry to install the virtual environment in the project folder. This allows VSCode to detect the virtual environment.

```
poetry config virtualenvs.in-project true
```

Install the project dependencies. **Make sure the terminal is in the `python` folder before running this command.**

```
poetry install
```

Now kill the current terminal and open a new one, and it should indicate that it is using the local virtual environment for Ouroboros. If not, you can run `poetry shell` to activate the shell in the current terminal.

Poetry makes it easier to run the built-in CLI or server with the following commands:

- `ouroboros-cli` - Runs the CLI in the current folder. Use `--help` to learn more.

- `ouroboros-server` - Runs the server that communicates with the Electron app.


## Running the App in Development Mode

In the Electron VSCode window, run `npm run dev` to start the Electron app. This will start the main server in the background in a Docker container.

The initial Docker build may take up to a minute, but after that it should only take a second or so.