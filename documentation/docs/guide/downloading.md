# Download Ouroboros

[Docker](https://www.docker.com/) **is required** to run Ouroboros. Make sure to install and run it before running Ouroboros.

Prebuilt applications are available in [Releases](https://github.com/ChengLabResearch/ouroboros/releases).

- Windows: `*-setup.exe`
- Mac: `*.dmg`
- Linux: Multiple options available

When you open the app, the GUI will open immediately, but **the local server that runs the processing could take up a minute to start** the first time it is run. There is an indicator in the GUI that shows if the server is connected.

Currently, none of the apps are notarized. 

For Mac, if an error occurs when you try to run the app, find the app installation (should be called `Ouroboros.app`) and run `xattr -d com.apple.quarantine Ouroboros.app`. 