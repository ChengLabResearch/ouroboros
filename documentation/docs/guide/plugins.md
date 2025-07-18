# Ouroboros Plugins

### Recommended Plugins

- [Neuroglancer Plugin](https://github.com/We-Gold/neuroglancer-plugin)
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