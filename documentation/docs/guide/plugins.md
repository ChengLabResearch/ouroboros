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

### Preinstalled Production Plugins

Production packages can bundle plugin folders under
`extra-resources/preinstalled-plugins/<plugin-name>/`. On startup, Ouroboros
copies missing bundled plugins into the normal user-data plugin folder before it
loads plugins.

Bundled plugin installs are idempotent and version-aware. If a user already has
the same plugin installed, Ouroboros compares the `version` field in each
plugin's `package.json`; it upgrades only when the bundled version is newer and
skips existing installs that are current, newer, or not safely comparable.

### Creating a Plugin

See the [template README](https://github.com/ChengLabResearch/ouroboros/blob/main/plugins/plugin-template/README.md) for more information.

### Directory Context in Plugin Iframes

When Ouroboros posts a `send-directory-contents` message to plugin
iframes, the payload contains the selected directory's `directoryPath`
and `directoryName` but the `nodes` field is always an empty object.
This broadcast fires on every root change and is intended as a
notification only — plugins should not depend on receiving the full
recursive file tree through it.

To read the directory tree, plugins request it explicitly with the
`request-directory-contents` message described below.

### Requesting Directory Contents

Plugins ask the provider for the entries of a folder (optionally
recursive) by posting a `request-directory-contents` message. The
provider replies with a `send-directory-contents-response` message
addressed only to the requesting iframe.

**Request (plugin → provider):**

```js
window.parent.postMessage(
    {
        type: 'request-directory-contents',
        data: {
            path: '/absolute/path/under/current/root',
            recursive: true,        // optional; defaults to false
            requestId: 'abc-123'    // optional; the provider echoes it back
        }
    },
    '*'
)
```

**Response (provider → requesting iframe):**

```js
window.addEventListener('message', (event) => {
    if (event.data?.type !== 'send-directory-contents-response') return
    const { path, nodes, requestId, error } = event.data.data
    // `nodes` is a path-keyed nested map of { name, path, children? }
    // matching the shape the pre-#106 broadcast used.
    // `error` is present only when the request could not be fully served:
    //   - code: 'denied'     path outside the current root, or no root open
    //   - code: 'not-found'  path does not exist / is not a directory
    //   - code: 'limit'      enumeration hit the aggregate cap; `truncated: true`
    //                        and `nodes` holds the partial result
    //   - code: 'internal'   unexpected error; see `message`
})
```

If `requestId` is omitted from the request, the provider generates one
and echoes it back on the response so plugins can still correlate.

**Path-traversal guard.** Requests for paths outside the current root are
rejected with `code: 'denied'`. Plugins get exactly what the user can see
in the File Explorer and nothing above it.

**Truncation.** When the aggregate visible-path budget
(`FILE_EXPLORER_WATCH_LIMIT`) is exceeded mid-walk, the enumeration stops
and the response is returned with `error.code: 'limit'`,
`error.truncated: true`, and a partial `nodes` map. Callers issuing
recursive requests on very large trees should be prepared to handle
this.

**Ignore rules.** The same ignore filter the File Explorer uses (dotfiles
and the `node_modules`, `__pycache__`, `venv` segments) applies here.
Plugins cannot bypass it.

For details and the values behind other file explorer limits, see the
[Technical Constants](../reference/technical-constants.md) reference.
