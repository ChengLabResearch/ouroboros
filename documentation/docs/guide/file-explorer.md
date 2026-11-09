# File Explorer

## Large Folder Behavior

The Ouroboros File Explorer is designed to open scan and output folders
directly, but it is not a general-purpose file browser. It loads
subfolders on demand: opening a folder shows only that folder's direct
entries, and subfolder contents stream in when you expand each subfolder.
When you point it at a folder that contains many files or deeply nested
subfolders, the following behavior applies:

- Only the root's direct entries are loaded up front. Nested subfolders
  are loaded and watched only when you expand them.
- Collapsing a subfolder keeps its state and watcher alive for a short
  window (thirty seconds by default) so quickly re-expanding is free. If
  you leave it collapsed past that window, the underlying watcher is torn
  down and the subfolder's cached children are dropped from renderer
  state; re-expanding later behaves like a fresh load.
- The panel stops loading new paths after one hundred thousand visible
  entries across all currently open watchers. When this happens,
  Ouroboros shows a warning toast asking you to pick a smaller folder or
  expand the folder in smaller pieces.
- `node_modules`, `__pycache__`, `venv`, and any directory whose name
  starts with `.` are always skipped.

See [Technical Constants](../reference/technical-constants.md) for the
exact values and why they were chosen.
