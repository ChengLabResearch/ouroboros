// Cross-surface constants shared between the Electron main process and the
// renderer. Keep this file dependency-free so both bundles can import it
// without pulling in Node- or DOM-specific modules.

/**
 * How long to wait after the renderer collapses a folder before the main
 * process closes the watcher on that folder and the renderer prunes the
 * matching subtree from state. If the user re-expands the folder before this
 * timer elapses, both teardowns are cancelled and existing state is kept.
 *
 * See documentation/docs/reference/technical-constants.md for the rationale.
 */
export const COLLAPSE_TEARDOWN_DELAY_MS = 30_000
