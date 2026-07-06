# Technical Constants

This page documents hardcoded implementation limits that affect Ouroboros
behavior at runtime. Values here are extracted directly from the shipping
source. If you change them locally, update this page in the same commit.

The constants below were introduced or touched by the [#51][issue-51]
out-of-memory mitigation stack. Historical context:

- [#104][pr-104] added the file explorer watcher depth, path-count, and
  ignored-segment guardrails.
- [#105][pr-105] added the file explorer update batch size and flush
  interval.
- [#106][pr-106] stopped broadcasting the recursive file tree to plugin
  iframes on every directory change.

## File explorer

All file explorer limits live in
[`src/main/event-handlers/filesystem.ts`][filesystem-ts] at the top of the
file. They apply to the recursive watcher that backs the File Explorer
panel in the Electron app.

### Watcher depth limit

- Symbol: `FILE_EXPLORER_WATCH_DEPTH`
- Value: `6`
- Source: [`src/main/event-handlers/filesystem.ts`][filesystem-ts] line 8
- Also passed to the underlying `watcher` package as its `depth` option and
  reported inside every `folder-contents-error` payload.

**Why it exists.** The file explorer used to open folders with an
unbounded recursive watcher. Deep dependency trees, build outputs, or
scientific datasets could push watcher and renderer state well past the
V8 heap and trigger the crash described in [#51][issue-51].

**User-visible behavior when reached.** Directories nested more than six
levels below the selected root are simply not walked, so their contents do
not appear in the File Explorer panel. The panel still works normally at
shallower depths.

**When to reconsider.** Raise this if you routinely need to see files
deeper than six levels from the folder you open in Ouroboros and you have
already excluded large auxiliary trees (see the ignored segments below).
Prefer opening a lower-level folder before increasing the constant.

### Watcher path-count limit

- Symbol: `FILE_EXPLORER_WATCH_LIMIT`
- Value: `100_000`
- Source: [`src/main/event-handlers/filesystem.ts`][filesystem-ts] line 9
- Passed to the underlying `watcher` package as its `limit` option and
  reported inside every `folder-contents-error` payload.

**Why it exists.** The watcher mirrors every discovered path into
renderer state. Very large folders can therefore multiply main-process
watcher memory with renderer heap. One hundred thousand visible paths is
the ceiling picked in [#104][pr-104] as a balance between usefulness for
typical scan/output folders and keeping the renderer well under its
default 4 GB heap.

**User-visible behavior when reached.** Once the visible add count crosses
the limit, the main process emits `folder-contents-error` with a message
of the form `File explorer stopped loading after 100000 visible paths.
Choose a smaller folder or expand the folder in smaller pieces.` The
existing renderer alert system surfaces this as a warning toast. The
watcher stops delivering further add events for that session; already
loaded paths remain visible.

**When to reconsider.** Raise this only after profiling shows headroom in
both the main watcher and the renderer heap for your typical workload.
Lower it if you routinely work on constrained machines and want the
warning to appear sooner.

### Ignored path segments

- Symbol: `IGNORED_PATH_SEGMENTS`
- Value: `new Set(['node_modules', '__pycache__', 'venv'])`
- Source: [`src/main/event-handlers/filesystem.ts`][filesystem-ts] line 12

**Why it exists.** These directories are almost always noise in an
Ouroboros workflow (JavaScript dependencies, Python bytecode caches,
Python virtual environments) and are the fastest way for a project folder
to blow past the watcher path limit above.

Note that in addition to the explicit set, the watcher's `shouldIgnorePath`
helper also skips any path segment that begins with `.`, so dotfiles and
dot-prefixed directories such as `.git`, `.venv`, and `.cache` are
excluded even though they are not listed by name.

**User-visible behavior when reached.** Matching paths never appear in
the File Explorer panel and do not count toward the path-count limit. No
warning is shown, because ignoring these segments is the intended default.

**When to reconsider.** Add a segment here if you find another commonly
large directory that most users would never want to browse from
Ouroboros. Remove one only if a real workflow genuinely needs it to be
visible, since exposing these trees is the fastest way back to the
out-of-memory behavior tracked in [#51][issue-51].

### Update batch size

- Symbol: `FILE_EXPLORER_UPDATE_BATCH_LIMIT`
- Value: `500`
- Source: [`src/main/event-handlers/filesystem.ts`][filesystem-ts] line 10

**Why it exists.** [#105][pr-105] introduced batching so that a large
initial folder scan no longer produces one IPC message and one React
state update per discovered path. Five hundred events per batch amortizes
the IPC round-trip and lets the renderer apply the whole batch inside a
single `setNodes` call.

**User-visible behavior when reached.** As soon as five hundred watcher
events have accumulated in the main process, the pending queue is flushed
immediately over the `folder-contents-update-batch` IPC channel instead
of waiting for the flush interval below. The File Explorer panel then
updates once for the whole batch.

**When to reconsider.** Increase the batch size if profiling shows IPC or
React reconciliation is still the bottleneck on very large folders.
Decrease it if you see visible latency between file system changes and
the first batch appearing in the panel.

### Update flush interval

- Symbol: `FILE_EXPLORER_UPDATE_INTERVAL_MS`
- Value: `100` (milliseconds)
- Source: [`src/main/event-handlers/filesystem.ts`][filesystem-ts] line 11

**Why it exists.** The batch flush also runs on a timer so that small
bursts of changes still reach the renderer promptly, even when the
five-hundred-event threshold above is never hit.

**User-visible behavior when reached.** Any queued events are flushed at
most one hundred milliseconds after the first event in the queue was
enqueued. The panel therefore feels responsive to individual file
create/delete events while still coalescing large scans.

**When to reconsider.** Shorten the interval if the panel feels laggy
during light file activity. Lengthen it if renderer CPU during heavy
scans is a problem and you would rather trade responsiveness for
throughput.

## Plugin directory broadcast

- Broadcast payload: `{ directoryPath, directoryName, nodes: {} }`
- Source:
  [`src/renderer/src/contexts/IFrameContext.tsx`][iframe-context]
  (`broadcastDirectoryContext`)

**Why it exists.** Before [#106][pr-106], every directory context change
posted the full recursive `nodes` tree into each plugin iframe. For the
same large folders that motivated [#51][issue-51] this cloned the entire
file tree on every update and stacked on top of the watcher and renderer
costs above.

**User-visible behavior.** Plugins still receive the selected directory
path and name through the existing `send-directory-contents` message.
The `nodes` field is present for shape compatibility but is always an
empty object. Plugins that need to browse the directory tree should not
depend on this broadcast; the on-demand plugin file API tracked by
[#102][issue-102] is the intended replacement.

**When to reconsider.** Do not restore the full-tree broadcast without
first landing a memory-safe, on-demand alternative. If you need a
minimal listing for a plugin, prefer adding a targeted IPC call over
resurrecting the broadcast.

## Where the values come from

Every value above was read directly from the tree at the time this page
was written. The five file explorer constants live together as a block at
the top of [`src/main/event-handlers/filesystem.ts`][filesystem-ts]
(lines 8-12). If you edit any of them, update this page in the same
commit so operators and plugin authors can rely on it.

[issue-51]: https://github.com/ChengLabResearch/ouroboros/issues/51
[issue-102]: https://github.com/ChengLabResearch/ouroboros/issues/102
[pr-104]: https://github.com/ChengLabResearch/ouroboros/pull/104
[pr-105]: https://github.com/ChengLabResearch/ouroboros/pull/105
[pr-106]: https://github.com/ChengLabResearch/ouroboros/pull/106
[filesystem-ts]: https://github.com/ChengLabResearch/ouroboros/blob/main/src/main/event-handlers/filesystem.ts
[iframe-context]: https://github.com/ChengLabResearch/ouroboros/blob/main/src/renderer/src/contexts/IFrameContext.tsx
