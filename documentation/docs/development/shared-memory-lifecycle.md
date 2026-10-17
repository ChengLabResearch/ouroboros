# Shared-memory ownership

`python/ouroboros/helpers/mem.py` owns shared NumPy allocation, attachment,
serialization, and release. Its lifecycle handling adapts
[`lftomo/util/mem.py` at 8d4c884](https://github.com/den-sq/lftomo/blob/8d4c884e9f42ad913a38f6592f85ba9419f7a0f1/lftomo/util/mem.py):
non-owning serialization, cleanup guards, and manager cleanup after owner death.
Ouroboros does not import lftomo's role-based reuse planner or storage interfaces;
there are no consumers of those facilities here.

## Allocation and handoff

The manager server's `shared_memory_context` is the sole registry of managed
allocations. Each entry contains a live `SharedMemory` handle and the allocation's
release policy. The prior global termed-name list and its proxy are removed.
The standard library's names-only tracker is replaced, not maintained alongside
the owning registry. `list_segments` derives its answer from that registry.

`SharedNPArray()` and `TermedNPArray()` allocate through the same server RPC.
The server establishes ownership before returning a name. The client opens a
local attachment and constructs array views; failed client attachment or view
construction releases the server allocation. Inherited `ShareableList()`
registration also opens an owning server handle before acknowledging the client.

Serializing `SharedNPArray` transfers only its name, original shape, dtype, and
view description. Deserialization opens a non-owning attachment; it does not
transfer unlink authority or cached NumPy data. The server keeps the buffer alive
even if the producer exits before the consumer opens it. This matters on Windows,
where the allocation disappears when its last handle closes.

## Release and failure handling

- Run-scoped arrays and standard manager allocations live until manager shutdown.
- Termed arrays are released by `remove_termed()` at the existing per-volume
  completion point in `VolumeCache.remove_volume()`. They do not accumulate until
  shutdown. Validation and owner release are atomic at the server; an unknown or
  run-scoped name is rejected without closing the supplied array.
- On POSIX, release unlinks the name and closes the manager's handle. Already-open
  attachments remain valid until they close; new attachments cannot open the name.
- On Windows, release closes the manager's handle. Existing attachments keep the
  allocation alive until they close. The supplied array, when present, is closed
  after the server acknowledges release.
- Closing a deserialized attachment never unlinks the creator's allocation.
  Explicit array cleanup is idempotent and handles partial construction.

Normal manager shutdown uses `SharedMemoryServer.shutdown()` and that same
owning registry. A client-side `shutdown()` override would be replaced by
`BaseManager.start()`'s finalizer and is not used for allocation cleanup.

The manager detaches from the owner's POSIX process group, as in lftomo, so it
can clean up after that group is killed. A watcher waits on Python's parent-process
sentinel on all supported platforms; owner death releases tracked allocations
and exits the manager. POSIX `SIGTERM` to the manager also triggers cleanup.
Forcibly killing the manager itself cannot execute its cleanup code; Windows
closes its handles, while POSIX retains its standard resource-tracker fallback.

The server tracker and startup hooks depend on CPython manager internals. Native
CI exercises Python 3.11 and 3.12 on Linux, Windows, and macOS. Tests prove producer
exit before receiver attachment, per-buffer release, surviving mapped consumers,
serialization authority, failed construction, normal/exception shutdown, and
owner termination. Multiprocessing coverage includes the live server code.

Before raising the Python upper bound in `python/pyproject.toml`, re-check these
three CPython interfaces against every newly supported version:

- `BaseManager._run_server` and startup/finalization: the mirrored signature,
  parent-process sentinel, and replacement of the client `shutdown()` method.
- `multiprocessing.managers.dispatch`: request signatures and the server's
  exception-to-`RemoteError` conversion used by the release-result adapter.
- `SharedMemoryServer.shared_memory_context`: the tracker interface consumed by
  allocation registration, listing, release, and server shutdown.

Extend the native lifecycle matrix to those versions and verify handoff, early
release, and owner-death cleanup on all three operating systems before changing
the supported-version contract.

## RAM-reporting integration

The deferred integration and real-data acceptance from #145 are tracked in
[#139](https://github.com/ChengLabResearch/ouroboros/issues/139), to be completed
after rebasing PR #144 onto this ownership change. Its existing mapping observer
must observe server-owned handles at allocation/registration and refresh after
release, as well as observing each
process-local attachment. The owning registry must remain the sole allocation
authority; observation must not become another owner or registry of lifetimes.
The new serialization path runs the attachment constructor, so observation belongs
there rather than in a redundant `__setstate__` path.

Keeping a server handle mapped does not by itself fix the Linux PSS coverage gap:
resident segments queued after the producer closes can remain untouched by every
currently mapped process. #139 must validate accounting for that state using the
owning registry as the source of live segment identities and sizes. A proposed
USS-plus-live-segment metric needs its own defined semantics and tests against
double-counting; allocated capacity must not silently be labeled residency.

Real-data slicing/backprojection and RAM overhead validation follow that rebase.
Focused lifecycle tests on this branch do not establish scientific-pipeline or
RAM-collector acceptance. The separate manifest-publication defect remains in
scope for the RAM-reporting branch.
