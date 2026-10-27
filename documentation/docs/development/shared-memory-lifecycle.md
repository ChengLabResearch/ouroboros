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
release policy. `list_segments` derives its answer from that registry, which
implements the standard library's shared-memory tracker interface.

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

The supported Python range is `>=3.11,<3.13`. Component-level lifecycle tests
do not establish full application compatibility with a newer Python release;
raising that cap requires dependency, platform, scientific-pipeline, and
RAM-reporting validation.

## RAM-reporting integration

Memory reporting derives observations directly from `_OwnedSharedMemoryTracker`'s
`_segments`, the sole registry of live handles and release policy. Allocation,
registration and release publish a snapshot under the owner's lock: segment
names and sizes, plus the manager's mapping addresses on macOS. These temporary
manifests are one-way observations; no allocation state is read back from them.
There is no parallel attachment registry or weak-reference mapping tracker.

A reporting session starts before the scientific pipeline and is shared by
overlapping operations in the same backend. Manager startup passes its context
through the existing initializer seam, before any caller initializer. Worker
pools, `SharedNPArray` attachments, serialization and TIFF mappings have no
reporting hooks. Disabled reporting publishes no observations. Existing owner
handles govern handoff, per-volume release and owner-death cleanup. The slicing
pool starts before downloads can create shared buffers, preventing fork workers
from retaining released download allocations.

Linux subtracts `Pss_Shmem` from process PSS, then queries each live managed
segment's residency once with `mincore`. This includes resident queued buffers
whose producer has closed even if no remaining process has accessed their pages.
macOS adds the same managed residency to `libproc` private residency, excluding
the owner's segment ranges to avoid counting a sole-mapper segment twice. Its
reader and controller are tested in native CI. POSIX residency queries use a
temporary read-only mapping without faulting in the data or taking unlink
authority.

Windows adds manager-owned allocation sizes to the native private working-set
counter. This is explicitly an allocated managed-buffer term, not a claim that
all those bytes are resident. None of the platforms adds a separate resident
TIFF-cache term; ordinary mapped-file pages can still contribute through their
process counters.
See [memory measurements](../reference/run-statistics.md#memory-measurements) for
metric names, platform requirements and remaining semantic differences.

Process and allocation changes invalidate an in-progress sweep. Manifest
publication and reads share the bounded Windows sharing-violation retry used by
collector control/result files. Persistent publication failure invalidates the
report's memory coverage without changing scientific allocation behavior.
Focused tests cover observations after producer attachment closure, explicit
release, shutdown and inherited `ShareableList` registration, along with missing
or duplicated managed-buffer accounting. The native CI matrix runs ownership
and collector tests on Python 3.11/3.12.

Focused tests do not establish a pipeline overhead bound or replace validation
on the intended platform and workload.
