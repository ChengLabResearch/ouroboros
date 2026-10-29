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

Shared arrays default to C order; `order="F"` selects Fortran storage for every
view returned by an allocation. Download destinations request Fortran order to
match CloudVolume's decoded cutouts; other allocations retain their defaults.

Serializing `SharedNPArray` transfers only its name, original shape, dtype, base
storage order, and view description. Basic indexed views retain the base order
even when the resulting view is not contiguous. Deserialization opens a
non-owning attachment; it does not
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

## Compatibility audit (2026-10-17)

The three manager interfaces remain compatible in the inspected CPython releases
below. No production-code change was needed. The audit compared the official
release sources and ran the expanded lifecycle suite on Linux x86-64 using
isolated, standard GIL-enabled CPython builds.

| CPython source and runtime | NumPy in isolated probe | Linux lifecycle result |
| --- | --- | --- |
| [3.11.17](https://github.com/python/cpython/blob/v3.11.17/Lib/multiprocessing/managers.py) | 1.26.4 | 34 passed |
| [3.12.15](https://github.com/python/cpython/blob/v3.12.15/Lib/multiprocessing/managers.py) | 1.26.4 | 34 passed |
| [3.13.16](https://github.com/python/cpython/blob/v3.13.16/Lib/multiprocessing/managers.py) | 2.5.4 | 34 passed |
| [3.14.8](https://github.com/python/cpython/blob/v3.14.8/Lib/multiprocessing/managers.py) | 2.5.4 | 34 passed |
| [3.15.0](https://github.com/python/cpython/blob/v3.15.0/Lib/multiprocessing/managers.py) | 2.5.4 | 34 passed |

Findings at each audited interface:

- **Startup/finalization:** `BaseManager.start`, `_run_server`, and
  `_finalize_manager` have unchanged parsed implementations across these releases.
  The initializer signature, address-pipe handshake, and installed shutdown
  finalizer still match the override. The parent-process sentinel remains usable
  with `fork`, `spawn`, and `forkserver`; the tests now cover initializer forwarding,
  producer exit before attachment, and owner-death cleanup under every available
  start method. CPython 3.13 moves child finalization out of `_bootstrap` into the
  launcher/exit paths; the manager finalizer and server shutdown contract remain
  intact.
- **RPC dispatch:** the request tuple, return protocol, and conversion of a
  server-raised exception to `RemoteError` are unchanged. From 3.13, `dispatch`
  deletes its result reference in a `finally` block to break a reference cycle.
  This does not change the release-result adapter. Live probes verify both an
  allocation error arriving as `RemoteError` and rejected removal preserving the
  public `FileNotFoundError` behavior.
- **Server tracker:** `SharedMemoryServer` still consumes `register_segment`,
  `destroy_segment`, `unlink`, and `segment_names` from `shared_memory_context`.
  Registration, listing, explicit release, and shutdown therefore still use the
  replacement owning registry. CPython's original tracker changes its mutable
  default list to `None` in 3.13; this does not affect our replacement.

Two adjacent changes matter for future upgrades:

- [Python 3.13 adds `SharedMemory(track=...)`](https://docs.python.org/3.13/library/multiprocessing.shared_memory.html).
  Its default remains `True`, preserving the current resource-tracker behavior
  for this multiprocessing family. Windows ignores this option. The audit does
  not change that policy or claim support for unrelated standalone attachers.
- [Python 3.14 changes the Linux default to `forkserver`](https://docs.python.org/3.14/library/multiprocessing.html#contexts-and-start-methods).
  The former spawn-only lifecycle tests would not have covered that default.
  Handoff, initializer/RPC behavior, and owner-death tests now exercise all
  available start methods explicitly. Linux probes confirmed `fork` as the
  default through 3.13 and `forkserver` in 3.14/3.15.

Each isolated probe used psutil 7.1.3, pytest 9.1.1, click 8.5.0, and
typing-extensions 4.16.0. The command, from `python/`, was
`python -m pytest test/test_shared_memory_lifecycle.py -q -W error::ResourceWarning`,
with BLAS/OpenMP thread counts set to one. All five runs had empty stderr and no
pytest warning summary. The native CI matrix retains Python 3.11/3.12 on Linux,
Windows, and macOS and runs the same expanded tests. The
[audit CI run](https://github.com/ChengLabResearch/ouroboros/actions/runs/39046361979)
passed on both supported versions: 34 tests on each Linux/macOS job, and 17 passed
plus the POSIX-only skip on each Windows job. No warnings appeared in those test
summaries. The full local supported-environment suite passed 270 tests with 99%
helper coverage.

These results audit the shared-memory component, not the full application on
new Python releases. Python 3.13–3.15 Windows/macOS runs, free-threaded builds,
the production dependency set, scientific pipelines, and RAM-reporting integration
were not validated by these probes. The project remains `>=3.11,<3.13`; raising
that cap still requires dependency and platform validation.

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
