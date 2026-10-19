# Status presentation

This document is the presentation reference for slicing, backprojection, and the
autoseg plugin. It defines display semantics, not backend states or readiness rules.
Each repository maps its existing observations at render time. Do not store the
mapped values as another execution or connection authority.

| Display status | Meaning | Appearance |
| --- | --- | --- |
| Waiting | A step is pending or an expected observation is awaited. | Neutral |
| Running | The source reports that work is active. | Blue |
| Complete | The source reports successful completion of this step. | Green |
| Warning | A reported condition needs attention; it does not establish task failure. | Amber |
| Failed | The source explicitly reports failure of the named operation. | Red |
| Interrupted | The source explicitly reports interrupted execution. | Red |
| Unknown | The available observation does not establish the state or outcome. | Neutral |

Use **Complete**, not Succeeded or Completed, on state labels; use **Failed**, not
Error, for an operation's state. Sentences can say “Slicing completed.” Notice
severity (`info`, `success`, `warning`, `error`) describes a message, not execution
state. Contextual labels such as Connected or Loads on first run are appropriate
when they describe the actual observation more precisely than a generic state.

Main-app step values `pending/running/succeeded/failed/interrupted` map to
`Waiting/Running/Complete/Failed/Interrupted`. Autoseg's
`pending/in_progress/completed/error/warning` normally map to
`Waiting/Running/Complete/Failed/Warning`. These are display adapters; wire values
stay unchanged. A known source-specific meaning takes precedence over blindly
translating a string: autoseg's lazy ML-runtime `warning` means the model is not
loaded yet, so it uses neutral **Loads on first run**, not a startup failure.

The backend summary and checklist show separate observed facts. In the main app,
launch-command completion does not establish backend readiness. A reachable
backend plus a failed local launch displays **Connected — local launch failed**,
with **Failed** on the launch row and **Connected** on the connection row. With no
local launch observation, omit that row. Do not create additional probes to fill
out a checklist. Autoseg's backend and browser volume observations must be named
separately, since they can disagree.

Task percentages describe their reported stages. They do not establish the overall
outcome. Show the reported terminal result independently of notice dismissal.
Communication trouble or a missing job does not establish Failed; preserve existing
last-reported information and identify an unknown outcome where already observed.
Keep workflow-specific stages rather than inventing a universal lifecycle.

The main app's `STATUS_LABELS` and `StatusLabel` own its display vocabulary and
state appearance. Plugin views follow this document through their local display
adapter. A shared cross-repository runtime package is not required.

## Notices

Use compact notices beside the information they explain: launch diagnostics in the
initialization section; submission, task, and communication diagnostics in the task
section. Keep the reported text and failure-stage details, including statistics-report
messages. Long paths and messages wrap, and explicit line breaks remain readable.

Severity uses a subdued background and border with these accents: info `#a9ceff`,
success `#a7ddb5`, warning `#f3d18b`, error `#ffb3a8`. Errors announce with `role="alert"`;
other notices use `role="status"`. A request diagnostic has a visible, keyboard-accessible
**Dismiss** button. Reported outcomes, steps, launch observations, and communication
warnings remain source-derived and do not disappear when that diagnostic is dismissed.

The main app moves only task-start request failures (`/slice/`, `/backproject/`) and
previous-task deletion failures into these retained notices. Local form/input errors
(including invalid options and Neuroglancer JSON), filesystem warnings, and rendering
errors keep their existing global alerts. A moved failure is not also shown as a toast.

`ServerContext` remains the owner of request errors; there is no copied notice history
or page-local dismissal flag. Dismiss clears only that request's error, preserving
results, pending state, and task/stream data. A new submission clears its earlier start
and deletion notices before preparing the request. Successful requests retain their
existing error-clearing behavior. Input editing or an unrelated successful request does
not clear a retained diagnostic. Unlike autoseg's error list, these notices follow the
existing request lifecycle rather than accumulating past failures.

Undismissed errors retain the existing provider lifetime and can still be seen on
returning to a page. A dismissed error cannot reappear on navigation because it was
cleared at its owner. No navigation/reload storage is added. Each deletion request is
identified by its actual task-specific URL (`/delete/?task_id=…`), so one workflow cannot
display another workflow's deletion failure. The saved task response remains the source
of that task ID until the existing deletion/submission lifecycle replaces it.
