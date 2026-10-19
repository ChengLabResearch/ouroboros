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
