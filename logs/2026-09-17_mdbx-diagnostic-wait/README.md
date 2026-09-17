# Diagnostic held by the idle gate

The GDB reproduction command in `command.json` was not launched. The runner
waited 180 seconds for ten consecutive seconds below 3% aggregate CPU activity
and exited with `Machine did not become idle; no benchmark started`.
`idle.jsonl` retains every sample. This attempt has no timing or diagnostic result.
