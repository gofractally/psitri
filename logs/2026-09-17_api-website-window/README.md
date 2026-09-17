# Website measurement window

The clean `5ba8186` benchmark binaries were queued for an idle cloud runner.
The psizam benchmark controller remained active, so the idle wait was interrupted
without launching a benchmark. No timing samples were produced. A coordinated
pause between the other workload's stages, or its completion, is required before
collecting the paired series. Existing correctness checks and isolation thresholds
remain intact.
