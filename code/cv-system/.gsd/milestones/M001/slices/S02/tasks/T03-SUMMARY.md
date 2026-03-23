---
id: T03
parent: S02
milestone: M001
provides:
  - shutdown() method for proper resource cleanup
  - Context manager protocol (__enter__, __exit__)
  - _cleanup() helper for internal cleanup
  - Re-initialization prevention via _initialized flag
requires:
  - task: T01
    provides: HardwareManager with OpenNI2 state
  - task: T02
    provides: Frame capture methods that depend on initialized state
affects: [Main orchestration in S05]
key_files:
  - src/cv_system/hardware/manager.py (added shutdown() and context manager)
key_decisions:
  - Context manager propagates exceptions (returns False)
  - _cleanup() tries all cleanup operations even if some fail
  - Re-initialization raises HardwareError with clear message
patterns_established:
  - Always clean up resources, even on exception paths
  - Use context manager for reliable resource management
drill_down_paths:
  - .gsd/milestones/M001/slices/S02/tasks/T03-PLAN.md
duration: Implemented with T01
verification_result: pass
completed_at: 2026-03-22T23:55:00Z
---

# T03: Lifecycle management and error handling

**Shutdown and context manager for reliable resource cleanup.**

## What Happened

Implemented lifecycle management in HardwareManager:

1. **shutdown()**: Public method to stop streams, close device, unload OpenNI2 context. Sets _initialized=False.
2. **_cleanup()**: Internal helper that tries all cleanup operations even if some fail (bare except for each operation)
3. **__enter__()**: Returns self for with statement usage
4. **__exit__()**: Calls shutdown() and returns False to propagate exceptions

The _initialized flag prevents re-initialization without shutdown. All cleanup operations are wrapped in try/except to ensure partial cleanup doesn't prevent full cleanup.

## Deviations
None — integrated into HardwareManager as designed. Methods follow T03-PLAN.md exactly.

## Files Created/Modified
- `src/cv_system/hardware/manager.py` — Added shutdown(), _cleanup(), __enter__(), __exit__() (30 lines)
