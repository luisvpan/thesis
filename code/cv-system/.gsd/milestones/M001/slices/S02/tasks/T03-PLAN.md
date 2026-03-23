# T03: Lifecycle management and error handling

**Slice:** S02
**Milestone:** M001

## Goal
Implement initialize(), shutdown(), and context manager protocol with robust error handling.

## Must-Haves

### Truths
- initialize() sets up OpenNI2 context and streams
- shutdown() properly stops streams and releases resources
- Context manager protocol (__enter__, __exit__) works
- Exceptions are raised with clear messages for failures
- Resources are cleaned up even if exceptions occur

### Artifacts
- `src/cv_system/hardware/manager.py` — shutdown() method and __enter__/__exit__
- Proper resource cleanup in all exit paths
- Custom HardwareError exception class

### Key Links
- T01 → initialize() uses that code
- T02 → frame methods depend on initialized streams
- Main → uses context manager protocol

## Steps
1. Create HardwareError custom exception in manager.py
2. Implement shutdown(self) method:
   - Stop streams if they exist
   - Close device if it exists
   - Shutdown OpenNI2 context if it exists
   - Set all instance vars to None to prevent reuse
3. Implement __enter__(self) -> HardwareManager:
   - Return self for with statement usage
4. Implement __exit__(self, exc_type, exc_val, exc_tb):
   - Call shutdown()
   - Return False to propagate exceptions (or handle specific ones)
5. Update initialize() to check if already initialized (raise error)
6. Add try/except blocks with HardwareError wrapping
7. Add docstrings for all lifecycle methods
8. Run `uv run ruff check src/` to verify linting

## Context
- Ensure streams are stopped before closing device
- Always clean up even on exceptions
- Prevent re-initialization without shutdown (state check)
- Context manager is Pythonic and ensures cleanup
