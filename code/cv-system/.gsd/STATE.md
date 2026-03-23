# GSD State

**Active Milestone:** M001 — CV Core Pipeline (No WebSocket)
**Active Slice:** S03 — Calibrator
**Active Task:** Not started
**Phase:** Execute

## Progress Summary
- **S01 (Configuration Foundation)**: ✅ Complete — Pydantic models, config loader, example file
- **S02 (Hardware Manager)**: ✅ Complete — OpenNI2 integration, frame capture, lifecycle
- **S03 (Calibrator)**: ⏳ Next — Homography + dmax_map generation
- **S04 (Coordinate Transformer)**: ⏸️ Pending — Stateless mapping service
- **S05 (Detection Layer + Integration)**: ⏸️ Pending — Touch detection + wiring

## Recent Decisions
- Calibration method: Config file only (no interactive UI)
- Touch detection: Depth-based only using ring buffer + dmax
- Architecture: 4-layer modular architecture (Hardware, Calibration, Transform, Detection)
- Stack: Python 3.12, uv, OpenNI2, OpenCV, NumPy, Pydantic

## Blockers
- None

## Next Action
**Resume with `/gsd auto`** in a fresh session to continue with S03: Calibrator (homography matrix + dmax_map generation)

## Session Handoff
Context is getting full. All S01 and S02 work is complete with summaries written. Next session should:
1. Read .gsd/STATE.md to resume from S03
2. Read M001-CONTEXT.md, M001-RESEARCH.md for context
3. Read S03-PLAN.md and start executing S03 tasks

