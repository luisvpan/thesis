# PROJECT GOALS — CV System

## Mission Statement

Implement the **computer vision system** that serves as the bridge between the physical world (work table with tangible blocks) and the digital world (Language Runtime). The system calibrates the relationship between a Kinect V2 and a projector, detects interactions on the work surface, and communicates events to the Language Runtime for processing and feedback projection.

```
Physical table    →  [CV System]  →  Language Runtime  →  Projector
(blocks, hands)      ^^^^^^^^^^^^^^     (evaluates program)  (AR feedback)
                     This is what
                     we build here
```

## Scope

### In Scope

- Kinect V2 integration (depth and RGB streams) via OpenNI2
- Per-session calibration: 4-point homography + dmax_map generation
- Camera ↔ projector coordinate transformation
- Touch/interaction detection on the work surface
- Tangible language piece detection (future, YOLO)
- Event communication to the Language Runtime via WebSocket
- Externalized configuration (no magic numbers)

### Out of Scope

- Language Runtime (compiler, runtime, program evaluation)
- Physical piece design for the tangible language
- AR feedback rendering (responsibility of the Language Runtime + projector)
- Development or administration UI/IDE

## Context

This system is part of a thesis developing a tangible programming language for children aged 6-9. Children manipulate physical blocks on a table. A depth camera (Kinect V2) detects the blocks and interactions, and a projector overlays visual feedback (augmented reality) on the table.

The existing code is inherited from a previous thesis (originally for Kinect V1) and was adapted through trial and error. The current file (`../computer-vision-manager/ultimate_calibrate_area.py`, ~500 lines) is monolithic, fragile, and full of magic numbers. This refactor aims to turn it into a maintainable and extensible system.

## Key Decisions

Design decisions are documented as ADRs in `docs/adr/`:

| ADR | Decision |
|-----|----------|
| ADR-001 | Python as the system language |
| ADR-002 | uv as the package manager |
| ADR-003 | 4-point homography for calibration |
| ADR-004 | 4-layer modular architecture |
| ADR-005 | dmax_map as in-memory per-session reference |
| ADR-006 | AsyncAPI for WebSocket protocol documentation |
| ADR-007 | Makefile for startup coordination |
| ADR-008 | Ruff as linter and formatter |

## Architecture Overview

The system is organized in 4 layers (see ADR-004 and `docs/architecture/cv-subsystem-architecture.likec4`):

1. **Hardware Manager** — Kinect V2 integration (OpenNI2). Single point of contact with the hardware.
2. **Calibrator** — Per-session calibration process. Produces homography + dmax_map.
3. **Coordinate Transformer** — Stateless camera ↔ projector mapping service.
4. **Detection Layer** — Touch detection (ring buffer + dmax) and piece detection (YOLO, future).

Plus a **WebSocket Bridge** for communication with the Language Runtime.

## Technical Constraints

- **Hardware:** Kinect V2 (depth 512x424, RGB 1920x1080, 30fps), projector (1920x1080)
- **OS:** Windows recommended for Kinect V2 / OpenNI2 driver compatibility
- **Latency:** Detection must operate at the Kinect's frame rate (30fps) for responsive interaction with children
- **Concurrent users:** 1 table per system instance
- **Recalibration:** Once at the start of each session (~17s for 500 dmax_map frames)

## Open Questions

1. **Marker detection:** Kinect RGB or external camera? (Preliminary decision: Kinect RGB, see ADR-003)
2. **YOLO for pieces:** Which model? Custom training or transfer learning? Detection frequency?
3. **WebSocket protocol:** Concrete message and schema definitions (document with AsyncAPI, see ADR-006)
4. **Performance targets:** Maximum acceptable latency from detection to projected visual feedback?

## References

- Hartley, R. & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision*, 2nd ed. Cambridge University Press.
- Torralba, A., Isola, P. & Freeman, W.T. (2024). *Foundations of Computer Vision*. MIT Press. Ch. 41: Homographies.
- Guo, Y. et al. (2018). "A real-time interactive system of surface reconstruction and dynamic projection mapping with RGB-depth sensor and projector." *IJDSN*, 14(7).
- OpenCV Homography Tutorial: https://docs.opencv.org/4.13.0/d9/dab/tutorial_homography.html
- AsyncAPI Specification: https://www.asyncapi.com/docs/reference/specification/v3.1.0

---

**Document Status:** Living document
**Last Updated:** 2026-03-21
