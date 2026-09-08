# ADR-001: Python as the language for the computer vision system

## Status

Accepted

## Date

2026-03-19

## Context

The computer vision (CV) system requires a programming language that supports integration with the Kinect V2, image processing, and eventually object detection with YOLO. The codebase inherited from the previous thesis is written in Python. The rest of the project (Language Runtime) uses TypeScript with Bun/Elysia.

Three options were evaluated:

**Python:** Mature ecosystem for CV (OpenCV, NumPy, ultralytics/YOLO). Bindings available for OpenNI2. The team has experience with the language. Expensive OpenCV operations execute in C++ under the hood, mitigating Python's performance limitations. The downside is that it remains a separate process from the Bun/Elysia server.

**C#:** The Microsoft Kinect SDK V2 was designed for C#, providing direct access to native APIs without OpenNI2. Better performance than Python for direct numerical processing. However, the team has no experience with C#, the learning curve is significant, and integration with the Bun/Elysia stack would have the same separate-process issue as Python. OpenCV is available via Emgu CV but with less abundant documentation.

**TypeScript:** Coherent with the Language Runtime stack, potentially allowing a single process. However, `opencv4nodejs` has maintenance issues, `opencv-js` (WASM) only exposes a limited subset of OpenCV, and no OpenNI2 bindings exist for Node/Bun. Integrating the CV stack would require FFI or native wrappers, introducing complexity and fragility.

## Decision

We use **Python** for the CV system.

## Consequences

Communication between the CV system (Python) and the Language Runtime (Bun/Elysia) occurs via WebSocket. This means they are two separate processes that must coordinate at startup. The separation is architecturally sound: if the vision module crashes, the server stays up, and vice versa. `uv` is used as the package and environment manager (see ADR-002).
