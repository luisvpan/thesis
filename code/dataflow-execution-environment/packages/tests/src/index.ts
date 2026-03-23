// Test entry point - exports all integration test files

// End-to-End Pipeline Tests
export * from "./end-to-end/arithmetic.test.js";
export * from "./end-to-end/types.test.js";

// Type Validation Tests
export * from "./types/validation.test.js";
export * from "./types/properties.test.js";

// Curriculum Type Tests
export * from "./curriculum/shapes.test.js";
export * from "./curriculum/comparison.test.js";

// Set Operation Tests
export * from "./sets/union.test.js";
export * from "./sets/filter-sort.test.js";

// Temporal Operation Tests
export * from "./temporal/fby.test.js";
export * from "./temporal/streams.test.js";

// Error Handling Tests
export * from "./errors/division-zero.test.js";
export * from "./errors/cycles.test.js";

// Performance Tests
export * from "./performance/compilation.test.js";
export * from "./performance/execution.test.js";

// Demand-Driven Semantics Tests
export * from "./demand-driven/no-outputs.test.js";
export * from "./demand-driven/unreferenced.test.js";
