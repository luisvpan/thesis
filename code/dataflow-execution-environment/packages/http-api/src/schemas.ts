import { t } from "elysia";
import type {
  DataflowProgram,
  DataflowNode,
  DataflowEdge,
  DataValue,
  ValidationError,
  DataType
} from "@dataflow/shared/types";

const DataTypeSchema = t.Union([
  t.Literal("natural"),
  t.Literal("integer"),
  t.Literal("decimal"),
  t.Literal("fraction"),
  t.Literal("text"),
  t.Literal("boolean"),
  t.Literal("shape"),
  t.Literal("car"),
  t.Literal("food"),
  t.Literal("animal"),
  t.Literal("person"),
  t.Literal("set"),
  t.Literal("stream")
]);

const NodeMetadataSchema = t.Optional(t.Object({
  label: t.Optional(t.String()),
  blockId: t.Optional(t.String()),
  position: t.Optional(t.Array(t.Number())),
  isLiteral: t.Optional(t.Boolean())
}));

const DataSourceNodeSchema = t.Object({
  id: t.String(),
  type: t.Literal("DataSource"),
  dataType: DataTypeSchema,
  value: t.Unknown(),
  metadata: NodeMetadataSchema
});

const TransformationNodeSchema = t.Object({
  id: t.String(),
  type: t.Literal("Transformation"),
  dataType: DataTypeSchema,
  operation: t.String(),
  inputs: t.Array(t.String()),
  metadata: NodeMetadataSchema
});

const OutputNodeSchema = t.Object({
  id: t.String(),
  type: t.Literal("Output"),
  dataType: DataTypeSchema,
  input: t.String(),
  metadata: NodeMetadataSchema
});

const DataflowNodeSchema = t.Union([
  DataSourceNodeSchema,
  TransformationNodeSchema,
  OutputNodeSchema
]);

const DataflowEdgeSchema = t.Object({
  id: t.String(),
  from: t.String(),
  to: t.String(),
  toPort: t.Optional(t.Number())
});

const DataflowProgramMetadataSchema = t.Object({
  programId: t.String(),
  activityId: t.Optional(t.String()),
  level: t.Optional(t.Number())
});

const DataflowProgramSchema = t.Any();

const PrimitiveValueSchema = t.Union([
  t.Object({ kind: t.Literal("natural"), value: t.Number() }),
  t.Object({ kind: t.Literal("integer"), value: t.Number() }),
  t.Object({ kind: t.Literal("decimal"), value: t.Number() }),
  t.Object({ kind: t.Literal("fraction"), value: t.Object({ numerator: t.Number(), denominator: t.Number() }) }),
  t.Object({ kind: t.Literal("text"), value: t.String() }),
  t.Object({ kind: t.Literal("boolean"), value: t.Boolean() })
]);

const CurriculumValueSchema = t.Union([
  t.Object({
    kind: t.Literal("shape"),
    type: t.Union([t.Literal("circle"), t.Literal("triangle"), t.Literal("square"), t.Literal("rectangle")]),
    size: t.Union([t.Literal("small"), t.Literal("medium"), t.Literal("large")]),
    color: t.String()
  }),
  t.Object({
    kind: t.Literal("car"),
    color: t.String()
  }),
  t.Object({
    kind: t.Literal("food"),
    taste: t.Union([t.Literal("sweet"), t.Literal("salty"), t.Literal("sour"), t.Literal("bitter")]),
    color: t.String()
  }),
  t.Object({
    kind: t.Literal("animal"),
    type: t.Union([t.Literal("dog"), t.Literal("cat"), t.Literal("bird"), t.Literal("fish"), t.Literal("rabbit"), t.Literal("turtle")]),
    color: t.String()
  }),
  t.Object({
    kind: t.Literal("person"),
    ageGroup: t.Union([t.Literal("child"), t.Literal("teenager"), t.Literal("adult"), t.Literal("senior")]),
    gender: t.Union([t.Literal("male"), t.Literal("female")])
  })
]);

const SetValueSchema = t.Object({
  kind: t.Literal("set"),
  elementType: DataTypeSchema,
  elements: t.Array(t.Unknown())
});

const StreamValueSchema = t.Object({
  kind: t.Literal("stream"),
  elementType: DataTypeSchema
});

const DataValueSchema = t.Any();

const ValidationErrorSchema = t.Object({
  code: t.String(),
  message: t.String(),
  childMessage: t.Optional(t.String()),
  nodeId: t.Optional(t.String()),
  suggestion: t.Optional(t.String()),
  example: t.Optional(t.String())
});

export const CompileRequestSchema = t.Object({
  program: DataflowProgramSchema
});

export const CompileSuccessResponseSchema = t.Object({
  success: t.Literal(true),
  programId: t.String(),
  errors: t.Array(t.Never()),
  warnings: t.Array(t.Never())
});

export const CompileFailureResponseSchema = t.Object({
  success: t.Literal(false),
  programId: t.String(),
  errors: t.Array(ValidationErrorSchema),
  warnings: t.Array(ValidationErrorSchema)
});

export const CompileResponseSchema = t.Union([
  CompileSuccessResponseSchema,
  CompileFailureResponseSchema
]);

export const ExecuteOptionsSchema = t.Object({
  maxTimesteps: t.Optional(t.Number()),
  includeTrace: t.Optional(t.Boolean()),
  traceLevel: t.Optional(t.Union([
    t.Literal("none"),
    t.Literal("basic"),
    t.Literal("medium"),
    t.Literal("detailed")
  ]))
});

export const ExecuteRequestSchema = t.Object({
  program: DataflowProgramSchema,
  options: t.Optional(ExecuteOptionsSchema)
});

export const ExecutionTraceSchema = t.Object({
  executionOrder: t.Array(t.String()),
  nodeEvaluations: t.Record(t.String(), t.Unknown()),
  cacheHits: t.Number(),
  cacheMisses: t.Number(),
  totalTime: t.String()
});

export const ExecuteSuccessResponseSchema = t.Object({
  success: t.Literal(true),
  programId: t.String(),
  outputs: t.Array(DataValueSchema),
  trace: t.Optional(ExecutionTraceSchema)
});

export const ExecuteFailureResponseSchema = t.Object({
  success: t.Literal(false),
  programId: t.String(),
  errors: t.Array(ValidationErrorSchema),
  warnings: t.Array(ValidationErrorSchema)
});

export const ExecuteResponseSchema = t.Union([
  ExecuteSuccessResponseSchema,
  ExecuteFailureResponseSchema
]);

export const HealthResponseSchema = t.Object({
  status: t.Literal("healthy"),
  version: t.String(),
  uptime: t.Number()
});

export const ErrorResponseSchema = t.Object({
  success: t.Literal(false),
  error: t.Union([
    t.Literal("Invalid request format"),
    t.Literal("Invalid JSON format"),
    t.Literal("Not found"),
    t.Literal("Too many requests"),
    t.Literal("Payload too large"),
    t.Literal("Too many concurrent requests"),
    t.Literal("Request timeout"),
    t.Literal("Internal server error")
  ]),
  message: t.Optional(t.String())
});
