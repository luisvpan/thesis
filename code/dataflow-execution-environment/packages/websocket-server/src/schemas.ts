import { t } from "elysia";
import type { DataflowProgram, ValidationError, MissingInput } from "@dataflow/shared/types";

const DataflowProgramSchema = t.Any();

const ValidationErrorSchema = t.Object({
  code: t.String(),
  message: t.String(),
  childMessage: t.Optional(t.String()),
  nodeId: t.Optional(t.String()),
  suggestion: t.Optional(t.String()),
  example: t.Optional(t.String())
});

const MissingInputSchema = t.Object({
  port: t.Number(),
  description: t.String(),
  childMessage: t.String()
});

const NodeStateCompletedSchema = t.Object({
  status: t.Literal("completed"),
  value: t.Unknown()
});

const NodeStatePendingSchema = t.Object({
  status: t.Literal("pending"),
  missingInputs: t.Array(MissingInputSchema)
});

const NodeStateProcessingSchema = t.Object({
  status: t.Literal("processing")
});

const NodeStateErrorSchema = t.Object({
  status: t.Literal("error"),
  error: t.String()
});

const NodeStateSchema = t.Union([
  NodeStateCompletedSchema,
  NodeStatePendingSchema,
  NodeStateProcessingSchema,
  NodeStateErrorSchema
]);

export const ValidateProgramMessageSchema = t.Object({
  type: t.Literal("validate_program"),
  messageId: t.Optional(t.String()),
  program: DataflowProgramSchema
});

export const EvaluateIncrementalMessageSchema = t.Object({
  type: t.Literal("evaluate_incremental"),
  messageId: t.Optional(t.String()),
  program: DataflowProgramSchema
});

export const SubscribeNodeMessageSchema = t.Object({
  type: t.Literal("subscribe_node"),
  messageId: t.Optional(t.String()),
  nodeId: t.String()
});

export const UnsubscribeNodeMessageSchema = t.Object({
  type: t.Literal("unsubscribe_node"),
  messageId: t.Optional(t.String()),
  nodeId: t.String()
});

export const ClientMessageSchema = t.Union([
  ValidateProgramMessageSchema,
  EvaluateIncrementalMessageSchema,
  SubscribeNodeMessageSchema,
  UnsubscribeNodeMessageSchema
]);

export const ValidationResultMessageSchema = t.Object({
  type: t.Literal("validation_result"),
  messageId: t.String(),
  errors: t.Array(ValidationErrorSchema),
  warnings: t.Array(ValidationErrorSchema)
});

export const EvaluationResultMessageSchema = t.Object({
  type: t.Literal("evaluation_result"),
  messageId: t.String(),
  nodeStates: t.Record(t.String(), NodeStateSchema),
  changedNodes: t.Array(t.String())
});

export const NodeStateChangedMessageSchema = t.Object({
  type: t.Literal("node_state_changed"),
  nodeId: t.String(),
  messageId: t.String(),
  state: t.Optional(NodeStateSchema)
});

export const ErrorMessageSchema = t.Union([
  t.Object({
    type: t.Literal("error"),
    code: t.Literal("CONNECTION_LIMIT_EXCEEDED"),
    message: t.Optional(t.String())
  }),
  t.Object({
    type: t.Literal("error"),
    code: t.Literal("MESSAGE_PARSE_ERROR"),
    messageId: t.Optional(t.String()),
    message: t.String()
  }),
  t.Object({
    type: t.Literal("error"),
    code: t.Literal("RATE_LIMIT_EXCEEDED"),
    messageId: t.String(),
    message: t.String()
  }),
  t.Object({
    type: t.Literal("error"),
    code: t.Literal("RUNTIME_NOT_FOUND"),
    messageId: t.String(),
    message: t.Literal("Client runtime not found")
  }),
  t.Object({
    type: t.Literal("error"),
    code: t.Literal("INVALID_MESSAGE"),
    messageId: t.String(),
    message: t.Literal("Unknown message type")
  }),
  t.Object({
    type: t.Literal("error"),
    code: t.Literal("MESSAGE_ERROR"),
    messageId: t.String(),
    message: t.String()
  }),
  t.Object({
    type: t.Literal("error"),
    code: t.Literal("EVALUATION_TIMEOUT"),
    messageId: t.String(),
    message: t.Literal("Evaluation took too long")
  })
]);

export const ServerMessageSchema = t.Union([
  ValidationResultMessageSchema,
  EvaluationResultMessageSchema,
  NodeStateChangedMessageSchema,
  ErrorMessageSchema
]);
