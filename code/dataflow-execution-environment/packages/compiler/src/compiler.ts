import { Lexer } from "chevrotain";
import { allTokens } from "./lexer/tokens.js";
import { DataflowParser } from "./parser/dataflow-parser.js";
import { AstBuilder, Program, statementToNode } from "./ast/index.js";
import { DagValidator } from "./validation/dag-validator.js";
import type { DataflowProgram, DataflowNode, DataflowEdge } from "@dataflow/shared/types";
import type { ValidationResult } from "@dataflow/shared/types";

export class Compiler {
  private lexer: Lexer;
  private parser: DataflowParser;
  private astBuilder: AstBuilder;
  private validator: DagValidator;

  constructor() {
    this.lexer = new Lexer(allTokens);
    this.parser = new DataflowParser();
    this.astBuilder = new AstBuilder();
    this.validator = new DagValidator();
  }

  compile(program: DataflowProgram): ValidationResult & { graph?: { nodes: DataflowNode[]; edges: DataflowEdge[] } } {
    const { graph } = program;

    const validationResult = this.validate(graph);
    if (!validationResult.success) {
      return validationResult;
    }

    return {
      success: true,
      errors: [],
      warnings: [],
      graph: this.buildGraph(graph)
    };
  }

  parse(source: string): Program {
    const lexResult = this.lexer.tokenize(source);
    if (lexResult.errors.length > 0) {
      throw new Error(`Lexical errors: ${lexResult.errors.map(e => e.message).join(", ")}`);
    }
    this.parser.input = lexResult.tokens;
    const cst = this.parser.program();
    return this.astBuilder.program(cst);
  }

  private validate(graph: { nodes: DataflowNode[]; edges: DataflowEdge[] }): ValidationResult {
    const statements = graph.nodes.map(node => ({
      type: node.type === "DataSource" ? "SourceStatement" as const :
             node.type === "Transformation" ? "TransformStatement" as const :
             "OutputStatement" as const,
      id: node.id,
      dataType: typeof node.dataType === "string" ? node.dataType : JSON.stringify(node.dataType),
      value: (node as any).value,
      operation: (node as any).operation,
      inputs: (node as any).inputs,
      input: (node as any).input
    }));

    return this.validator.validate(statements);
  }

  private buildGraph(graph: { nodes: DataflowNode[]; edges: DataflowEdge[] }): { nodes: DataflowNode[]; edges: DataflowEdge[] } {
    return graph;
  }
}
