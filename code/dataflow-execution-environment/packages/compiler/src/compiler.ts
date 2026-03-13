import { Lexer } from "chevrotain";
import { allTokens } from "./lexer/tokens.js";
import { DataflowParser } from "./parser/dataflow-parser.js";
import { AstBuilder, Program, statementToNode } from "./ast/index.js";
import { DagValidator } from "./validation/dag-validator.js";
import type { DataflowProgram, DataflowNode, DataflowEdge } from "@dataflow/shared/types";
import type { ValidationResult } from "@dataflow/shared/types";
import { getGenerator as getBuiltinGenerator } from "@dataflow/shared/generators";

export class Compiler {
  private lexer: Lexer;
  private parser: DataflowParser;
  private astBuilder: AstBuilder;
  private validator: DagValidator;
  private edgeCounter = 0;

  constructor() {
    this.lexer = new Lexer(allTokens);
    this.parser = new DataflowParser();
    this.astBuilder = new AstBuilder();
    this.validator = new DagValidator();
  }

  compile(source: string): ValidationResult & { program?: DataflowProgram } {
    const ast = this.parse(source);

    const { nodes, edges } = this.buildProgramFromAst(ast);

    const validationResult = this.validator.validate(ast.statements);
    if (!validationResult.success) {
      return validationResult;
    }

    return {
      success: true,
      errors: [],
      warnings: [],
      program: {
        metadata: {
          programId: `prog_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
        },
        graph: { nodes, edges }
      }
    };
  }

  parse(source: string): Program {
    const lexResult = this.lexer.tokenize(source);
    if (lexResult.errors.length > 0) {
      throw new Error(`Lexical errors: ${lexResult.errors.map(e => e.message).join(", ")}`);
    }

    this.parser.input = lexResult.tokens;
    const cst = this.parser.program();
    if (!cst) {
      if (this.parser.errors.length > 0) {
        this.parser.errors.forEach(e => {
          const line = e.token.startLine || "EOF";
          const column = e.token.startColumn || "";
          const offset = e.token.startOffset || "";
          const tokenImage = e.token.tokenType.name ? `${e.token.tokenType.name}` : "EOF";

          console.error(`Parser error: ${e.message} at token of type --> ${tokenImage} <-- (line ${line}, column ${column}, offset ${offset})`)
        });
      }
      throw new Error('Parser failed to generate CST');
    }

    return this.astBuilder.visit(cst);
  }

  private buildProgramFromAst(ast: Program): { nodes: DataflowNode[]; edges: DataflowEdge[] } {
    this.edgeCounter = 0;
    const nodes: DataflowNode[] = [];
    const edges: DataflowEdge[] = [];
    const literalData = new Map<string, { value: unknown; dataType: string }>();

    for (const stmt of ast.statements) {
      if (stmt.type === "TransformStatement") {
        for (const input of stmt.inputs) {
          if (input.startsWith("literal_") && !literalData.has(input)) {
            const match = input.match(/^literal_(\w+)_(.+)$/);
            if (match) {
              const type = match[1];
              const valueJson = match[2];
              const value = JSON.parse(`"${valueJson}"`);
              
              let nodeDataType: string;
              let nodeValue: unknown;

              if (type === "number") {
                nodeDataType = Number.isInteger(value) && value >= 0 ? "natural" : "integer";
                nodeValue = value;
              } else if (type === "string") {
                nodeDataType = "text";
                nodeValue = { kind: "text", value };
              } else if (type === "boolean") {
                nodeDataType = "boolean";
                nodeValue = { kind: "boolean", value };
              } else if (type === "object") {
                nodeDataType = stmt.dataType;
                nodeValue = value;
              } else {
                nodeDataType = "unknown";
                nodeValue = value;
              }

              literalData.set(input, { value: nodeValue, dataType: nodeDataType });
            }
          }
        }
      }
    }

    for (const literalId of literalData.keys()) {
      const literalInfo = literalData.get(literalId)!;
      const literalNode = {
        id: literalId,
        type: "DataSource",
        dataType: literalInfo.dataType,
        value: literalInfo.value,
        metadata: { isLiteral: true }
      };
      nodes.push(literalNode);
    }

    for (const stmt of ast.statements) {
      const node = statementToNode(stmt);

      if (node.type === "DataSource" && this.isStreamDeclaration(node.dataType, node.value)) {
        nodes.push(this.convertStreamDeclaration(node));
      } else if (!literalData.has(node.id)) {
        nodes.push(node);
      }
    }

    for (const stmt of ast.statements) {
      if (stmt.type === "TransformStatement") {
        const inputIds = this.resolveLiteralInputs(stmt.inputs);

        for (let i = 0; i < stmt.inputs.length; i++) {
          const inputId = inputIds[i];
          edges.push({
            id: `edge_${this.edgeCounter++}`,
            from: inputId,
            to: stmt.id,
            toPort: i
          });
        }
      } else if (stmt.type === "OutputStatement") {
        edges.push({
          id: `edge_${this.edgeCounter++}`,
          from: stmt.input,
          to: stmt.id
        });
      }
    }

    return { nodes, edges };
  }

  private resolveLiteralInputs(inputs: string[]): string[] {
    return inputs;
  }

  private createLiteralNodeFromId(literalId: string, dataType: string): DataflowNode {
    const match = literalId.match(/^literal_(\w+)_(.+)$/);
    if (!match) {
      throw new Error(`Invalid literal ID format: ${literalId}`);
    }

    const type = match[1];
    const valueJson = match[2];
    const value = JSON.parse(`"${valueJson}"`);
    
    let nodeDataType: string;
    let nodeValue: unknown;

    if (type === "number") {
      nodeDataType = Number.isInteger(value) && value >= 0 ? "natural" : "integer";
      nodeValue = value;
    } else if (type === "string") {
      nodeDataType = "text";
      nodeValue = { kind: "text", value };
    } else if (type === "boolean") {
      nodeDataType = "boolean";
      nodeValue = { kind: "boolean", value };
    } else if (type === "object") {
      nodeDataType = dataType;
      nodeValue = value;
    } else {
      nodeDataType = "unknown";
      nodeValue = value;
    }

    return {
      id: literalId,
      type: "DataSource",
      dataType: nodeDataType,
      value: nodeValue,
      metadata: { isLiteral: true }
    };
  }

  private createLiteralNode(value: unknown, index: number): DataflowNode {
    let dataType: string;
    let nodeValue: unknown;

    if (typeof value === "number") {
      dataType = Number.isInteger(value) && value >= 0 ? "natural" : "integer";
      nodeValue = value;
    } else if (typeof value === "string") {
      dataType = "text";
      nodeValue = { kind: "text", value };
    } else if (typeof value === "boolean") {
      dataType = "boolean";
      nodeValue = { kind: "boolean", value };
    } else {
      dataType = "unknown";
      nodeValue = value;
    }

    return {
      id: `literal_${index}`,
      type: "DataSource",
      dataType,
      value: nodeValue,
      metadata: { isLiteral: true }
    };
  }

  private isStreamDeclaration(dataType: string | { kind: string }, value: unknown): boolean {
    const dataTypeStr = typeof dataType === "string" ? dataType : (dataType as { kind: string }).kind;
    const isStream = dataTypeStr.startsWith("stream");
    const valueObj = value as { type?: string };
    return isStream && valueObj?.type === "stream";
  }

  private convertStreamDeclaration(node: DataflowNode): DataflowNode {
    if (node.type !== "DataSource") {
      return node;
    }
    
    const dataSourceNode = node as { id: string; type: "DataSource"; dataType: string; value: unknown; metadata?: unknown };
    const value = dataSourceNode.value as { type: "stream"; source: { type: "generator" | "sensor" | "external"; name: string } };
    const dataType = dataSourceNode.dataType as string;
    
    if (value.source.type === "generator") {
      const generatorFactory = getBuiltinGenerator(value.source.name);
      if (!generatorFactory) {
        throw new Error(`Unknown generator: ${value.source.name}`);
      }
      
      const elementType = this.extractElementType(dataType);
      
      return {
        ...node,
        value: {
          kind: "stream",
          elementType,
          generatorFactory: generatorFactory,
          generator: generatorFactory()
        }
      } as DataflowNode;
    }
    
    return node;
  }

  private extractElementType(dataType: string): string {
    const match = dataType.match(/^stream<(.+)>$/);
    return match ? match[1] : "unknown";
  }
}
