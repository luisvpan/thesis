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
    const nodes: DataflowNode[] = ast.statements.map(stmt => {
      const node = statementToNode(stmt);
      
      if (node.type === "DataSource" && this.isStreamDeclaration(node.dataType, node.value)) {
        return this.convertStreamDeclaration(node);
      }
      
      return node;
    });
    const edges: DataflowEdge[] = [];

    for (const stmt of ast.statements) {
      if (stmt.type === "TransformStatement") {
        for (let i = 0; i < stmt.inputs.length; i++) {
          edges.push({
            id: `edge_${this.edgeCounter++}`,
            from: stmt.inputs[i],
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
