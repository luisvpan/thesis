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
      throw new Error('Parser failed to generate CST');
    }
    return this.astBuilder.program(cst);
  }

  private buildProgramFromAst(ast: Program): { nodes: DataflowNode[]; edges: DataflowEdge[] } {
    this.edgeCounter = 0;
    const nodes: DataflowNode[] = ast.statements.map(stmt => statementToNode(stmt));
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
}
