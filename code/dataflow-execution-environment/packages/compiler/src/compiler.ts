import { Lexer } from "chevrotain";
import { allTokens } from "./lexer/tokens.js";
import { DataflowParser } from "./parser/dataflow-parser.js";
import { AstBuilder, Program, statementToNode, TransformStatement } from "./ast/index.js";
import { DagValidator } from "./validation/dag-validator.js";
import type { DataflowProgram, DataflowNode, DataflowEdge, DataSourceNode } from "@dataflow/shared/types";
import type { ValidationResult } from "@dataflow/shared/types";
import { getGenerator as getBuiltinGenerator } from "@dataflow/shared/generators";
import { OPERATION_REGISTRY } from "@dataflow/shared/operations";
import type { DataType } from "@dataflow/shared/types";

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

    const expandedAst = this.expandNestedOperations(ast);

    const { nodes, edges } = this.buildProgramFromAst(expandedAst);

    const validationResult = this.validator.validate(expandedAst.statements);
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

  private expandNestedOperations(ast: Program): Program {
    if (!ast.nestedOperations || ast.nestedOperations.size === 0) {
      return ast;
    }

    const expandedStatements: TransformStatement[] = [];
    const allStatements = [...ast.statements];
    const dataTypeMap = new Map<string, DataType>();

    for (const stmt of ast.statements) {
      if (stmt.type === "SourceStatement") {
        dataTypeMap.set(stmt.id, stmt.dataType as DataType);
      } else if (stmt.type === "TransformStatement") {
        dataTypeMap.set(stmt.id, stmt.dataType as DataType);
      }
    }

    for (const [nestedOpId, nestedOp] of ast.nestedOperations) {
      const inputTypes: DataType[] = [];
      for (const inputId of nestedOp.inputs) {
        const inputType = dataTypeMap.get(inputId);
        if (!inputType) {
          const inferredType = this.inferTypeForLiteralOrNested(inputId, dataTypeMap);
          inputTypes.push(inferredType || "natural");
        } else {
          inputTypes.push(inputType);
        }
      }

      const outputType = this.inferOutputType(nestedOp.operation, inputTypes);
      if (outputType) {
        dataTypeMap.set(nestedOpId, outputType);
      }

      const nestedStmt: TransformStatement = {
        type: "TransformStatement",
        id: nestedOpId,
        dataType: outputType || "natural",
        operation: nestedOp.operation,
        inputs: nestedOp.inputs
      };

      expandedStatements.push(nestedStmt);
    }

    return {
      type: "Program",
      statements: [...expandedStatements, ...allStatements] as any
    };
  }

  private inferOutputType(operation: string, inputTypes: DataType[]): DataType | null {
    const registryEntry = OPERATION_REGISTRY[operation as any];
    if (!registryEntry) return null;

    const contract = registryEntry.contracts[0];
    if (inputTypes.length !== contract.arity) return null;

    return contract.outputType as DataType;
  }

  private inferTypeForLiteralOrNested(inputId: string, dataTypeMap: Map<string, DataType>): DataType | null {
    if (inputId.startsWith("literal_number")) {
      const match = inputId.match(/^literal_number_(.+)$/);
      if (match) {
        try {
          const value = JSON.parse(atob(match[1]));
          const num = value as number;
          if (!Number.isInteger(num)) {
            return "decimal";
          }
          return num >= 0 ? "natural" : "integer";
        } catch {
          // Ignore parsing errors
        }
      }
      return null;
    }
    if (inputId.startsWith("literal_string")) {
      return "text";
    }
    if (inputId.startsWith("literal_boolean")) {
      return "boolean";
    }
    if (inputId.startsWith("literal_object")) {
      const match = inputId.match(/^literal_object_(.+)$/);
      if (match) {
        try {
          const value = JSON.parse(atob(match[1]));
          if (typeof value === 'object' && value !== null && 'kind' in value) {
            const kind = (value as { kind: string }).kind;
            if (['fraction', 'shape', 'car', 'food', 'animal', 'person'].includes(kind)) {
              return kind as DataType;
            }
          }
        } catch {
          // Ignore parsing errors
        }
      }
      return null;
    }
    if (inputId.startsWith("nested_op_")) {
      return dataTypeMap.get(inputId) || null;
    }
    return null;
  }

  private buildProgramFromAst(ast: Program): { nodes: DataflowNode[]; edges: DataflowEdge[] } {
    this.edgeCounter = 0;
    const nodes: DataflowNode[] = [];
    const edges: DataflowEdge[] = [];
    const literalData = new Map<string, { value: unknown; dataType: DataType }>();

      for (const stmt of ast.statements) {
        if (stmt.type === "TransformStatement") {
          for (const input of stmt.inputs) {
            if (input.startsWith("literal_") && !literalData.has(input)) {
              const match = input.match(/^literal_(\w+)_(.+)$/);
              if (match) {
                const type = match[1];
                const valueJson = match[2];
                let value: unknown;

                try {
                  if (type === "string") {
                    value = valueJson;
                  } else if (type === "boolean") {
                    value = valueJson === "true";
                  } else {
                    value = JSON.parse(atob(valueJson));
                  }
                } catch {
                  value = valueJson;
                }

                let nodeDataType: DataType;
                let nodeValue: unknown;

                if (type === "number") {
                  const numValue = value as number;
                  if (!Number.isInteger(numValue)) {
                    nodeDataType = "decimal";
                  } else if (numValue >= 0) {
                    nodeDataType = "natural";
                  } else {
                    nodeDataType = "integer";
                  }
                  nodeValue = numValue;
                } else if (type === "string") {
                  nodeDataType = "text";
                  nodeValue = { kind: "text", value };
                } else if (type === "boolean") {
                  nodeDataType = "boolean";
                  nodeValue = { kind: "boolean", value };
                } else if (type === "object") {
                  if (typeof value === 'object' && value !== null && 'kind' in value) {
                    const kind = (value as { kind: string }).kind;
                    if (['fraction', 'shape', 'car', 'food', 'animal', 'person'].includes(kind)) {
                      nodeDataType = kind as DataType;
                    } else {
                      nodeDataType = stmt.dataType as DataType;
                    }
                  } else {
                    nodeDataType = stmt.dataType as DataType;
                  }
                  nodeValue = value;
                } else {
                  nodeDataType = "text";
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
      const literalNode: DataSourceNode = {
        id: literalId,
        type: "DataSource" as const,
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
        if (node.type === "DataSource" && typeof node.dataType === "string" && node.dataType.startsWith("set<")) {
          const elementType = node.dataType.slice(4, -1);
          const wrappedValue = this.wrapSetElements(node.value as unknown[], elementType);
          nodes.push({ ...node, value: wrappedValue });
        } else {
          nodes.push(node);
        }
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

  private wrapSetElements(elements: unknown[], elementType: string): { kind: string; elements: unknown[] } {
    const wrappedElements = elements.map(elem => {
      if (typeof elem === 'object' && elem !== null && 'kind' in elem) {
        return elem;
      }
      
      switch (elementType) {
        case 'integer':
        case 'natural':
          return { kind: elementType, value: elem as number };
        case 'decimal':
          return { kind: 'decimal', value: elem as number };
        case 'fraction':
          return elem;
        case 'text':
          return { kind: 'text', value: elem as string };
        case 'boolean':
          return { kind: 'boolean', value: elem as boolean };
        default:
          return elem;
      }
    });
    
    return { kind: 'set', elements: wrappedElements };
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
    const value = JSON.parse(atob(valueJson));

    let nodeDataType: DataType;
    let nodeValue: unknown;

    if (type === "number") {
      const numValue = value as number;
      if (Number.isInteger(numValue)) {
        if (numValue >= 0) {
          nodeDataType = "natural";
          nodeValue = { kind: "natural", value: numValue };
        } else {
          nodeDataType = "integer";
          nodeValue = { kind: "integer", value: numValue };
        }
      } else {
        nodeDataType = "decimal";
        nodeValue = { kind: "decimal", value: numValue };
      }
    } else if (type === "string") {
      nodeDataType = "text";
      nodeValue = { kind: "text", value };
    } else if (type === "boolean") {
      nodeDataType = "boolean";
      nodeValue = { kind: "boolean", value };
    } else if (type === "object") {
      nodeDataType = dataType as DataType;
      nodeValue = value;
    } else {
      nodeDataType = "text";
      nodeValue = value;
    }

    return {
      id: literalId,
      type: "DataSource" as const,
      dataType: nodeDataType,
      value: nodeValue,
      metadata: { isLiteral: true }
    };
  }

  private createLiteralNode(value: unknown, index: number): DataSourceNode {
    let dataType: DataType;
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
      dataType = "text";
      nodeValue = value;
    }

    return {
      id: `literal_${index}`,
      type: "DataSource" as const,
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
