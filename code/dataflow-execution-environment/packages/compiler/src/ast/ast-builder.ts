import { CstNode, IToken } from "chevrotain";
import type { Program, Statement, SourceStatement, TransformStatement, OutputStatement } from "./ast-types.js";

export class AstBuilder {
  program(ctx: CstNode): Program {
    const statements = ctx.children?.statement || [];
    return {
      type: "Program",
      statements: statements.map((s: any) => this.statement(s))
    };
  }

  statement(ctx: CstNode): Statement {
    if (ctx.children?.sourceStatement) {
      return this.sourceStatement((ctx.children.sourceStatement[0] as CstNode));
    } else if (ctx.children?.transformStatement) {
      return this.transformStatement((ctx.children.transformStatement[0] as CstNode));
    } else if (ctx.children?.outputStatement) {
      return this.outputStatement((ctx.children.outputStatement[0] as CstNode));
    }
    throw new Error('Unknown statement type');
  }

  sourceStatement(ctx: CstNode): SourceStatement {
    return {
      type: "SourceStatement",
      id: (ctx.children.Identifier[0] as IToken).image,
      dataType: this.typeDeclaration(ctx.children.typeDeclaration[0] as CstNode),
      value: this.value(ctx.children.value[0] as CstNode)
    };
  }

  transformStatement(ctx: CstNode): TransformStatement {
    const args: string[] = [];

    const operationExpression = ctx.children.operationExpression[0] as CstNode;
    if (operationExpression.children.argumentList) {
      const argList = ((operationExpression.children.argumentList[0] as CstNode).children.argument || []) as CstNode[];
      for (const arg of argList) {
        if (arg.children.Identifier) {
          args.push((arg.children.Identifier[0] as IToken).image);
        }
      }
    }

    return {
      type: "TransformStatement",
      id: (ctx.children.Identifier[0] as IToken).image,
      dataType: this.typeDeclaration(ctx.children.typeDeclaration[0] as CstNode),
      operation: this.extractOperationName(operationExpression),
      inputs: args
    };
  }

  outputStatement(ctx: CstNode): OutputStatement {
    return {
      type: "OutputStatement",
      id: (ctx.children.Identifier[0] as IToken).image,
      dataType: this.typeDeclaration(ctx.children.typeDeclaration[0] as CstNode),
      input: (ctx.children.Identifier[1] as IToken).image
    };
  }

  private extractOperationName(operationExprCtx: CstNode): string {
    const operationNameCtx = operationExprCtx.children.operationName?.[0] as CstNode;
    if (!operationNameCtx) return "UNKNOWN";

    const operationKeys = [
      "Add", "Subtract", "Multiply", "Divide", "Compare",
      "Filter", "Union", "Intersection", "Difference", "Complement",
      "Next", "First", "Fby", "Accumulate", "Sort",
      "AlphabeticalSort",
      "And", "Or", "Not",
      "CompareBySize", "CompareByColor", "CompareByType",
      "CompareByTaste", "CompareByAgeGroup", "CompareByGender",
      "FilterBySize", "FilterByColor", "FilterByType",
      "FilterByTaste", "FilterByAgeGroup", "FilterByGender"
    ];

    for (const key of operationKeys) {
      const token = operationNameCtx.children[key]?.[0] as IToken | undefined;
      if (token?.image) {
        return token.image.toUpperCase();
      }
    }

    return "UNKNOWN";
  }

  typeDeclaration(ctx: CstNode): string {
    const children = ctx.children;
    if (children.Natural) return "natural";
    if (children.Integer) return "integer";
    if (children.Decimal) return "decimal";
    if (children.Text) return "text";
    if (children.Boolean) return "boolean";
    if (children.setType) return this.setType(ctx);
    if (children.streamType) return this.streamType(ctx);
    return "unknown";
  }

  setType(ctx: CstNode): string {
    const typeDecl = this.typeDeclaration(ctx.children.typeDeclaration[0] as CstNode);
    return `set<${typeDecl}>`;
  }

  streamType(ctx: CstNode): string {
    const typeDecl = this.typeDeclaration(ctx.children.typeDeclaration[0] as CstNode);
    return `stream<${typeDecl}>`;
  }

  value(ctx: CstNode): unknown {
    const children = ctx.children;
    if (children.literal) {
      return this.literal(ctx);
    } else if (children.arrayLiteral) {
      return this.arrayLiteral(ctx);
    }
    return undefined;
  }

  literal(ctx: CstNode): unknown {
    const children = ctx.children;
    if (children.NumberLiteral) {
      return parseFloat((children.NumberLiteral[0] as IToken).image);
    } else if (children.StringLiteral) {
      return (children.StringLiteral[0] as IToken).image.slice(1, -1);
    } else if (children.objectLiteral) {
      return this.objectLiteral(ctx);
    } else if (children.True) {
      return true;
    } else if (children.False) {
      return false;
    }
    return undefined;
  }

  objectLiteral(ctx: CstNode): Record<string, unknown> {
    const obj: Record<string, unknown> = {};
    const children = ctx.children;
    const properties = children.Identifier;
    const literals = children.literal;
    for (let i = 0; i < properties.length; i++) {
      const key = (properties[i] as IToken).image;
      const value = this.literal(literals[i] as CstNode);
      obj[key] = value;
    }
    return obj;
  }

  arrayLiteral(ctx: CstNode): unknown[] {
    const children = ctx.children;
    return ((children.value as CstNode[]) || []).map((v: CstNode) => this.value(v));
  }
}
