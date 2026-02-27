import type { Program, Statement, SourceStatement, TransformStatement, OutputStatement } from "./ast-types.js";

export class AstBuilder {
  program(ctx: any): Program {
    return {
      type: "Program",
      statements: (ctx.statement || []).map((s: any) => this.statement(s))
    };
  }

  statement(ctx: any): Statement {
    if (ctx.sourceStatement) {
      return this.sourceStatement(ctx.sourceStatement[0]);
    } else if (ctx.transformStatement) {
      return this.transformStatement(ctx.transformStatement[0]);
    } else {
      return this.outputStatement(ctx.outputStatement[0]);
    }
  }

  sourceStatement(ctx: any): SourceStatement {
    return {
      type: "SourceStatement",
      id: ctx.Identifier[0].image,
      dataType: this.typeDeclaration(ctx.typeDeclaration[0]),
      value: this.value(ctx.value[0])
    };
  }

  transformStatement(ctx: any): TransformStatement {
    const args: string[] = [];
    if (ctx.operationExpression[0].children.argumentList) {
      const argList = ctx.operationExpression[0].children.argumentList[0].children.argument || [];
      for (const arg of argList) {
        if (arg.children.Identifier) {
          args.push(arg.children.Identifier[0].image);
        }
      }
    }
    
    return {
      type: "TransformStatement",
      id: ctx.Identifier[0].image,
      dataType: this.typeDeclaration(ctx.typeDeclaration[0]),
      operation: ctx.operationExpression[0].children.operationName[0].tokenType.name.toUpperCase(),
      inputs: args
    };
  }

  outputStatement(ctx: any): OutputStatement {
    return {
      type: "OutputStatement",
      id: ctx.Identifier[0].image,
      dataType: this.typeDeclaration(ctx.typeDeclaration[0]),
      input: ctx.Identifier[1].image
    };
  }

  typeDeclaration(ctx: any): string {
    if (ctx.Natural) return "natural";
    if (ctx.Integer) return "integer";
    if (ctx.Decimal) return "decimal";
    if (ctx.Text) return "text";
    if (ctx.Boolean) return "boolean";
    if (ctx.setType) return this.setType(ctx.setType[0]);
    if (ctx.streamType) return this.streamType(ctx.streamType[0]);
    return "unknown";
  }

  setType(ctx: any): string {
    const elementType = this.typeDeclaration(ctx.typeDeclaration[0]);
    return `set<${elementType}>`;
  }

  streamType(ctx: any): string {
    const elementType = this.typeDeclaration(ctx.typeDeclaration[0]);
    return `stream<${elementType}>`;
  }

  value(ctx: any): unknown {
    if (ctx.literal) {
      return this.literal(ctx.literal[0]);
    } else if (ctx.arrayLiteral) {
      return this.arrayLiteral(ctx.arrayLiteral[0]);
    }
    return undefined;
  }

  literal(ctx: any): unknown {
    if (ctx.NumberLiteral) {
      return parseFloat(ctx.NumberLiteral[0].image);
    } else if (ctx.StringLiteral) {
      return ctx.StringLiteral[0].image.slice(1, -1);
    } else if (ctx.objectLiteral) {
      return this.objectLiteral(ctx.objectLiteral[0]);
    } else if (ctx.True) {
      return true;
    } else if (ctx.False) {
      return false;
    }
    return undefined;
  }

  objectLiteral(ctx: any): Record<string, unknown> {
    const obj: Record<string, unknown> = {};
    const properties = ctx.Identifier;
    for (let i = 0; i < properties.length; i++) {
      const key = properties[i].image;
      const value = this.literal(ctx.literal[i]);
      obj[key] = value;
    }
    return obj;
  }

  arrayLiteral(ctx: any): unknown[] {
    return (ctx.value || []).map((v: any) => this.value(v));
  }
}
