import type { Program, Statement, SourceStatement, TransformStatement, OutputStatement } from "./ast-types.js";

export class AstBuilder {
  program(ctx: any): Program {
    const statements = ctx.children?.statement || ctx.statement || [];
    return {
      type: "Program",
      statements: statements.map((s: any) => this.statement(s))
    };
  }

  statement(ctx: any): Statement {
    const name = ctx.name;
    if (name === "sourceStatement") {
      return this.sourceStatement(ctx);
    } else if (name === "transformStatement") {
      return this.transformStatement(ctx);
    } else if (name === "outputStatement") {
      return this.outputStatement(ctx);
    }
    throw new Error('Unknown statement type');
  }

  sourceStatement(ctx: any): SourceStatement {
    return {
      type: "SourceStatement",
      id: ctx.children.Identifier[0].image,
      dataType: this.typeDeclaration(ctx.children.typeDeclaration[0]),
      value: this.value(ctx.children.value[0])
    };
  }

  transformStatement(ctx: any): TransformStatement {
    const args: string[] = [];
    if (ctx.children.operationExpression[0].children.argumentList) {
      const argList = ctx.children.operationExpression[0].children.argumentList[0].children.argument || [];
      for (const arg of argList) {
        if (arg.children.Identifier) {
          args.push(arg.children.Identifier[0].image);
        }
      }
    }
    
    return {
      type: "TransformStatement",
      id: ctx.children.Identifier[0].image,
      dataType: this.typeDeclaration(ctx.children.typeDeclaration[0]),
      operation: ctx.children.operationExpression[0].children.operationName[0].tokenType.name.toUpperCase(),
      inputs: args
    };
  }

  outputStatement(ctx: any): OutputStatement {
    return {
      type: "OutputStatement",
      id: ctx.children.Identifier[0].image,
      dataType: this.typeDeclaration(ctx.children.typeDeclaration[0]),
      input: ctx.children.Identifier[1].image
    };
  }

  typeDeclaration(ctx: any): string {
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

  setType(ctx: any): string {
    const typeDecl = this.typeDeclaration(ctx.children.typeDeclaration);
    return `set<${typeDecl}>`;
  }

  streamType(ctx: any): string {
    const typeDecl = this.typeDeclaration(ctx.children.typeDeclaration);
    return `stream<${typeDecl}>`;
  }

  value(ctx: any): unknown {
    const children = ctx.children;
    if (children.literal) {
      return this.literal(ctx);
    } else if (children.arrayLiteral) {
      return this.arrayLiteral(ctx);
    }
    return undefined;
  }

  literal(ctx: any): unknown {
    const children = ctx.children;
    if (children.NumberLiteral) {
      return parseFloat(children.NumberLiteral[0].image);
    } else if (children.StringLiteral) {
      return children.StringLiteral[0].image.slice(1, -1);
    } else if (children.objectLiteral) {
      return this.objectLiteral(ctx);
    } else if (children.True) {
      return true;
    } else if (children.False) {
      return false;
    }
    return undefined;
  }

  objectLiteral(ctx: any): Record<string, unknown> {
    const obj: Record<string, unknown> = {};
    const children = ctx.children;
    const properties = children.Identifier;
    const literals = children.literal;
    for (let i = 0; i < properties.length; i++) {
      const key = properties[i].image;
      const value = this.literal(literals[i]);
      obj[key] = value;
    }
    return obj;
  }

  arrayLiteral(ctx: any): unknown[] {
    const children = ctx.children;
    return (children.value || []).map((v: any) => this.value(v));
  }
}
