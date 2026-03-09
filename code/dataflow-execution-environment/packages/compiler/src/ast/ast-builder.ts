import { IToken } from "chevrotain";
import { DataflowParser } from "../parser"; // Importa tu clase de Parser
import { ArgumentCstChildren, ArgumentListCstChildren, ArrayLiteralCstChildren, LiteralCstChildren, ObjectLiteralCstChildren, OperationExpressionCstChildren, OperationNameCstChildren, OutputStatementCstChildren, ProgramCstChildren, SetTypeCstChildren, SourceStatementCstChildren, StatementCstChildren, StreamTypeCstChildren, TransformStatementCstChildren, TypeDeclarationCstChildren, ValueCstChildren } from "../types/cst-generated-types";

const parserInstance = new DataflowParser();
const BaseVisitor = parserInstance.getBaseCstVisitorConstructor();

export class AstBuilder extends BaseVisitor {
  constructor() {
    super();
    this.validateVisitor();
  }

  program(ctx: ProgramCstChildren) {
    return {
      type: "Program",
      statements: ctx.statement?.map((s) => this.visit(s)) || []
    };
  }

  statement(ctx: StatementCstChildren) {
    if (ctx.sourceStatement) return this.visit(ctx.sourceStatement);
    if (ctx.transformStatement) return this.visit(ctx.transformStatement);
    if (ctx.outputStatement) return this.visit(ctx.outputStatement);
  }

  sourceStatement(ctx: SourceStatementCstChildren) {
    return {
      type: "SourceStatement",
      id: ctx.Identifier[0].image,
      dataType: this.visit(ctx.typeDeclaration),
      value: this.visit(ctx.value) // El visitor entra solo a la subregla
    };
  }

  typeDeclaration(ctx: TypeDeclarationCstChildren) {
    if (ctx.Natural) return "natural";
    if (ctx.Integer) return "integer";
    if (ctx.Decimal) return "decimal";
    if (ctx.Text) return "text";
    if (ctx.Boolean) return "boolean";
    if (ctx.setType) return this.visit(ctx.setType);
    if (ctx.streamType) return this.visit(ctx.streamType);
  }

  setType(ctx: SetTypeCstChildren) {
    return `set<${this.visit(ctx.typeDeclaration)}>`;
  }

  streamType(ctx: StreamTypeCstChildren) {
    return `stream<${this.visit(ctx.typeDeclaration)}>`;
  }

  value(ctx: ValueCstChildren) {
    if (ctx.literal) return this.visit(ctx.literal);
    if (ctx.objectLiteral) return this.visit(ctx.objectLiteral);
    if (ctx.arrayLiteral) return this.visit(ctx.arrayLiteral);
    if (ctx.setLiteral) return this.visit(ctx.setLiteral);
    if (ctx.streamLiteral) return this.visit(ctx.streamLiteral);
  }

  literal(ctx: LiteralCstChildren) {
    if (ctx.NumberLiteral) return parseFloat(ctx.NumberLiteral[0].image);
    if (ctx.StringLiteral) return ctx.StringLiteral[0].image.slice(1, -1);
    if (ctx.True) return true;
    if (ctx.False) return false;
  }

  objectLiteral(ctx: ObjectLiteralCstChildren) {
    const obj: Record<string, unknown> = {};

    ctx.Identifier?.forEach((id: IToken, index: number) => {
      obj[id.image] = this.visit(ctx.literal![index]);
    });
    return obj;
  }

  arrayLiteral(ctx: ArrayLiteralCstChildren) {
    return ctx.value?.map((v) => this.visit(v)) || [];
  }

  setLiteral(ctx: any) {
    return ctx.value?.map((v: any) => this.visit(v)) || [];
  }

  sensorSource(ctx: any) {
    return {
      type: "sensor",
      name: ctx.StringLiteral[0].image.slice(1, -1)
    };
  }

  generatorSource(ctx: any) {
    return {
      type: "generator",
      name: ctx.Identifier[0].image
    };
  }

  externalSource(ctx: any) {
    return {
      type: "external",
      name: ctx.StringLiteral[0].image.slice(1, -1)
    };
  }

  streamLiteral(ctx: any) {
    const source = ctx.sensorSource ? this.visit(ctx.sensorSource) :
                  ctx.generatorSource ? this.visit(ctx.generatorSource) :
                  ctx.externalSource ? this.visit(ctx.externalSource) : null;

    return {
      type: "stream",
      source
    };
  }

  transformStatement(ctx: TransformStatementCstChildren) {
    const opExpr = ctx.operationExpression[0];
    return {
      type: "TransformStatement",
      id: ctx.Identifier[0].image,
      dataType: this.visit(ctx.typeDeclaration),
      ...this.visit(opExpr)
    };
  }

  // 1. Añade el método que faltaba para Output
  outputStatement(ctx: OutputStatementCstChildren) {
    return {
      type: "OutputStatement",
      id: ctx.Identifier[0].image,
      dataType: this.visit(ctx.typeDeclaration),
      input: ctx.Identifier[1].image // Segundo identificador
    };
  }

  operationExpression(ctx: OperationExpressionCstChildren) {
    // Al llamar a visit(ctx.operationName), se ejecutará el método de abajo
    const operationToken = this.visit(ctx.operationName) as IToken;
    const operationName = operationToken.image.toUpperCase();

    const inputs = ctx.argumentList
      ? this.visit(ctx.argumentList)
      : [];

    return { operation: operationName, inputs };
  }

  operationName(ctx: OperationNameCstChildren) {
    // ctx aquí son los children de la regla operationName
    const allChildren = Object.values(ctx);
    if (allChildren.length > 0 && allChildren[0].length > 0) {
      return allChildren[0][0];
    }
    throw new Error("Missing operation name token");
  }

  argumentList(ctx: ArgumentListCstChildren) {
    return ctx.argument?.map((arg) => this.visit(arg)) || [];
  }

  argument(ctx: ArgumentCstChildren) {
    if (ctx.Identifier) return ctx.Identifier[0].image;
    if (ctx.literal) return this.visit(ctx.literal);
  }
}