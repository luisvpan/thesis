import type { CstNode, IToken } from "chevrotain";
import { parserInstance } from "./parser";
import type {
  Program,
  Statement,
  SourceStatement,
  TransformStatement,
  SinkStatement,
  Operation,
  Expression,
  Literal,
  ObjectLiteral,
  ObjectProperty,
  StringLiteral,
  ArrayLiteral,
  NumberLiteral,
} from "./ast";

// CST Node types
interface ProgramCstNode extends CstNode {
  children: {
    statement?: CstNode[];
  };
}

interface StatementCstNode extends CstNode {
  children: {
    sourceStatement?: CstNode[];
    transformStatement?: CstNode[];
    sinkStatement?: CstNode[];
  };
}

interface SourceStatementCstNode extends CstNode {
  children: {
    Identifier: IToken[];
    literal?: CstNode[];
  };
}

interface TransformStatementCstNode extends CstNode {
  children: {
    Identifier: IToken[];
    operation?: CstNode[];
    argumentList?: CstNode[];
  };
}

interface SinkStatementCstNode extends CstNode {
  children: {
    Identifier: IToken[];
  };
}

interface OperationCstNode extends CstNode {
  children: {
    Sum?: IToken[];
    Substract?: IToken[];
    Multiply?: IToken[];
    Divide?: IToken[];
    LessThan?: IToken[];
    GreaterThan?: IToken[];
    OrderAsc?: IToken[];
    OrderDesc?: IToken[];
    Filter?: IToken[];
  };
}

interface ArgumentListCstNode extends CstNode {
  children: {
    expression: CstNode[];
  };
}

interface ExpressionCstNode extends CstNode {
  children: {
    Identifier?: IToken[];
    literal?: CstNode[];
  };
}

interface LiteralCstNode extends CstNode {
  children: {
    objectLiteral?: CstNode[];
    arrayLiteral?: CstNode[];
    StringLiteral?: IToken[];
    NumberLiteral?: IToken[];
  };
}

interface ArrayLiteralCstNode extends CstNode {
  children: {
    expression?: CstNode[];
  };
}

interface ObjectLiteralCstNode extends CstNode {
  children: {
    kvPair?: CstNode[];
  };
}

interface KvPairCstNode extends CstNode {
  children: {
    StringLiteral: IToken[];
    NumberLiteral?: IToken[];
  };
}

// Helper: Remove quotes from string literal
function unquote(str: string): string {
  if (str.startsWith('"') && str.endsWith('"')) {
    return str.slice(1, -1);
  }
  return str;
}

// Get the base visitor class from the parser
const BaseCstVisitor = parserInstance.getBaseCstVisitorConstructor();

export class DataflowAstVisitor extends BaseCstVisitor {
  constructor() {
    super();
    this.validateVisitor();
  }

  program(ctx: ProgramCstNode["children"]): Program {
    const statements: Statement[] = [];

    if (ctx.statement) {
      for (const stmtCst of ctx.statement) {
        statements.push(this.visit(stmtCst));
      }
    }

    return {
      type: "Program",
      statements,
    };
  }

  statement(ctx: StatementCstNode["children"]): Statement {
    if (ctx.sourceStatement) {
      return this.visit(ctx.sourceStatement[0]);
    } else if (ctx.transformStatement) {
      return this.visit(ctx.transformStatement[0]);
    } else if (ctx.sinkStatement) {
      return this.visit(ctx.sinkStatement[0]);
    }
    throw new Error("Unknown statement type");
  }

  sourceStatement(ctx: SourceStatementCstNode["children"]): SourceStatement {
    return {
      type: "SourceStatement",
      identifier: ctx.Identifier[0].image,
      value: ctx.literal ? this.visit(ctx.literal[0]) : undefined,
    };
  }

  transformStatement(ctx: TransformStatementCstNode["children"]): TransformStatement {
    return {
      type: "TransformStatement",
      identifier: ctx.Identifier[0].image,
      operation: ctx.operation ? this.visit(ctx.operation[0]) : undefined,
      arguments: ctx.argumentList ? this.visit(ctx.argumentList[0]) : [],
    };
  }

  sinkStatement(ctx: SinkStatementCstNode["children"]): SinkStatement {
    return {
      type: "SinkStatement",
      identifier: ctx.Identifier[0].image,
      sourceIdentifier: ctx.Identifier[1]?.image,
    };
  }

  operation(ctx: OperationCstNode["children"]): Operation {
    if (ctx.Sum) return "sum";
    if (ctx.Substract) return "substract";
    if (ctx.Multiply) return "multiply";
    if (ctx.Divide) return "divide";
    if (ctx.LessThan) return "less_than";
    if (ctx.GreaterThan) return "greater_than";
    if (ctx.OrderAsc) return "order_asc";
    if (ctx.OrderDesc) return "order_desc";
    if (ctx.Filter) return "filter";
    throw new Error("Unknown operation");
  }

  argumentList(ctx: ArgumentListCstNode["children"]): Expression[] {
    return ctx.expression.map((expr) => this.visit(expr));
  }

  expression(ctx: ExpressionCstNode["children"]): Expression {
    if (ctx.Identifier) {
      return {
        type: "Identifier",
        name: ctx.Identifier[0].image,
      };
    } else if (ctx.literal) {
      return this.visit(ctx.literal[0]);
    }
    throw new Error("Unknown expression type");
  }

  literal(ctx: LiteralCstNode["children"]): Literal {
    if (ctx.objectLiteral) {
      return this.visit(ctx.objectLiteral[0]);
    } else if (ctx.arrayLiteral) {
      return this.visit(ctx.arrayLiteral[0]);
    } else if (ctx.StringLiteral) {
      return {
        type: "StringLiteral",
        value: unquote(ctx.StringLiteral[0].image),
      } as StringLiteral;
    } else if (ctx.NumberLiteral) {
      return {
        type: "NumberLiteral",
        value: ctx.NumberLiteral[0].image,
      } as NumberLiteral;
    }
    throw new Error("Unknown literal type");
  }

  arrayLiteral(ctx: ArrayLiteralCstNode["children"]): ArrayLiteral {
    return {
      type: "ArrayLiteral",
      elements: ctx.expression ? ctx.expression.map((expr) => this.visit(expr)) : [],
    };
  }

  objectLiteral(ctx: ObjectLiteralCstNode["children"]): ObjectLiteral {
    const properties: ObjectProperty[] = [];

    if (ctx.kvPair) {
      for (const kvPairCst of ctx.kvPair) {
        properties.push(this.visit(kvPairCst));
      }
    }

    return {
      type: "ObjectLiteral",
      properties,
    };
  }

  kvPair(ctx: KvPairCstNode["children"]): ObjectProperty {
    // First StringLiteral is the key
    const key = unquote(ctx.StringLiteral[0].image);

    // Value can be second StringLiteral or NumberLiteral
    let value: string;
    if (ctx.StringLiteral.length > 1) {
      value = unquote(ctx.StringLiteral[1].image);
    } else if (ctx.NumberLiteral) {
      value = ctx.NumberLiteral[0].image;
    } else {
      throw new Error("KV pair must have a value");
    }

    return { key, value };
  }
}

// Singleton visitor instance
export const visitorInstance = new DataflowAstVisitor();
