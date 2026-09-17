import type { CstNode, IToken } from "chevrotain";
import type {
  CriterionSubtype,
  DataLiteral,
  Expression,
  GroupLiteral,
  Literal,
  ObjectLiteral,
  ObjectProperty,
  Program,
  SinkStatement,
  SourceStatement,
  Statement,
  TransformStatement,
} from "./ast";
import { isCPACategory } from "./ast";
import { parserInstance } from "./parser";
import { syntaxError } from "../runtime/errors";

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
    /** [0] es el nombre del nodo; [1], si está, el de la operación. */
    Identifier: IToken[];
    argumentList?: CstNode[];
  };
}

interface SinkStatementCstNode extends CstNode {
  children: {
    Identifier: IToken[];
  };
}

interface ArgumentListCstNode extends CstNode {
  children: {
    Identifier: IToken[];
  };
}

interface LiteralCstNode extends CstNode {
  children: {
    objectLiteral?: CstNode[];
    group?: CstNode[];
  };
}

interface GroupCstNode extends CstNode {
  children: {
    LBracket: IToken[];
    objectLiteral?: CstNode[];
  };
}

interface ObjectLiteralCstNode extends CstNode {
  children: {
    LBrace: IToken[];
    kvPair?: CstNode[];
  };
}

interface KvPairCstNode extends CstNode {
  children: {
    StringLiteral: IToken[];
    NumberLiteral?: IToken[];
    kvArrayLiteral?: CstNode[];
  };
}

interface KvArrayLiteralCstNode extends CstNode {
  children: {
    StringLiteral?: IToken[];
  };
}

function unquote(str: string): string {
  return str.startsWith('"') && str.endsWith('"') ? str.slice(1, -1) : str;
}

function positionOf(token: IToken): { line?: number; column?: number } {
  return { line: token.startLine, column: token.startColumn };
}

const CRITERION_SUBTYPES: readonly string[] = ["filter", "order"];

const BaseCstVisitor = parserInstance.getBaseCstVisitorConstructor();

export class DataflowAstVisitor extends BaseCstVisitor {
  constructor() {
    super();
    this.validateVisitor();
  }

  program(ctx: ProgramCstNode["children"]): Program {
    return {
      type: "Program",
      statements: (ctx.statement ?? []).map((statement) => this.visit(statement) as Statement),
    };
  }

  statement(ctx: StatementCstNode["children"]): Statement {
    if (ctx.sourceStatement) return this.visit(ctx.sourceStatement[0]);
    if (ctx.transformStatement) return this.visit(ctx.transformStatement[0]);
    if (ctx.sinkStatement) return this.visit(ctx.sinkStatement[0]);
    throw syntaxError("Sentencia desconocida");
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
      operation: ctx.Identifier[1]?.image,
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

  argumentList(ctx: ArgumentListCstNode["children"]): Expression[] {
    return ctx.Identifier.map((token) => ({ type: "Identifier" as const, name: token.image }));
  }

  literal(ctx: LiteralCstNode["children"]): Literal {
    if (ctx.objectLiteral) return this.visit(ctx.objectLiteral[0]);
    if (ctx.group) return this.visit(ctx.group[0]);
    throw syntaxError("Literal desconocido");
  }

  group(ctx: GroupCstNode["children"]): GroupLiteral {
    const elements = (ctx.objectLiteral ?? []).map((element) => this.visit(element) as ObjectLiteral);

    // Los grupos son solo de datos: los criterios no se agrupan (§4.1).
    for (const element of elements) {
      if (element.type !== "DataLiteral") {
        throw syntaxError(
          "Un grupo reúne objetos de datos; los criterios no se agrupan (cada uno va en su propio source)",
          positionOf(ctx.LBracket[0])
        );
      }
    }

    return { type: "GroupLiteral", elements: elements as DataLiteral[] };
  }

  objectLiteral(ctx: ObjectLiteralCstNode["children"]): ObjectLiteral {
    const properties: ObjectProperty[] = (ctx.kvPair ?? []).map((pair) => this.visit(pair));
    const at = positionOf(ctx.LBrace[0]);

    const sourceType = properties.find((property) => property.key === "sourceType")?.value;

    if (typeof sourceType === "string" && CRITERION_SUBTYPES.includes(sourceType)) {
      const declared = properties.find((property) => property.key === "properties")?.value;

      return {
        type: "CriterionLiteral",
        sourceType: sourceType as CriterionSubtype,
        properties: Array.isArray(declared) ? declared : [],
        values: properties.filter((property) => !["sourceType", "properties"].includes(property.key)),
      };
    }

    if (sourceType !== undefined && sourceType !== "data") {
      throw syntaxError(
        `"sourceType" admite "data", "filter" u "order"; se escribió "${String(sourceType)}"`,
        at
      );
    }

    const text = (key: string): string => {
      const value = properties.find((property) => property.key === key)?.value;
      return typeof value === "string" ? value : "";
    };

    const category = text("category");
    if (category !== "" && !isCPACategory(category)) {
      throw syntaxError(
        `"category" admite "abstracto", "pictorico" o "concreto"; se escribió "${category}"`,
        at
      );
    }

    return {
      type: "DataLiteral",
      sourceType: "data",
      category,
      objType: text("type"),
      subtype: text("subtype"),
      // La gramática exige `quantity`; omitirla se tolera como 1 por leniencia.
      quantity: text("quantity") || "1",
      attributes: properties.filter(
        (property) => !["sourceType", "category", "type", "subtype", "quantity"].includes(property.key)
      ),
    };
  }

  kvPair(ctx: KvPairCstNode["children"]): ObjectProperty {
    const key = unquote(ctx.StringLiteral[0].image);

    if (ctx.kvArrayLiteral) {
      return { key, value: this.visit(ctx.kvArrayLiteral[0]) };
    }
    if (ctx.StringLiteral.length > 1) {
      return { key, value: unquote(ctx.StringLiteral[1].image) };
    }
    if (ctx.NumberLiteral) {
      return { key, value: ctx.NumberLiteral[0].image };
    }

    throw syntaxError(`La propiedad "${key}" no tiene valor`, positionOf(ctx.StringLiteral[0]));
  }

  kvArrayLiteral(ctx: KvArrayLiteralCstNode["children"]): string[] {
    return (ctx.StringLiteral ?? []).map((token) => unquote(token.image));
  }
}

// Singleton visitor instance
export const visitorInstance = new DataflowAstVisitor();
