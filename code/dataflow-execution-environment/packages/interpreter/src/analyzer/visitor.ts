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
  OtherLiteral,
  ArrayLiteral,
  NumberLiteral,
  CategoryValue,
  TypeValue,
  ShapeTypeValue,
  SizeValue,
  FoodTypeValue,
  ColorValue,
  SimpleObjectLiteral,
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
    literal: CstNode[];
  };
}

interface TransformStatementCstNode extends CstNode {
  children: {
    Identifier: IToken[];
    operation: CstNode[];
    argumentList: CstNode[];
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
    numberLiteral?: CstNode[];
    otherLiteral?: CstNode[];
  };
}

interface NumberLiteralCstNode extends CstNode {
  children: {
    NumberLiteral: IToken[];
  };
}

interface ArrayLiteralCstNode extends CstNode {
  children: {
    expression: CstNode[];
  };
}

interface ObjectLiteralCstNode extends CstNode {
  children: {
    objectProperties: CstNode[];
  };
}

interface ObjectPropertiesCstNode extends CstNode {
  children: {
    objectProperty: CstNode[];
  };
}

interface ObjectPropertyCstNode extends CstNode {
  children: {
    categoryProperty?: CstNode[];
    typeProperty?: CstNode[];
    valueProperty?: CstNode[];
    subtypeProperty?: CstNode[];
    sizeProperty?: CstNode[];
    amountProperty?: CstNode[];
    colorProperty?: CstNode[];
  };
}

interface CategoryPropertyCstNode extends CstNode {
  children: {
    categoryValue: CstNode[];
  };
}

interface TypePropertyCstNode extends CstNode {
  children: {
    typeValue: CstNode[];
  };
}

interface ValuePropertyCstNode extends CstNode {
  children: {
    numberLiteral: CstNode[];
  };
}

interface SubtypePropertyCstNode extends CstNode {
  children: {
    subtypeValue: CstNode[];
  };
}

interface SizePropertyCstNode extends CstNode {
  children: {
    sizeValue: CstNode[];
  };
}

interface AmountPropertyCstNode extends CstNode {
  children: {
    numberLiteral: CstNode[];
  };
}

interface ColorPropertyCstNode extends CstNode {
  children: {
    colorValue: CstNode[];
  };
}

interface CategoryValueCstNode extends CstNode {
  children: {
    Abstract?: IToken[];
    Pictorial?: IToken[];
    Concrete?: IToken[];
  };
}

interface TypeValueCstNode extends CstNode {
  children: {
    RationalType?: IToken[];
    ShapeType?: IToken[];
    FoodType?: IToken[];
    MontessoriType?: IToken[];
    // Shape subtypes
    Circle?: IToken[];
    Square?: IToken[];
    Triangle?: IToken[];
    Rectangle?: IToken[];
    Diamond?: IToken[];
    Star?: IToken[];
    Trapezoid?: IToken[];
    // Food subtypes
    Grape?: IToken[];
    Pear?: IToken[];
    Apple?: IToken[];
    Burger?: IToken[];
    Pasta?: IToken[];
  };
}

interface SubtypeValueCstNode extends CstNode {
  children: {
    // Shape subtypes
    Circle?: IToken[];
    Square?: IToken[];
    Triangle?: IToken[];
    Rectangle?: IToken[];
    Diamond?: IToken[];
    Star?: IToken[];
    Trapezoid?: IToken[];
    // Food subtypes
    Grape?: IToken[];
    Pear?: IToken[];
    Apple?: IToken[];
    Burger?: IToken[];
    Pasta?: IToken[];
  };
}

interface SizeValueCstNode extends CstNode {
  children: {
    Small?: IToken[];
    Medium?: IToken[];
    Large?: IToken[];
  };
}

interface ColorValueCstNode extends CstNode {
  children: {
    Purple?: IToken[];
    Green?: IToken[];
    Red?: IToken[];
    Orange?: IToken[];
    Blue?: IToken[];
    Yellow?: IToken[];
  };
}

interface OtherLiteralCstNode extends CstNode {
  children: {
    // Categories
    Abstract?: IToken[];
    Pictorial?: IToken[];
    Concrete?: IToken[];
    // Types
    RationalType?: IToken[];
    ShapeType?: IToken[];
    FoodType?: IToken[];
    MontessoriType?: IToken[];
    // Shape subtypes
    Circle?: IToken[];
    Square?: IToken[];
    Triangle?: IToken[];
    Rectangle?: IToken[];
    Diamond?: IToken[];
    Star?: IToken[];
    Trapezoid?: IToken[];
    // Size values
    Small?: IToken[];
    Medium?: IToken[];
    Large?: IToken[];
    // Food subtypes
    Grape?: IToken[];
    Pear?: IToken[];
    Apple?: IToken[];
    Burger?: IToken[];
    Pasta?: IToken[];
    // Color values
    Purple?: IToken[];
    Green?: IToken[];
    Red?: IToken[];
    Orange?: IToken[];
    Blue?: IToken[];
    Yellow?: IToken[];
  };
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
      value: this.visit(ctx.literal[0]),
    };
  }

  transformStatement(ctx: TransformStatementCstNode["children"]): TransformStatement {
    return {
      type: "TransformStatement",
      identifier: ctx.Identifier[0].image,
      operation: this.visit(ctx.operation[0]),
      arguments: this.visit(ctx.argumentList[0]),
    };
  }

  sinkStatement(ctx: SinkStatementCstNode["children"]): SinkStatement {
    return {
      type: "SinkStatement",
      identifier: ctx.Identifier[0].image,
      sourceIdentifier: ctx.Identifier[1].image,
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
    } else if (ctx.numberLiteral) {
      return this.visit(ctx.numberLiteral[0]);
    } else if (ctx.otherLiteral) {
      return this.visit(ctx.otherLiteral[0]);
    }
    throw new Error("Unknown literal type");
  }

  numberLiteral(ctx: NumberLiteralCstNode["children"]): NumberLiteral {
    return {
      type: "NumberLiteral",
      value: ctx.NumberLiteral[0].image,
    };
  }

  arrayLiteral(ctx: ArrayLiteralCstNode["children"]): ArrayLiteral {
    return {
      type: "ArrayLiteral",
      elements: ctx.expression.map((expr) => this.visit(expr)),
    };
  }

  objectLiteral(ctx: ObjectLiteralCstNode["children"]): ObjectLiteral {
    const properties = this.visit(ctx.objectProperties[0]) as Record<string, unknown>;

    return {
      type: "ObjectLiteral",
      ...properties,
    } as SimpleObjectLiteral;
  }

  objectProperties(ctx: ObjectPropertiesCstNode["children"]): Record<string, unknown> {
    const result: Record<string, unknown> = {};

    for (const prop of ctx.objectProperty) {
      const propValue = this.visit(prop) as { key: string; value: unknown };
      result[propValue.key] = propValue.value;
    }

    return result;
  }

  objectProperty(ctx: ObjectPropertyCstNode["children"]): { key: string; value: unknown } {
    if (ctx.categoryProperty) {
      return { key: "category", value: this.visit(ctx.categoryProperty[0]) };
    } else if (ctx.typeProperty) {
      return { key: "objectType", value: this.visit(ctx.typeProperty[0]) };
    } else if (ctx.valueProperty) {
      return { key: "value", value: this.visit(ctx.valueProperty[0]) };
    } else if (ctx.subtypeProperty) {
      return { key: "subtype", value: this.visit(ctx.subtypeProperty[0]) };
    } else if (ctx.sizeProperty) {
      return { key: "size", value: this.visit(ctx.sizeProperty[0]) };
    } else if (ctx.amountProperty) {
      return { key: "amount", value: this.visit(ctx.amountProperty[0]) };
    } else if (ctx.colorProperty) {
      return { key: "color", value: this.visit(ctx.colorProperty[0]) };
    }
    throw new Error("Unknown object property");
  }

  categoryProperty(ctx: CategoryPropertyCstNode["children"]): CategoryValue {
    return this.visit(ctx.categoryValue[0]);
  }

  typeProperty(ctx: TypePropertyCstNode["children"]): TypeValue | ShapeTypeValue | FoodTypeValue {
    return this.visit(ctx.typeValue[0]);
  }

  valueProperty(ctx: ValuePropertyCstNode["children"]): string {
    const numLiteral = this.visit(ctx.numberLiteral[0]) as NumberLiteral;
    return numLiteral.value;
  }

  subtypeProperty(ctx: SubtypePropertyCstNode["children"]): ShapeTypeValue | FoodTypeValue {
    return this.visit(ctx.subtypeValue[0]);
  }

  sizeProperty(ctx: SizePropertyCstNode["children"]): SizeValue {
    return this.visit(ctx.sizeValue[0]);
  }

  amountProperty(ctx: AmountPropertyCstNode["children"]): string {
    const numLiteral = this.visit(ctx.numberLiteral[0]) as NumberLiteral;
    return numLiteral.value;
  }

  colorProperty(ctx: ColorPropertyCstNode["children"]): ColorValue {
    return this.visit(ctx.colorValue[0]);
  }

  categoryValue(ctx: CategoryValueCstNode["children"]): CategoryValue {
    if (ctx.Abstract) return "abstracto";
    if (ctx.Pictorial) return "pictorico";
    if (ctx.Concrete) return "concreto";
    throw new Error("Unknown category value");
  }

  // v2.1.0: Spanish types
  typeValue(ctx: TypeValueCstNode["children"]): TypeValue | ShapeTypeValue | FoodTypeValue {
    if (ctx.RationalType) return "racional";
    if (ctx.ShapeType) return "forma";
    if (ctx.FoodType) return "comida";
    if (ctx.MontessoriType) return "montessori";
    // Shape subtypes
    if (ctx.Circle) return "circulo";
    if (ctx.Square) return "cuadrado";
    if (ctx.Triangle) return "triangulo";
    if (ctx.Rectangle) return "rectangulo";
    if (ctx.Diamond) return "rombo";
    if (ctx.Star) return "estrella";
    if (ctx.Trapezoid) return "trapecio";
    // Food subtypes
    if (ctx.Grape) return "uva";
    if (ctx.Pear) return "pera";
    if (ctx.Apple) return "manzana";
    if (ctx.Burger) return "hamburguesa";
    if (ctx.Pasta) return "pasta";
    throw new Error("Unknown type value");
  }

  subtypeValue(ctx: SubtypeValueCstNode["children"]): ShapeTypeValue | FoodTypeValue {
    // Shape subtypes
    if (ctx.Circle) return "circulo";
    if (ctx.Square) return "cuadrado";
    if (ctx.Triangle) return "triangulo";
    if (ctx.Rectangle) return "rectangulo";
    if (ctx.Diamond) return "rombo";
    if (ctx.Star) return "estrella";
    if (ctx.Trapezoid) return "trapecio";
    // Food subtypes
    if (ctx.Grape) return "uva";
    if (ctx.Pear) return "pera";
    if (ctx.Apple) return "manzana";
    if (ctx.Burger) return "hamburguesa";
    if (ctx.Pasta) return "pasta";
    throw new Error("Unknown subtype value");
  }

  sizeValue(ctx: SizeValueCstNode["children"]): SizeValue {
    if (ctx.Small) return "pequeño";
    if (ctx.Medium) return "mediano";
    if (ctx.Large) return "grande";
    throw new Error("Unknown size value");
  }

  colorValue(ctx: ColorValueCstNode["children"]): ColorValue {
    if (ctx.Purple) return "morado";
    if (ctx.Green) return "verde";
    if (ctx.Red) return "rojo";
    if (ctx.Orange) return "naranja";
    if (ctx.Blue) return "azul";
    if (ctx.Yellow) return "amarillo";
    throw new Error("Unknown color value");
  }

  otherLiteral(ctx: OtherLiteralCstNode["children"]): OtherLiteral {
    let value: OtherLiteral["value"];

    // Categories
    if (ctx.Abstract) value = "abstracto";
    else if (ctx.Pictorial) value = "pictorico";
    else if (ctx.Concrete) value = "concreto";
    // Types (Spanish)
    else if (ctx.RationalType) value = "racional";
    else if (ctx.ShapeType) value = "forma";
    else if (ctx.FoodType) value = "comida";
    else if (ctx.MontessoriType) value = "montessori";
    // Shape subtypes
    else if (ctx.Circle) value = "circulo";
    else if (ctx.Square) value = "cuadrado";
    else if (ctx.Triangle) value = "triangulo";
    else if (ctx.Rectangle) value = "rectangulo";
    else if (ctx.Diamond) value = "rombo";
    else if (ctx.Star) value = "estrella";
    else if (ctx.Trapezoid) value = "trapecio";
    // Size values
    else if (ctx.Small) value = "pequeño";
    else if (ctx.Medium) value = "mediano";
    else if (ctx.Large) value = "grande";
    // Food subtypes
    else if (ctx.Grape) value = "uva";
    else if (ctx.Pear) value = "pera";
    else if (ctx.Apple) value = "manzana";
    else if (ctx.Burger) value = "hamburguesa";
    else if (ctx.Pasta) value = "pasta";
    // Color values
    else if (ctx.Purple) value = "morado";
    else if (ctx.Green) value = "verde";
    else if (ctx.Red) value = "rojo";
    else if (ctx.Orange) value = "naranja";
    else if (ctx.Blue) value = "azul";
    else if (ctx.Yellow) value = "amarillo";
    else throw new Error("Unknown other literal value");

    return {
      type: "OtherLiteral",
      value,
    };
  }
}

// Singleton visitor instance
export const visitorInstance = new DataflowAstVisitor();
