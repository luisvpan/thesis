import { CstParser } from "chevrotain";
import {
  allTokens,
  Source,
  Transform,
  Output,
  Natural,
  Integer,
  Decimal,
  Fraction,
  Text,
  Boolean,
  Shape,
  Car,
  Food,
  Animal,
  Person,
  Set,
  Stream,
  Add,
  Subtract,
  Multiply,
  Divide,
  Compare,
  Filter,
  Union,
  Intersection,
  Difference,
  Complement,
  Next,
  First,
  Fby,
  Accumulate,
  Sensor,
  Generator,
  External,
  Sort,
  AlphabeticalSort,
  And,
  Or,
  Not,
  CompareBySize,
  CompareByColor,
  CompareByType,
  CompareByTaste,
  CompareByAgeGroup,
  CompareByGender,
  FilterBySize,
  FilterByColor,
  FilterByType,
  FilterByTaste,
  FilterByAgeGroup,
  FilterByGender,
  True,
  False,
  Equals,
  Colon,
  Semicolon,
  LParen,
  RParen,
  LBrace,
  RBrace,
  LBracket,
  RBracket,
  Comma,
  AngleLeft,
  AngleRight,
  Dot,
  Slash,
  Minus,
  NumberLiteral,
  StringLiteral,
  Identifier
} from "../lexer/tokens.js";

export class DataflowParser extends CstParser {
  constructor() {
    super(allTokens);
    this.performSelfAnalysis();
  }

  program = this.RULE("program", () => {
    this.MANY(() => this.SUBRULE(this.statement));
  });

  statement = this.RULE("statement", () => {
    this.OR([
      { ALT: () => this.SUBRULE(this.sourceStatement) },
      { ALT: () => this.SUBRULE(this.transformStatement) },
      { ALT: () => this.SUBRULE(this.outputStatement) }
    ]);
  });

  sourceStatement = this.RULE("sourceStatement", () => {
    this.CONSUME(Source);
    const id = this.CONSUME(Identifier);
    this.CONSUME(Colon);
    this.SUBRULE(this.typeDeclaration);
    this.CONSUME(Equals);
    this.SUBRULE(this.value);
    this.CONSUME(Semicolon);
  });

  transformStatement = this.RULE("transformStatement", () => {
    this.CONSUME(Transform);
    const id = this.CONSUME(Identifier);
    this.CONSUME(Colon);
    this.SUBRULE(this.typeDeclaration);
    this.CONSUME(Equals);
    this.SUBRULE(this.operationExpression);
    this.CONSUME(Semicolon);
  });

  outputStatement = this.RULE("outputStatement", () => {
    this.CONSUME(Output);
    this.CONSUME1(Identifier);
    this.CONSUME(Colon);
    this.SUBRULE(this.typeDeclaration);
    this.CONSUME(Equals);
    this.CONSUME2(Identifier);
    this.CONSUME(Semicolon);
  });

  typeDeclaration = this.RULE("typeDeclaration", () => {
    this.OR([
      { ALT: () => this.CONSUME(Natural) },
      { ALT: () => this.CONSUME(Integer) },
      { ALT: () => this.CONSUME(Decimal) },
      { ALT: () => this.CONSUME(Fraction) },
      { ALT: () => this.CONSUME(Text) },
      { ALT: () => this.CONSUME(Boolean) },
      { ALT: () => this.CONSUME(Shape) },
      { ALT: () => this.CONSUME(Car) },
      { ALT: () => this.CONSUME(Food) },
      { ALT: () => this.CONSUME(Animal) },
      { ALT: () => this.CONSUME(Person) },
      { ALT: () => this.SUBRULE(this.setType) },
      { ALT: () => this.SUBRULE(this.streamType) }
    ]);
  });

  setType = this.RULE("setType", () => {
    this.CONSUME(Set);
    this.CONSUME(AngleLeft);
    this.SUBRULE(this.typeDeclaration);
    this.CONSUME(AngleRight);
  });

  streamType = this.RULE("streamType", () => {
    this.CONSUME(Stream);
    this.CONSUME(AngleLeft);
    this.SUBRULE(this.typeDeclaration);
    this.CONSUME(AngleRight);
  });

  value = this.RULE("value", () => {
    this.OR([
      { ALT: () => this.SUBRULE(this.literal) },
      {
        ALT: () => this.SUBRULE(this.objectLiteral),
        GATE: () => this.LA(2).tokenType === Identifier
      },
      { ALT: () => this.SUBRULE(this.arrayLiteral) },
      { ALT: () => this.SUBRULE(this.setLiteral) },
      { ALT: () => this.SUBRULE(this.streamLiteral) }
    ]);
  });

  literal = this.RULE("literal", () => {
    this.OR([
      { ALT: () => this.SUBRULE(this.fractionLiteral) },
      { ALT: () => this.CONSUME(NumberLiteral) },
      { ALT: () => this.CONSUME(StringLiteral) },
      { ALT: () => this.CONSUME(True) },
      { ALT: () => this.CONSUME(False) }
    ]);
  });

  objectLiteral = this.RULE("objectLiteral", () => {
    this.CONSUME(LBrace);
    this.MANY_SEP({
      SEP: Comma,
      DEF: () => {
        this.CONSUME(Identifier);
        this.CONSUME(Colon);
        this.SUBRULE(this.literal);
      }
    });
    this.CONSUME(RBrace);
  });

  arrayLiteral = this.RULE("arrayLiteral", () => {
    this.CONSUME(LBracket);
    this.MANY_SEP({
      SEP: Comma,
      DEF: () => this.SUBRULE(this.value)
    });
    this.CONSUME(RBracket);
  });

  setLiteral = this.RULE("setLiteral", () => {
    this.CONSUME(LBrace);
    this.MANY_SEP({
      SEP: Comma,
      DEF: () => this.SUBRULE(this.value)
    });
    this.CONSUME(RBrace);
  });

  streamLiteral = this.RULE("streamLiteral", () => {
    this.CONSUME(Stream);
    this.CONSUME(AngleLeft);
    this.SUBRULE(this.typeDeclaration);
    this.CONSUME(AngleRight);
    this.CONSUME(LParen);
    this.OR([
      { ALT: () => this.SUBRULE(this.sensorSource) },
      { ALT: () => this.SUBRULE(this.generatorSource) },
      { ALT: () => this.SUBRULE(this.externalSource) }
    ]);
    this.CONSUME(RParen);
  });

  fractionLiteral = this.RULE("fractionLiteral", () => {
    this.CONSUME1(NumberLiteral);
    this.CONSUME(Slash);
    this.CONSUME2(NumberLiteral);
  });

  sensorSource = this.RULE("sensorSource", () => {
    this.CONSUME(Sensor);
    this.CONSUME(LParen);
    this.CONSUME(StringLiteral);
    this.CONSUME(RParen);
  });

  generatorSource = this.RULE("generatorSource", () => {
    this.CONSUME(Generator);
    this.CONSUME(LParen);
    this.CONSUME(Identifier);
    this.CONSUME(RParen);
  });

  externalSource = this.RULE("externalSource", () => {
    this.CONSUME(External);
    this.CONSUME(LParen);
    this.CONSUME(StringLiteral);
    this.CONSUME(RParen);
  });

  operationExpression = this.RULE("operationExpression", () => {
    this.SUBRULE(this.operationName);
    this.CONSUME(LParen);
    this.OPTION(() => {
      this.SUBRULE(this.argumentList);
    });
    this.CONSUME(RParen);
  });

  operationName = this.RULE("operationName", () => {
    this.OR([
      { ALT: () => this.CONSUME(Add) },
      { ALT: () => this.CONSUME(Subtract) },
      { ALT: () => this.CONSUME(Multiply) },
      { ALT: () => this.CONSUME(Divide) },
      { ALT: () => this.CONSUME(Compare) },
      { ALT: () => this.CONSUME(Filter) },
      { ALT: () => this.CONSUME(Union) },
      { ALT: () => this.CONSUME(Intersection) },
      { ALT: () => this.CONSUME(Difference) },
      { ALT: () => this.CONSUME(Complement) },
      { ALT: () => this.CONSUME(Next) },
      { ALT: () => this.CONSUME(First) },
      { ALT: () => this.CONSUME(Fby) },
      { ALT: () => this.CONSUME(Accumulate) },
      { ALT: () => this.CONSUME(Sort) },
      { ALT: () => this.CONSUME(AlphabeticalSort) },
      { ALT: () => this.CONSUME(And) },
      { ALT: () => this.CONSUME(Or) },
      { ALT: () => this.CONSUME(Not) },
      { ALT: () => this.CONSUME(CompareBySize) },
      { ALT: () => this.CONSUME(CompareByColor) },
      { ALT: () => this.CONSUME(CompareByType) },
      { ALT: () => this.CONSUME(CompareByTaste) },
      { ALT: () => this.CONSUME(CompareByAgeGroup) },
      { ALT: () => this.CONSUME(CompareByGender) },
      { ALT: () => this.CONSUME(FilterBySize) },
      { ALT: () => this.CONSUME(FilterByColor) },
      { ALT: () => this.CONSUME(FilterByType) },
      { ALT: () => this.CONSUME(FilterByTaste) },
      { ALT: () => this.CONSUME(FilterByAgeGroup) },
      { ALT: () => this.CONSUME(FilterByGender) },
      { ALT: () => this.CONSUME(Identifier) }
    ]);
  });

  argumentList = this.RULE("argumentList", () => {
    this.MANY_SEP({
      SEP: Comma,
      DEF: () => this.SUBRULE(this.argument)
    });
  });

  argument = this.RULE("argument", () => {
    this.OR([
      { GATE: () => this.LA(2).tokenType === LParen, ALT: () => this.SUBRULE(this.operationExpression) },
      { ALT: () => this.CONSUME(Identifier) },
      { ALT: () => this.SUBRULE(this.literal) }
    ]);
  });
}
