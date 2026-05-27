import { CstParser } from "chevrotain";
import {
  allTokens,
  Source,
  Transform,
  Sink,
  Sum,
  Substract,
  Multiply,
  Divide,
  LessThan,
  GreaterThan,
  OrderAsc,
  OrderDesc,
  Filter,
  Equals,
  Semicolon,
  Comma,
  Colon,
  LParen,
  RParen,
  LBrace,
  RBrace,
  LBracket,
  RBracket,
  StringLiteral,
  NumberLiteral,
  Identifier,
} from "./lexer";

export class DataflowParser extends CstParser {
  constructor() {
    super(allTokens);
    this.performSelfAnalysis();
  }

  // program ::= statement*
  public program = this.RULE("program", () => {
    this.MANY(() => {
      this.SUBRULE(this.statement);
    });
  });

  // statement ::= source_statement | transform_statement | sink_statement
  private statement = this.RULE("statement", () => {
    this.OR([
      { ALT: () => this.SUBRULE(this.sourceStatement) },
      { ALT: () => this.SUBRULE(this.transformStatement) },
      { ALT: () => this.SUBRULE(this.sinkStatement) },
    ]);
  });

  // source_statement ::= "source" identifier "=" literal? ";"
  private sourceStatement = this.RULE("sourceStatement", () => {
    this.CONSUME(Source);
    this.CONSUME(Identifier);
    this.CONSUME(Equals);
    this.OPTION(() => {
      this.SUBRULE(this.literal);
    });
    this.CONSUME(Semicolon);
  });

  // transform_statement ::= "transform" identifier "=" (operation "(" argument_list? ")")? ";"
  private transformStatement = this.RULE("transformStatement", () => {
    this.CONSUME(Transform);
    this.CONSUME(Identifier);
    this.CONSUME(Equals);
    this.OPTION(() => {
      this.SUBRULE(this.operation);
      this.CONSUME(LParen);
      this.OPTION2(() => {
        this.SUBRULE(this.argumentList);
      });
      this.CONSUME(RParen);
    });
    this.CONSUME(Semicolon);
  });

  // sink_statement ::= "sink" identifier "=" identifier? ";"
  private sinkStatement = this.RULE("sinkStatement", () => {
    this.CONSUME(Sink);
    this.CONSUME1(Identifier);
    this.CONSUME(Equals);
    this.OPTION(() => {
      this.CONSUME2(Identifier);
    });
    this.CONSUME(Semicolon);
  });

  // operation ::= "sum" | "substract" | "multiply" | "divide" | "less_than" | "greater_than" | "order_asc" | "order_desc" | "filter"
  private operation = this.RULE("operation", () => {
    this.OR([
      { ALT: () => this.CONSUME(Sum) },
      { ALT: () => this.CONSUME(Substract) },
      { ALT: () => this.CONSUME(Multiply) },
      { ALT: () => this.CONSUME(Divide) },
      { ALT: () => this.CONSUME(LessThan) },
      { ALT: () => this.CONSUME(GreaterThan) },
      { ALT: () => this.CONSUME(OrderAsc) },
      { ALT: () => this.CONSUME(OrderDesc) },
      { ALT: () => this.CONSUME(Filter) },
    ]);
  });

  // argument_list ::= identifier ("," identifier)*
  private argumentList = this.RULE("argumentList", () => {
    this.SUBRULE(this.expression);
    this.MANY(() => {
      this.CONSUME(Comma);
      this.SUBRULE2(this.expression);
    });
  });

  // expression ::= identifier | literal
  private expression = this.RULE("expression", () => {
    this.OR([
      { ALT: () => this.CONSUME(Identifier) },
      { ALT: () => this.SUBRULE(this.literal) },
    ]);
  });

  // literal ::= object_literal | group | array_literal | string_literal
  // Note: NumberLiteral is only allowed inside object kvPairs (for quantity values)
  // Note: group must come before arrayLiteral to prioritize object-only arrays
  private literal = this.RULE("literal", () => {
    this.OR([
      { ALT: () => this.SUBRULE(this.objectLiteral) },
      {
        GATE: () => this.isGroupAhead(),
        ALT: () => this.SUBRULE(this.group),
      },
      { ALT: () => this.SUBRULE(this.arrayLiteral) },
      { ALT: () => this.CONSUME(StringLiteral) },
    ]);
  });

  // Helper to check if we're looking at a group (array of objects only)
  private isGroupAhead(): boolean {
    // Look ahead: [ followed by { or ]
    const tokens = this.LA(1);
    if (tokens.tokenType !== LBracket) return false;
    const next = this.LA(2);
    return next.tokenType === LBrace || next.tokenType === RBracket;
  }

  // group ::= "[" (object_literal ("," object_literal)*)? "]"
  private group = this.RULE("group", () => {
    this.CONSUME(LBracket);
    this.OPTION(() => {
      this.SUBRULE(this.objectLiteral);
      this.MANY(() => {
        this.CONSUME(Comma);
        this.SUBRULE2(this.objectLiteral);
      });
    });
    this.CONSUME(RBracket);
  });

  // array_literal ::= "[" (expression ("," expression)*)? "]"
  private arrayLiteral = this.RULE("arrayLiteral", () => {
    this.CONSUME(LBracket);
    this.OPTION(() => {
      this.SUBRULE(this.expression);
      this.MANY(() => {
        this.CONSUME(Comma);
        this.SUBRULE2(this.expression);
      });
    });
    this.CONSUME(RBracket);
  });

  // object_literal ::= "{" (kv_pair ("," kv_pair)*)? "}"
  // Flexible: parses any key-value pairs where key is string and value is string or number
  private objectLiteral = this.RULE("objectLiteral", () => {
    this.CONSUME(LBrace);
    this.OPTION(() => {
      this.SUBRULE(this.kvPair);
      this.MANY(() => {
        this.CONSUME(Comma);
        this.SUBRULE2(this.kvPair);
      });
    });
    this.CONSUME(RBrace);
  });

  // kv_pair ::= string_literal ":" kv_value
  // kv_value ::= string_literal | rational_literal | array_literal
  private kvPair = this.RULE("kvPair", () => {
    this.CONSUME(StringLiteral);
    this.CONSUME(Colon);
    this.OR([
      { ALT: () => this.SUBRULE(this.kvArrayLiteral) },  // ["a", "b", "c"]
      { ALT: () => this.CONSUME2(StringLiteral) },
      { ALT: () => this.CONSUME(NumberLiteral) },
    ]);
  });

  // array_literal for kv_value ::= "[" (string_literal ("," string_literal)*)? "]"
  private kvArrayLiteral = this.RULE("kvArrayLiteral", () => {
    this.CONSUME(LBracket);
    this.OPTION(() => {
      this.CONSUME(StringLiteral);
      this.MANY(() => {
        this.CONSUME(Comma);
        this.CONSUME2(StringLiteral);
      });
    });
    this.CONSUME(RBracket);
  });
}

// Singleton parser instance
export const parserInstance = new DataflowParser();
