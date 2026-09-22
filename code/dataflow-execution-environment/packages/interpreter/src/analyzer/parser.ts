// Parser — LANGUAGE_SPEC.md §5.1
//
// Las tres declaraciones tienen su valor opcional: un nodo a medio escribir es
// sintácticamente válido y evalúa a `nulo` (§2.5).

import { CstParser } from "chevrotain";
import {
  allTokens,
  Colon,
  Comma,
  Equals,
  Identifier,
  LBrace,
  LBracket,
  LParen,
  NumberLiteral,
  RBrace,
  RBracket,
  RParen,
  Semicolon,
  Sink,
  Source,
  StringLiteral,
  Transform,
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

  // statement ::= source_decl | transform_decl | sink_decl
  private statement = this.RULE("statement", () => {
    this.OR([
      { ALT: () => this.SUBRULE(this.sourceStatement) },
      { ALT: () => this.SUBRULE(this.transformStatement) },
      { ALT: () => this.SUBRULE(this.sinkStatement) },
    ]);
  });

  // source_decl ::= "source" identifier "=" (object_literal | group)? ";"
  private sourceStatement = this.RULE("sourceStatement", () => {
    this.CONSUME(Source);
    this.CONSUME(Identifier);
    this.CONSUME(Equals);
    this.OPTION(() => {
      this.SUBRULE(this.literal);
    });
    this.CONSUME(Semicolon);
  });

  // transform_decl ::= "transform" identifier "=" (operation "(" argument_list? ")")? ";"
  // operation ::= identifier
  private transformStatement = this.RULE("transformStatement", () => {
    this.CONSUME(Transform);
    this.CONSUME(Identifier);
    this.CONSUME(Equals);
    this.OPTION(() => {
      this.CONSUME2(Identifier);
      this.CONSUME(LParen);
      this.OPTION2(() => {
        this.SUBRULE(this.argumentList);
      });
      this.CONSUME(RParen);
    });
    this.CONSUME(Semicolon);
  });

  // sink_decl ::= "sink" identifier "=" identifier? ";"
  private sinkStatement = this.RULE("sinkStatement", () => {
    this.CONSUME(Sink);
    this.CONSUME1(Identifier);
    this.CONSUME(Equals);
    this.OPTION(() => {
      this.CONSUME2(Identifier);
    });
    this.CONSUME(Semicolon);
  });

  // argument_list ::= identifier ("," identifier)*
  // Los argumentos son solo identificadores: todo dato o criterio se declara en
  // su propio `source` y se referencia por nombre.
  private argumentList = this.RULE("argumentList", () => {
    this.CONSUME(Identifier);
    this.MANY(() => {
      this.CONSUME(Comma);
      this.CONSUME2(Identifier);
    });
  });

  // object_literal | group
  private literal = this.RULE("literal", () => {
    this.OR([
      { ALT: () => this.SUBRULE(this.objectLiteral) },
      { ALT: () => this.SUBRULE(this.group) },
    ]);
  });

  // group ::= "[" (data_literal ("," data_literal)*)? "]"
  // La homogeneidad (solo datos) la comprueba el visitor: el parser acepta
  // cualquier object_literal y el criterio dentro de un grupo es un error de
  // sintaxis reportado con posición (§4.1).
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

  // object_literal ::= "{" (kv_pair ("," kv_pair)*)? "}"
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
      { ALT: () => this.SUBRULE(this.kvArrayLiteral) },
      { ALT: () => this.CONSUME2(StringLiteral) },
      { ALT: () => this.CONSUME(NumberLiteral) },
    ]);
  });

  // array_literal ::= "[" (string_literal ("," string_literal)*)? "]"
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
