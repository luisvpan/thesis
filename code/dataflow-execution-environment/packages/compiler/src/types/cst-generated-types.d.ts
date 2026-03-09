import type { CstNode, ICstVisitor, IToken } from "chevrotain";

export interface ProgramCstNode extends CstNode {
  name: "program";
  children: ProgramCstChildren;
}

export type ProgramCstChildren = {
  statement?: StatementCstNode[];
};

export interface StatementCstNode extends CstNode {
  name: "statement";
  children: StatementCstChildren;
}

export type StatementCstChildren = {
  sourceStatement?: SourceStatementCstNode[];
  transformStatement?: TransformStatementCstNode[];
  outputStatement?: OutputStatementCstNode[];
};

export interface SourceStatementCstNode extends CstNode {
  name: "sourceStatement";
  children: SourceStatementCstChildren;
}

export type SourceStatementCstChildren = {
  Source: IToken[];
  Identifier: IToken[];
  Colon: IToken[];
  typeDeclaration: TypeDeclarationCstNode[];
  Equals: IToken[];
  value: ValueCstNode[];
  Semicolon: IToken[];
};

export interface TransformStatementCstNode extends CstNode {
  name: "transformStatement";
  children: TransformStatementCstChildren;
}

export type TransformStatementCstChildren = {
  Transform: IToken[];
  Identifier: IToken[];
  Colon: IToken[];
  typeDeclaration: TypeDeclarationCstNode[];
  Equals: IToken[];
  operationExpression: OperationExpressionCstNode[];
  Semicolon: IToken[];
};

export interface OutputStatementCstNode extends CstNode {
  name: "outputStatement";
  children: OutputStatementCstChildren;
}

export type OutputStatementCstChildren = {
  Output: IToken[];
  Identifier: (IToken)[];
  Colon: IToken[];
  typeDeclaration: TypeDeclarationCstNode[];
  Equals: IToken[];
  Semicolon: IToken[];
};

export interface TypeDeclarationCstNode extends CstNode {
  name: "typeDeclaration";
  children: TypeDeclarationCstChildren;
}

export type TypeDeclarationCstChildren = {
  Natural?: IToken[];
  Integer?: IToken[];
  Decimal?: IToken[];
  Text?: IToken[];
  Boolean?: IToken[];
  setType?: SetTypeCstNode[];
  streamType?: StreamTypeCstNode[];
};

export interface SetTypeCstNode extends CstNode {
  name: "setType";
  children: SetTypeCstChildren;
}

export type SetTypeCstChildren = {
  Set: IToken[];
  AngleLeft: IToken[];
  typeDeclaration: TypeDeclarationCstNode[];
  AngleRight: IToken[];
};

export interface StreamTypeCstNode extends CstNode {
  name: "streamType";
  children: StreamTypeCstChildren;
}

export type StreamTypeCstChildren = {
  Stream: IToken[];
  AngleLeft: IToken[];
  typeDeclaration: TypeDeclarationCstNode[];
  AngleRight: IToken[];
};

export interface ValueCstNode extends CstNode {
  name: "value";
  children: ValueCstChildren;
}

export type ValueCstChildren = {
  literal?: LiteralCstNode[];
  arrayLiteral?: ArrayLiteralCstNode[];
};

export interface LiteralCstNode extends CstNode {
  name: "literal";
  children: LiteralCstChildren;
}

export type LiteralCstChildren = {
  NumberLiteral?: IToken[];
  StringLiteral?: IToken[];
  objectLiteral?: ObjectLiteralCstNode[];
  True?: IToken[];
  False?: IToken[];
};

export interface ObjectLiteralCstNode extends CstNode {
  name: "objectLiteral";
  children: ObjectLiteralCstChildren;
}

export type ObjectLiteralCstChildren = {
  LBrace: IToken[];
  Identifier?: IToken[];
  Colon?: IToken[];
  literal?: LiteralCstNode[];
  Comma?: IToken[];
  RBrace: IToken[];
};

export interface ArrayLiteralCstNode extends CstNode {
  name: "arrayLiteral";
  children: ArrayLiteralCstChildren;
}

export type ArrayLiteralCstChildren = {
  LBracket: IToken[];
  value?: ValueCstNode[];
  Comma?: IToken[];
  RBracket: IToken[];
};

export interface OperationExpressionCstNode extends CstNode {
  name: "operationExpression";
  children: OperationExpressionCstChildren;
}

export type OperationExpressionCstChildren = {
  operationName: OperationNameCstNode[];
  LParen: IToken[];
  argumentList?: ArgumentListCstNode[];
  RParen: IToken[];
};

export interface OperationNameCstNode extends CstNode {
  name: "operationName";
  children: OperationNameCstChildren;
}

export type OperationNameCstChildren = {
  Add?: IToken[];
  Subtract?: IToken[];
  Multiply?: IToken[];
  Divide?: IToken[];
  Compare?: IToken[];
  Filter?: IToken[];
  Union?: IToken[];
  Intersection?: IToken[];
  Difference?: IToken[];
  Complement?: IToken[];
  Next?: IToken[];
  First?: IToken[];
  Fby?: IToken[];
  Accumulate?: IToken[];
  Sort?: IToken[];
  AlphabeticalSort?: IToken[];
  And?: IToken[];
  Or?: IToken[];
  Not?: IToken[];
  CompareBySize?: IToken[];
  CompareByColor?: IToken[];
  CompareByType?: IToken[];
  CompareByTaste?: IToken[];
  CompareByAgeGroup?: IToken[];
  CompareByGender?: IToken[];
  FilterBySize?: IToken[];
  FilterByColor?: IToken[];
  FilterByType?: IToken[];
  FilterByTaste?: IToken[];
  FilterByAgeGroup?: IToken[];
  FilterByGender?: IToken[];
  Identifier?: IToken[];
};

export interface ArgumentListCstNode extends CstNode {
  name: "argumentList";
  children: ArgumentListCstChildren;
}

export type ArgumentListCstChildren = {
  argument?: ArgumentCstNode[];
  Comma?: IToken[];
};

export interface ArgumentCstNode extends CstNode {
  name: "argument";
  children: ArgumentCstChildren;
}

export type ArgumentCstChildren = {
  Identifier?: IToken[];
  literal?: LiteralCstNode[];
};

export interface ICstNodeVisitor<IN, OUT> extends ICstVisitor<IN, OUT> {
  program(children: ProgramCstChildren, param?: IN): OUT;
  statement(children: StatementCstChildren, param?: IN): OUT;
  sourceStatement(children: SourceStatementCstChildren, param?: IN): OUT;
  transformStatement(children: TransformStatementCstChildren, param?: IN): OUT;
  outputStatement(children: OutputStatementCstChildren, param?: IN): OUT;
  typeDeclaration(children: TypeDeclarationCstChildren, param?: IN): OUT;
  setType(children: SetTypeCstChildren, param?: IN): OUT;
  streamType(children: StreamTypeCstChildren, param?: IN): OUT;
  value(children: ValueCstChildren, param?: IN): OUT;
  literal(children: LiteralCstChildren, param?: IN): OUT;
  objectLiteral(children: ObjectLiteralCstChildren, param?: IN): OUT;
  arrayLiteral(children: ArrayLiteralCstChildren, param?: IN): OUT;
  operationExpression(children: OperationExpressionCstChildren, param?: IN): OUT;
  operationName(children: OperationNameCstChildren, param?: IN): OUT;
  argumentList(children: ArgumentListCstChildren, param?: IN): OUT;
  argument(children: ArgumentCstChildren, param?: IN): OUT;
}
