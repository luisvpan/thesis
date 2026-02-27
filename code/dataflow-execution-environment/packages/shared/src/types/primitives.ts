export type Natural = {
  kind: "natural";
  value: number;
};

export type Integer = {
  kind: "integer";
  value: number;
};

export type Decimal = {
  kind: "decimal";
  value: number;
};

export type Text = {
  kind: "text";
  value: string;
};

export type Boolean = {
  kind: "boolean";
  value: boolean;
};

export type Primitive = Natural | Integer | Decimal | Text | Boolean;
