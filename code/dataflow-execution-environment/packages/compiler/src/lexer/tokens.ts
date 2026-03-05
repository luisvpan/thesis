import { createToken, Lexer } from "chevrotain";

export const Source = createToken({ name: "Source", pattern: /source/i });
export const Transform = createToken({ name: "Transform", pattern: /transform/i });
export const Output = createToken({ name: "Output", pattern: /output/i });

export const Natural = createToken({ name: "Natural", pattern: /natural/i });
export const Integer = createToken({ name: "Integer", pattern: /integer/i });
export const Decimal = createToken({ name: "Decimal", pattern: /decimal/i });
export const Text = createToken({ name: "Text", pattern: /text/i });
export const Boolean = createToken({ name: "Boolean", pattern: /boolean/i });

export const Set = createToken({ name: "Set", pattern: /set/i });
export const Stream = createToken({ name: "Stream", pattern: /stream/i });

export const Add = createToken({ name: "Add", pattern: /ADD/ });
export const Subtract = createToken({ name: "Subtract", pattern: /SUBTRACT/ });
export const Multiply = createToken({ name: "Multiply", pattern: /MULTIPLY/ });
export const Divide = createToken({ name: "Divide", pattern: /DIVIDE/ });
export const Compare = createToken({ name: "Compare", pattern: /COMPARE/ });
export const Filter = createToken({ name: "Filter", pattern: /FILTER/ });
export const Union = createToken({ name: "Union", pattern: /UNION/ });
export const Intersection = createToken({ name: "Intersection", pattern: /INTERSECTION/ });
export const Difference = createToken({ name: "Difference", pattern: /DIFFERENCE/ });
export const Complement = createToken({ name: "Complement", pattern: /COMPLEMENT/ });
export const Next = createToken({ name: "Next", pattern: /NEXT/ });
export const First = createToken({ name: "First", pattern: /FIRST/ });
export const Fby = createToken({ name: "Fby", pattern: /FBY/ });
export const Accumulate = createToken({ name: "Accumulate", pattern: /ACCUMULATE/ });
export const Sort = createToken({ name: "Sort", pattern: /SORT/ });

export const True = createToken({ name: "True", pattern: /true/i });
export const False = createToken({ name: "False", pattern: /false/i });

export const Circle = createToken({ name: "Circle", pattern: /circle/i });
export const Triangle = createToken({ name: "Triangle", pattern: /triangle/i });
export const Square = createToken({ name: "Square", pattern: /square/i });
export const Rectangle = createToken({ name: "Rectangle", pattern: /rectangle/i });

export const Small = createToken({ name: "Small", pattern: /small/i });
export const Medium = createToken({ name: "Medium", pattern: /medium/i });
export const Large = createToken({ name: "Large", pattern: /large/i });

export const Red = createToken({ name: "Red", pattern: /red/i });
export const Blue = createToken({ name: "Blue", pattern: /blue/i });
export const Yellow = createToken({ name: "Yellow", pattern: /yellow/i });
export const Green = createToken({ name: "Green", pattern: /green/i });
export const Orange = createToken({ name: "Orange", pattern: /orange/i });
export const Purple = createToken({ name: "Purple", pattern: /purple/i });

export const Sweet = createToken({ name: "Sweet", pattern: /sweet/i });
export const Salty = createToken({ name: "Salty", pattern: /salty/i });
export const Sour = createToken({ name: "Sour", pattern: /sour/i });
export const Bitter = createToken({ name: "Bitter", pattern: /bitter/i });

export const Dog = createToken({ name: "Dog", pattern: /dog/i });
export const Cat = createToken({ name: "Cat", pattern: /cat/i });
export const Bird = createToken({ name: "Bird", pattern: /bird/i });
export const Fish = createToken({ name: "Fish", pattern: /fish/i });
export const Rabbit = createToken({ name: "Rabbit", pattern: /rabbit/i });
export const Turtle = createToken({ name: "Turtle", pattern: /turtle/i });

export const Child = createToken({ name: "Child", pattern: /child/i });
export const Teenager = createToken({ name: "Teenager", pattern: /teenager/i });
export const Adult = createToken({ name: "Adult", pattern: /adult/i });
export const Senior = createToken({ name: "Senior", pattern: /senior/i });

export const Male = createToken({ name: "Male", pattern: /male/i });
export const Female = createToken({ name: "Female", pattern: /female/i });

export const Equals = createToken({ name: "Equals", pattern: /=/ });
export const Colon = createToken({ name: "Colon", pattern: /:/ });
export const Semicolon = createToken({ name: "Semicolon", pattern: /;/ });
export const LParen = createToken({ name: "LParen", pattern: /\(/ });
export const RParen = createToken({ name: "RParen", pattern: /\)/ });
export const LBrace = createToken({ name: "LBrace", pattern: /{/ });
export const RBrace = createToken({ name: "RBrace", pattern: /}/ });
export const LBracket = createToken({ name: "LBracket", pattern: /\[/ });
export const RBracket = createToken({ name: "RBracket", pattern: /\]/ });
export const Comma = createToken({ name: "Comma", pattern: /,/ });
export const AngleLeft = createToken({ name: "AngleLeft", pattern: /</ });
export const AngleRight = createToken({ name: "AngleRight", pattern: />/ });
export const Dot = createToken({ name: "Dot", pattern: /\./ });
export const Minus = createToken({ name: "Minus", pattern: /-/ });

export const Identifier = createToken({ name: "Identifier", pattern: /[a-zA-Z_]\w*/ });

export const NumberLiteral = createToken({ name: "NumberLiteral", pattern: /[0-9]+(\.[0-9]+)?/ });

export const StringLiteral = createToken({ name: "StringLiteral", pattern: /"[^"]*"/ });

export const WhiteSpace = createToken({ 
  name: "WhiteSpace", 
  pattern: /[ \t\r\n]+/,
  group: Lexer.SKIPPED
});

export const Comment = createToken({
  name: "Comment",
  pattern: /\/\/[^\n]*|\/\*[\s\S]*?\*\//,
  group: Lexer.SKIPPED
});

export const allTokens = [
  WhiteSpace,
  Comment,
  Source,
  Transform,
  Output,
  Natural,
  Integer,
  Decimal,
  Text,
  Boolean,
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
  Sort,
  True,
  False,
  Circle,
  Triangle,
  Square,
  Rectangle,
  Small,
  Medium,
  Large,
  Red,
  Blue,
  Yellow,
  Green,
  Orange,
  Purple,
  Sweet,
  Salty,
  Sour,
  Bitter,
  Dog,
  Cat,
  Bird,
  Fish,
  Rabbit,
  Turtle,
  Child,
  Teenager,
  Adult,
  Senior,
  Male,
  Female,
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
  Minus,
  Identifier,
  NumberLiteral,
  StringLiteral
];
