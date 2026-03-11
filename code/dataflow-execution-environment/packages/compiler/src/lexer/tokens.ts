import { createToken, Lexer } from "chevrotain";

export const Source = createToken({ name: "Source", pattern: /\bsource\b/i });
export const Transform = createToken({ name: "Transform", pattern: /\btransform\b/i });
export const Output = createToken({ name: "Output", pattern: /\boutput\b/i });

export const Natural = createToken({ name: "Natural", pattern: /\bnatural\b/i });
export const Integer = createToken({ name: "Integer", pattern: /\binteger\b/i });
export const Decimal = createToken({ name: "Decimal", pattern: /\bdecimal\b/i });
export const Text = createToken({ name: "Text", pattern: /\btext\b/i });
export const Boolean = createToken({ name: "Boolean", pattern: /\bboolean\b/i });

export const Set = createToken({ name: "Set", pattern: /\bset\b/i });
export const Stream = createToken({ name: "Stream", pattern: /\bstream\b/i });

export const OperationKeyword = createToken({ name: "OperationKeyword", pattern: Lexer.NA })

export const CompareBySize = createToken({ name: "CompareBySize", pattern: /COMPARE_BY_SIZE/, categories: OperationKeyword });
export const CompareByColor = createToken({ name: "CompareByColor", pattern: /COMPARE_BY_COLOR/, categories: OperationKeyword });
export const CompareByType = createToken({ name: "CompareByType", pattern: /COMPARE_BY_TYPE/, categories: OperationKeyword });
export const CompareByTaste = createToken({ name: "CompareByTaste", pattern: /COMPARE_BY_TASTE/, categories: OperationKeyword });
export const CompareByAgeGroup = createToken({ name: "CompareByAgeGroup", pattern: /COMPARE_BY_AGE_GROUP/, categories: OperationKeyword });
export const CompareByGender = createToken({ name: "CompareByGender", pattern: /COMPARE_BY_GENDER/, categories: OperationKeyword });

export const FilterBySize = createToken({ name: "FilterBySize", pattern: /FILTER_BY_SIZE/, categories: OperationKeyword });
export const FilterByColor = createToken({ name: "FilterByColor", pattern: /FILTER_BY_COLOR/, categories: OperationKeyword });
export const FilterByType = createToken({ name: "FilterByType", pattern: /FILTER_BY_TYPE/, categories: OperationKeyword });
export const FilterByTaste = createToken({ name: "FilterByTaste", pattern: /FILTER_BY_TASTE/, categories: OperationKeyword });
export const FilterByAgeGroup = createToken({ name: "FilterByAgeGroup", pattern: /FILTER_BY_AGE_GROUP/, categories: OperationKeyword });
export const FilterByGender = createToken({ name: "FilterByGender", pattern: /FILTER_BY_GENDER/, categories: OperationKeyword });

export const AlphabeticalSort = createToken({ name: "AlphabeticalSort", pattern: /ALPHABETICAL_SORT/, categories: OperationKeyword });

export const And = createToken({ name: "And", pattern: /AND/, categories: OperationKeyword });
export const Or = createToken({ name: "Or", pattern: /OR/, categories: OperationKeyword });
export const Not = createToken({ name: "Not", pattern: /NOT/, categories: OperationKeyword });

export const Add = createToken({ name: "Add", pattern: /ADD/, categories: OperationKeyword });
export const Subtract = createToken({ name: "Subtract", pattern: /SUBTRACT/, categories: OperationKeyword });
export const Multiply = createToken({ name: "Multiply", pattern: /MULTIPLY/, categories: OperationKeyword });
export const Divide = createToken({ name: "Divide", pattern: /DIVIDE/, categories: OperationKeyword });
export const Compare = createToken({ name: "Compare", pattern: /COMPARE/, categories: OperationKeyword });
export const Filter = createToken({ name: "Filter", pattern: /FILTER/, categories: OperationKeyword });
export const Union = createToken({ name: "Union", pattern: /UNION/, categories: OperationKeyword });
export const Intersection = createToken({ name: "Intersection", pattern: /INTERSECTION/, categories: OperationKeyword });
export const Difference = createToken({ name: "Difference", pattern: /DIFFERENCE/, categories: OperationKeyword });
export const Complement = createToken({ name: "Complement", pattern: /COMPLEMENT/, categories: OperationKeyword });
export const Next = createToken({ name: "Next", pattern: /NEXT/, categories: OperationKeyword });
export const First = createToken({ name: "First", pattern: /FIRST/, categories: OperationKeyword });
export const Fby = createToken({ name: "Fby", pattern: /FBY/, categories: OperationKeyword });
export const Accumulate = createToken({ name: "Accumulate", pattern: /ACCUMULATE/, categories: OperationKeyword });

export const Sensor = createToken({ name: "Sensor", pattern: /\bsensor\b/i });
export const Generator = createToken({ name: "Generator", pattern: /\bgenerator\b/i });
export const External = createToken({ name: "External", pattern: /\bexternal\b/i });

export const Sort = createToken({ name: "Sort", pattern: /SORT/, categories: OperationKeyword });

export const True = createToken({ name: "True", pattern: /\btrue\b/i });
export const False = createToken({ name: "False", pattern: /\bfalse\b/i });

export const Circle = createToken({ name: "Circle", pattern: /\bcircle\b/i });
export const Triangle = createToken({ name: "Triangle", pattern: /\btriangle\b/i });
export const Square = createToken({ name: "Square", pattern: /\bsquare\b/i });
export const Rectangle = createToken({ name: "Rectangle", pattern: /\brectangle\b/i });

export const Small = createToken({ name: "Small", pattern: /\bsmall\b/i });
export const Medium = createToken({ name: "Medium", pattern: /\bmedium\b/i });
export const Large = createToken({ name: "Large", pattern: /\blarge\b/i });

export const Red = createToken({ name: "Red", pattern: /\bred\b/i });
export const Blue = createToken({ name: "Blue", pattern: /\bblue\b/i });
export const Yellow = createToken({ name: "Yellow", pattern: /\byellow\b/i });
export const Green = createToken({ name: "Green", pattern: /\bgreen\b/i });
export const Orange = createToken({ name: "Orange", pattern: /\borange\b/i });
export const Purple = createToken({ name: "Purple", pattern: /\bpurple\b/i });
export const Black = createToken({ name: "Black", pattern: /\bblack\b/i });
export const White = createToken({ name: "White", pattern: /\bwhite\b/i });

export const Sweet = createToken({ name: "Sweet", pattern: /\bsweet\b/i });
export const Salty = createToken({ name: "Salty", pattern: /\bsalty\b/i });
export const Sour = createToken({ name: "Sour", pattern: /\bsour\b/i });
export const Bitter = createToken({ name: "Bitter", pattern: /\bbitter\b/i });

export const Dog = createToken({ name: "Dog", pattern: /\bdog\b/i });
export const Cat = createToken({ name: "Cat", pattern: /\bcat\b/i });
export const Bird = createToken({ name: "Bird", pattern: /\bbird\b/i });
export const Fish = createToken({ name: "Fish", pattern: /\bfish\b/i });
export const Rabbit = createToken({ name: "Rabbit", pattern: /\brabbit\b/i });
export const Turtle = createToken({ name: "Turtle", pattern: /\bturtle\b/i });

export const Child = createToken({ name: "Child", pattern: /\bchild\b/i });
export const Teenager = createToken({ name: "Teenager", pattern: /\bteenager\b/i });
export const Adult = createToken({ name: "Adult", pattern: /\badult\b/i });
export const Senior = createToken({ name: "Senior", pattern: /\bsenior\b/i });

export const Male = createToken({ name: "Male", pattern: /\bmale\b/i });
export const Female = createToken({ name: "Female", pattern: /\bfemale\b/i });

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
  AlphabeticalSort,
  And,
  Or,
  Not,
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
  Black,
  White,
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
