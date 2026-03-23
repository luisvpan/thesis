export type ShapeType = "circle" | "triangle" | "square" | "rectangle";
export type Size = "small" | "medium" | "large";
export type Color = "red" | "blue" | "yellow" | "green" | "orange" | "purple";
export type Taste = "sweet" | "salty" | "sour" | "bitter";
export type AnimalType = "dog" | "cat" | "bird" | "fish" | "rabbit" | "turtle";
export type AgeGroup = "child" | "teenager" | "adult" | "senior";
export type Gender = "male" | "female";

export type Shape = {
  kind: "shape";
  type: ShapeType;
  size: Size;
  color: Color;
};

export type Car = {
  kind: "car";
  color: Color;
};

export type Food = {
  kind: "food";
  taste: Taste;
  color: Color;
};

export type Animal = {
  kind: "animal";
  type: AnimalType;
  color: Color;
};

export type Person = {
  kind: "person";
  ageGroup: AgeGroup;
  gender: Gender;
};

export type Curriculum = Shape | Car | Food | Animal | Person;
