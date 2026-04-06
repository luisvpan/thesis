// Tipos de cartas para el IDE de programación tangible

export type CardCategory = 'operator' | 'number' | 'animal' | 'food' | 'person' | 'car' | 'shape';

// Operadores matemáticos básicos
export type MathOperatorType = 'adicion' | 'sustraccion' | 'multiplicacion' | 'division';

// Operadores de orden y comparación
export type OrderOperatorType = 
  | 'orden-mayor-menor'
  | 'orden-menor-mayor'
  | 'comparar-figuras'
  | 'comparar-carros'
  | 'comparar-comidas'
  | 'comparar-animales'
  | 'comparar-personas';

// Operadores de filtrado
export type FilterOperatorType =
  | 'filtrar-general'
  | 'filtrar-figuras'
  | 'filtrar-carros'
  | 'filtrar-comidas'
  | 'filtrar-animales'
  | 'filtrar-personas';

// Operadores de conjuntos
export type SetOperatorType = 'union' | 'interseccion' | 'diferencia' | 'complemento';

// Unión de todos los operadores
export type OperatorType = MathOperatorType | OrderOperatorType | FilterOperatorType | SetOperatorType;

export type AnimalType = 'gato' | 'perro' | 'tortuga' | 'elefante' | 'jirafa';

export type FoodType = 'manzana' | 'hamburguesa' | 'uvas' | 'pasta' | 'peras';

export type PersonAge = 'bebe' | 'niño' | 'joven' | 'adulto';

export type PersonGender = 'mujer' | 'hombre';

export type CarColor = 'rojo' | 'negro' | 'amarillo' | 'azul-oscuro' | 'gris';

export type ShapeType = 'triangulo' | 'cuadrado' | 'rectangulo' | 'rombo' | 'circulo' | 'estrella' | 'trapecio';

export type ShapeSize = 'pequeña' | 'mediana' | 'grande';

export type ShapeColor = 'rojo' | 'azul' | 'amarillo' | 'verde';

export interface BaseCard {
  id: string;
  category: CardCategory;
}

export interface OperatorCard extends BaseCard {
  category: 'operator';
  operator: OperatorType;
}

export interface NumberCard extends BaseCard {
  category: 'number';
  value: number; // 0-9
}

export interface AnimalCard extends BaseCard {
  category: 'animal';
  animal: AnimalType;
}

export interface FoodCard extends BaseCard {
  category: 'food';
  food: FoodType;
}

export interface PersonCard extends BaseCard {
  category: 'person';
  gender: PersonGender;
  age: PersonAge;
}

export interface CarCard extends BaseCard {
  category: 'car';
  color: CarColor;
}

export interface ShapeCard extends BaseCard {
  category: 'shape';
  shape: ShapeType;
  size: ShapeSize;
  color: ShapeColor;
}

export type Card = OperatorCard | NumberCard | AnimalCard | FoodCard | PersonCard | CarCard | ShapeCard;