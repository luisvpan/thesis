// Tipos de cartas para el IDE de programación tangible

export type CardCategory = 'operator' | 'number' | 'animal' | 'food' | 'person' | 'car' | 'shape' | 'montessori' | 'cap' | 'stick';

// Operadores matemáticos básicos
export type MathOperatorType = 'adicion' | 'sustraccion' | 'multiplicacion' | 'division';

// Modos de visualización de división (partitivo vs cuotativo)
export type DivisionMode = 'partitivo' | 'cuotativo';

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

export function isMathOperatorType(op: OperatorType): op is MathOperatorType {
  return (
    op === 'adicion' ||
    op === 'sustraccion' ||
    op === 'multiplicacion' ||
    op === 'division'
  );
}

export function isFilterOperatorType(op: OperatorType): op is FilterOperatorType {
  return op.startsWith('filtrar-');
}

export function isOrderOperatorType(op: OperatorType): op is OrderOperatorType {
  return op === 'orden-mayor-menor' || op === 'orden-menor-mayor';
}

export type AnimalType = 'gato' | 'perro' | 'tortuga' | 'elefante' | 'jirafa';

export type FoodType = 'manzana' | 'hamburguesa' | 'uvas' | 'pasta' | 'peras' | 'naranja';

export type PersonAge = 'bebe' | 'niño' | 'joven' | 'adulto';

export type PersonGender = 'mujer' | 'hombre';

export type CarColor = 'rojo' | 'negro' | 'amarillo' | 'azul-oscuro' | 'gris';

export type ShapeType = 'triangulo' | 'cuadrado' | 'rectangulo' | 'rombo' | 'circulo' | 'estrella' | 'trapecio';

export type ShapeSize = 'pequeño' | 'mediano' | 'grande';

export type ShapeColor =
  | 'rojo'
  | 'azul'
  | 'amarillo'
  | 'verde'
  | 'morado'
  | 'naranja';

// Colores de cubos Montessori (cube_blue, cube_red, cube_yellow)
export type MontessoriColor = 'azul' | 'rojo' | 'amarillo';

// Colores de tapas (cap_blue, cap_white)
export type CapColor = 'azul' | 'blanco';

// Colores de palitos (stick_cyan, stick_orange, stick_red)
export type StickColor = 'cian' | 'naranja' | 'rojo';

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

export interface MontessoriCard extends BaseCard {
  category: 'montessori';
  color: MontessoriColor;
}

export interface CapCard extends BaseCard {
  category: 'cap';
  color: CapColor;
}

export interface StickCard extends BaseCard {
  category: 'stick';
  color: StickColor;
}

export type Card = OperatorCard | NumberCard | AnimalCard | FoodCard | PersonCard | CarCard | ShapeCard | MontessoriCard | CapCard | StickCard;