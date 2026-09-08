# Sistema de Cartas - IDE de Programación Tangible

Este directorio contiene todos los componentes de cartas para el entorno de programación tangible basado en realidad aumentada espacial.

## 📦 Estructura de Componentes

### Componente Base
- **CardBase.tsx**: Componente base que proporciona el contenedor con borde de color y fondo blanco para todas las cartas.

### Tipos de Cartas

#### 1. OperatorCard - Operadores (12 cartas)
Operadores matemáticos y de manipulación de datos:
- `suma` (+)
- `resta` (−)
- `multiplicacion` (×)
- `division` (÷)
- `tomar-primero` (↑)
- `tomar-ultimo` (↓)
- `ordenar-mayor-menor` (9→1)
- `ordenar-menor-mayor` (1→9)
- `interseccion` (∩)
- `union` (∪)
- `diferencia` (−)
- `contar` (#)

#### 2. NumberCard - Números (10 cartas)
Números del 0 al 9, cada uno con su propio color distintivo en gradiente.

#### 3. AnimalCard - Animales (5 cartas)
- Gato 🐱
- Perro 🐶
- Tortuga 🐢
- Elefante 🐘
- Jirafa 🦒

#### 4. CarCard - Carros (7 cartas)
Carros de diferentes colores:
- Rojo
- Negro
- Blanco
- Azul
- Verde
- Naranja
- Amarillo

#### 5. PersonCard - Personas (2 cartas)
- Mujer 👩
- Hombre 👨

#### 6. ShapeCard - Figuras Geométricas (105 cartas)
7 formas × 3 tamaños × 5 colores = 105 cartas

**Formas:**
- Triángulo
- Cuadrado
- Rectángulo
- Rombo
- Círculo
- Estrella
- Trapecio

**Tamaños:**
- Pequeña (P)
- Mediana (M)
- Grande (G)

**Colores:**
- Rojo
- Azul
- Morado
- Naranja
- Amarillo

## 🎨 Características de Diseño

### Tamaños de Carta
Todas las cartas soportan 3 tamaños configurables:
- `small`: 16×20 (w-16 h-20)
- `medium`: 20×28 (w-20 h-28) - por defecto
- `large`: 24×32 (w-24 h-32)

### Interactividad
- **isDraggable**: Habilita arrastrar y soltar (drag & drop)
- **onClick**: Callback cuando se hace clic en la carta
- **Animaciones**: Hover, tap, y drag con Motion (Framer Motion)

### Paleta de Colores
Cada categoría usa gradientes distintivos para fácil identificación visual:
- Operadores: Gradientes diversos según la operación
- Números: Gradiente único por número (del rojo al rosa)
- Animales: Colores relacionados con cada animal
- Carros: Gradientes basados en el color del carro
- Personas: Rosa para mujer, azul para hombre
- Figuras: 5 colores base con gradientes

## 📊 Total de Cartas

| Categoría | Cantidad |
|-----------|----------|
| Operadores | 12 |
| Números | 10 |
| Animales | 5 |
| Carros | 7 |
| Personas | 2 |
| Figuras Geométricas | 105 |
| **TOTAL** | **141 cartas** |

## 🚀 Uso

### Importar componentes individuales

```tsx
import { OperatorCard, NumberCard, AnimalCard } from './components/cards';

// Uso
<OperatorCard operator="suma" size="medium" isDraggable />
<NumberCard value={5} size="large" onClick={() => console.log('Clicked!')} />
<AnimalCard animal="gato" size="small" />
```

### Importar tipos

```tsx
import type { OperatorType, Card, ShapeColor } from './components/cards';
```

### Galería completa

```tsx
import { CardGallery } from './components/cards';

// Muestra todas las cartas organizadas por categoría
<CardGallery cardSize="medium" />
```

## 🎯 Propósito Educativo

Este sistema de cartas está diseñado específicamente para niños de 6 a 9 años, con:
- Íconos grandes y claros
- Colores vibrantes y distintivos
- Símbolos matemáticos simples
- Etiquetas de texto legibles
- Feedback visual inmediato
- Diseño apropiado para proyección en mesa (realidad aumentada espacial)

## 🔧 Tecnologías

- **React**: Componentes funcionales con TypeScript
- **Motion (Framer Motion)**: Animaciones fluidas
- **Tailwind CSS v4**: Estilos utilitarios
- **Lucide React**: Íconos SVG
- **TypeScript**: Tipos estrictos para seguridad

## 📝 Notas de Diseño

- Fondo blanco obligatorio (según especificaciones del proyecto)
- Bordes de colores para delimitación clara
- Símbolos matemáticos Unicode donde sea posible
- Emojis para representaciones visuales
- Gradientes para profundidad y atractivo visual
- Sombras para efecto de profundidad física (tangible)
