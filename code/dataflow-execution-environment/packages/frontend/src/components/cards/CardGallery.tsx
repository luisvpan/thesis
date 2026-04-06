import { OperatorCard } from './OperatorCard';
import { NumberCard } from './NumberCard';
import { AnimalCard } from './AnimalCard';
import { FoodCard } from './FoodCard';
import { CarCard } from './CarCard';
import { PersonCard } from './PersonCard';
import { ShapeCard } from './ShapeCard';
import type { OperatorType, AnimalType, FoodType, CarColor, PersonGender, PersonAge, ShapeType, ShapeSize, ShapeColor } from '../../types/card-types';

interface CardGalleryProps {
  cardSize?: 'small' | 'medium' | 'large';
}

export function CardGallery({ cardSize = 'medium' }: CardGalleryProps) {
  // Operadores matemáticos básicos
  const mathOperators: OperatorType[] = ['adicion', 'sustraccion', 'multiplicacion', 'division'];

  // Operadores de orden y comparación
  const orderOperators: OperatorType[] = [
    'orden-mayor-menor', 'orden-menor-mayor',
    'comparar-figuras', 'comparar-carros', 'comparar-comidas', 'comparar-animales', 'comparar-personas'
  ];

  // Operadores de filtrado
  const filterOperators: OperatorType[] = [
    'filtrar-general',
    'filtrar-figuras', 'filtrar-carros', 'filtrar-comidas', 'filtrar-animales', 'filtrar-personas'
  ];

  // Operadores de conjuntos
  const setOperators: OperatorType[] = ['union', 'interseccion', 'diferencia', 'complemento'];

  // Todos los operadores
  const operators: OperatorType[] = [...mathOperators, ...orderOperators, ...filterOperators, ...setOperators];

  const numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

  const animals: AnimalType[] = ['gato', 'perro', 'tortuga', 'elefante', 'jirafa'];

  const foods: FoodType[] = ['manzana', 'hamburguesa', 'uvas', 'pasta', 'peras'];

  const carColors: CarColor[] = ['rojo', 'negro', 'amarillo', 'azul-oscuro', 'gris'];

  const personGenders: PersonGender[] = ['mujer', 'hombre'];
  const personAges: PersonAge[] = ['bebe', 'niño', 'joven', 'adulto'];

  const shapes: ShapeType[] = ['triangulo', 'cuadrado', 'rectangulo', 'rombo', 'circulo', 'estrella', 'trapecio'];
  const shapeSizes: ShapeSize[] = ['pequeña', 'mediana', 'grande'];
  const shapeColors: ShapeColor[] = ['rojo', 'azul', 'amarillo', 'verde'];

  // Formas con variantes de tamaño y color
  const shapesWithVariants: ShapeType[] = ['triangulo', 'circulo', 'cuadrado'];
  
  // Formas sin variantes (solo una carta cada una)
  const shapesWithoutVariants: ShapeType[] = ['rectangulo', 'rombo', 'estrella', 'trapecio'];

  return (
    <div className="w-full h-full overflow-y-auto bg-white p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-6xl font-black text-center mb-4 text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-pink-600">
          🎴 Galería de Cartas
        </h1>
        <p className="text-center text-gray-600 mb-12 text-xl">
          Todas las cartas disponibles para el entorno de programación tangible
        </p>

        {/* Operadores */}
        <section className="mb-16">
          <div className="bg-gradient-to-r from-purple-400 to-pink-400 rounded-2xl p-1 mb-4 shadow-lg">
            <div className="bg-white rounded-xl px-6 py-3">
              <h2 className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-purple-600 to-pink-600">
                ⚙️ Operadores ({operators.length} cartas)
              </h2>
            </div>
          </div>
          <div className="bg-gray-50 rounded-2xl p-6 border-2 border-purple-200">
            {/* Operadores Matemáticos */}
            <div className="mb-8">
              <h3 className="text-2xl font-bold text-gray-700 mb-4 flex items-center gap-2">
                <span className="text-3xl">🔢</span> Matemáticos ({mathOperators.length})
              </h3>
              <div className="flex flex-wrap gap-4 justify-center">
                {mathOperators.map((op) => (
                  <div key={op} className="flex flex-col items-center">
                    <OperatorCard operator={op} size={cardSize} />
                  </div>
                ))}
              </div>
            </div>

            {/* Operadores de Orden y Comparación */}
            <div className="mb-8">
              <h3 className="text-2xl font-bold text-gray-700 mb-4 flex items-center gap-2">
                <span className="text-3xl">⚖️</span> Orden y Comparación ({orderOperators.length})
              </h3>
              <div className="flex flex-wrap gap-4 justify-center">
                {orderOperators.map((op) => (
                  <div key={op} className="flex flex-col items-center">
                    <OperatorCard operator={op} size={cardSize} />
                  </div>
                ))}
              </div>
            </div>

            {/* Operadores de Filtrado */}
            <div className="mb-8">
              <h3 className="text-2xl font-bold text-gray-700 mb-4 flex items-center gap-2">
                <span className="text-3xl">🔍</span> Filtrado ({filterOperators.length})
              </h3>
              <div className="flex flex-wrap gap-4 justify-center">
                {filterOperators.map((op) => (
                  <div key={op} className="flex flex-col items-center">
                    <OperatorCard operator={op} size={cardSize} />
                  </div>
                ))}
              </div>
            </div>

            {/* Operadores de Conjuntos */}
            <div>
              <h3 className="text-2xl font-bold text-gray-700 mb-4 flex items-center gap-2">
                <span className="text-3xl">🔗</span> Conjuntos ({setOperators.length})
              </h3>
              <div className="flex flex-wrap gap-4 justify-center">
                {setOperators.map((op) => (
                  <div key={op} className="flex flex-col items-center">
                    <OperatorCard operator={op} size={cardSize} />
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* Números */}
        <section className="mb-16">
          <div className="bg-gradient-to-r from-blue-400 to-cyan-400 rounded-2xl p-1 mb-4 shadow-lg">
            <div className="bg-white rounded-xl px-6 py-3">
              <h2 className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-cyan-600">
                🔢 Números
              </h2>
            </div>
          </div>
          <div className="bg-gray-50 rounded-2xl p-6 border-2 border-blue-200">
            <div className="flex flex-wrap gap-4 justify-center">
              {numbers.map((num) => (
                <div key={num} className="flex flex-col items-center">
                  <NumberCard value={num} size={cardSize} />
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Animales */}
        <section className="mb-16">
          <div className="bg-gradient-to-r from-green-400 to-emerald-400 rounded-2xl p-1 mb-4 shadow-lg">
            <div className="bg-white rounded-xl px-6 py-3">
              <h2 className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-green-600 to-emerald-600">
                🐾 Animales
              </h2>
            </div>
          </div>
          <div className="bg-gray-50 rounded-2xl p-6 border-2 border-green-200">
            <div className="flex flex-wrap gap-4 justify-center">
              {animals.map((animal) => (
                <div key={animal} className="flex flex-col items-center">
                  <AnimalCard animal={animal} size={cardSize} />
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Comidas */}
        <section className="mb-16">
          <div className="bg-gradient-to-r from-red-400 to-orange-400 rounded-2xl p-1 mb-4 shadow-lg">
            <div className="bg-white rounded-xl px-6 py-3">
              <h2 className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-red-600 to-orange-600">
                🍎 Comidas
              </h2>
            </div>
          </div>
          <div className="bg-gray-50 rounded-2xl p-6 border-2 border-red-200">
            <div className="flex flex-wrap gap-4 justify-center">
              {foods.map((food) => (
                <div key={food} className="flex flex-col items-center">
                  <FoodCard food={food} size={cardSize} />
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Carros */}
        <section className="mb-16">
          <div className="bg-gradient-to-r from-red-400 to-orange-400 rounded-2xl p-1 mb-4 shadow-lg">
            <div className="bg-white rounded-xl px-6 py-3">
              <h2 className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-red-600 to-orange-600">
                🚗 Carros
              </h2>
            </div>
          </div>
          <div className="bg-gray-50 rounded-2xl p-6 border-2 border-red-200">
            <div className="flex flex-wrap gap-4 justify-center">
              {carColors.map((color) => (
                <div key={color} className="flex flex-col items-center">
                  <CarCard color={color} size={cardSize} />
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Personas */}
        <section className="mb-16">
          <div className="bg-gradient-to-r from-pink-400 to-rose-400 rounded-2xl p-1 mb-4 shadow-lg">
            <div className="bg-white rounded-xl px-6 py-3">
              <h2 className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-pink-600 to-rose-600">
                👥 Personas
              </h2>
            </div>
          </div>
          <div className="bg-gray-50 rounded-2xl p-6 border-2 border-pink-200">
            <p className="text-center text-gray-700 mb-6 font-bold">
              {personGenders.length} géneros × {personAges.length} edades = {personGenders.length * personAges.length} cartas
            </p>
            <div className="grid grid-cols-5 gap-4 mb-2">
              {/* Encabezados de edades */}
              <div></div>
              {personAges.map((age) => (
                <div key={age} className="text-center font-bold text-base text-gray-600 capitalize">
                  {age}
                </div>
              ))}
            </div>
            {personGenders.map((gender) => (
              <div key={gender} className="grid grid-cols-5 gap-4 mb-3">
                {/* Etiqueta de género */}
                <div className="flex items-center font-bold text-base text-gray-600 capitalize">
                  {gender}
                </div>
                {/* Cartas */}
                {personAges.map((age) => (
                  <div key={`${gender}-${age}`} className="flex justify-center">
                    <PersonCard 
                      gender={gender} 
                      age={age} 
                      size={cardSize}
                    />
                  </div>
                ))}
              </div>
            ))}
          </div>
        </section>

        {/* Figuras */}
        <section className="mb-16">
          <div className="bg-gradient-to-r from-yellow-400 to-lime-400 rounded-2xl p-1 mb-4 shadow-lg">
            <div className="bg-white rounded-xl px-6 py-3">
              <h2 className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-yellow-600 to-lime-600">
                🎨 Figuras
              </h2>
            </div>
          </div>
          <div className="bg-gray-50 rounded-2xl p-6 border-2 border-yellow-200">
            <p className="text-center text-gray-700 mb-6 font-bold">
              {shapesWithVariants.length} formas con variantes × {shapeSizes.length} tamaños × {shapeColors.length} colores = {shapesWithVariants.length * shapeSizes.length * shapeColors.length} cartas
            </p>
            <p className="text-center text-gray-700 mb-6 font-bold">
              {shapesWithoutVariants.length} formas sin variantes = {shapesWithoutVariants.length} cartas
            </p>
            <div className="grid grid-cols-5 gap-4 mb-2">
              {/* Encabezados de tamaños */}
              <div></div>
              {shapeSizes.map((size) => (
                <div key={size} className="text-center font-bold text-base text-gray-600 capitalize">
                  {size}
                </div>
              ))}
            </div>
            {shapesWithVariants.map((shape) => (
              <div key={shape} className="grid grid-cols-5 gap-4 mb-3">
                {/* Etiqueta de forma */}
                <div className="flex items-center font-bold text-base text-gray-600 capitalize">
                  {shape}
                </div>
                {/* Cartas */}
                {shapeSizes.map((size) => (
                  <div key={`${shape}-${size}`} className="flex justify-center">
                    <ShapeCard 
                      shape={shape} 
                      size={size} 
                      color={shapeColors[0]} // Puedes cambiar esto para mostrar diferentes colores
                    />
                  </div>
                ))}
              </div>
            ))}
            {/* Formas sin variantes */}
            <div className="grid grid-cols-5 gap-4 mb-3">
              {/* Etiqueta de formas sin variantes */}
              <div className="flex items-center font-bold text-base text-gray-600 capitalize">
                Sin variantes
              </div>
              {/* Cartas */}
              {shapesWithoutVariants.map((shape) => (
                <div key={shape} className="flex justify-center">
                  <ShapeCard 
                    shape={shape} 
                    size={shapeSizes[1]} // Tamaño mediano por defecto
                    color={shapeColors[0]} // Puedes cambiar esto para mostrar diferentes colores
                  />
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Resumen total */}
        <section className="mb-8">
          <div className="bg-gradient-to-r from-indigo-400 to-purple-400 rounded-2xl p-1 shadow-xl">
            <div className="bg-white rounded-xl px-8 py-6">
              <h3 className="text-3xl font-black text-center text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-purple-600 mb-4">
                📊 Resumen Total de Cartas
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-center">
                <div className="bg-purple-50 rounded-xl p-3 border-2 border-purple-200">
                  <div className="text-4xl font-black text-purple-600">{operators.length}</div>
                  <div className="text-base font-bold text-gray-600">Operadores</div>
                </div>
                <div className="bg-blue-50 rounded-xl p-3 border-2 border-blue-200">
                  <div className="text-4xl font-black text-blue-600">{numbers.length}</div>
                  <div className="text-base font-bold text-gray-600">Números</div>
                </div>
                <div className="bg-green-50 rounded-xl p-3 border-2 border-green-200">
                  <div className="text-4xl font-black text-green-600">{animals.length}</div>
                  <div className="text-base font-bold text-gray-600">Animales</div>
                </div>
                <div className="bg-red-50 rounded-xl p-3 border-2 border-red-200">
                  <div className="text-4xl font-black text-red-600">{foods.length}</div>
                  <div className="text-base font-bold text-gray-600">Comidas</div>
                </div>
                <div className="bg-red-50 rounded-xl p-3 border-2 border-red-200">
                  <div className="text-4xl font-black text-red-600">{carColors.length}</div>
                  <div className="text-base font-bold text-gray-600">Carros</div>
                </div>
                <div className="bg-pink-50 rounded-xl p-3 border-2 border-pink-200">
                  <div className="text-4xl font-black text-pink-600">{personGenders.length * personAges.length}</div>
                  <div className="text-base font-bold text-gray-600">Personas</div>
                </div>
                <div className="bg-yellow-50 rounded-xl p-3 border-2 border-yellow-200">
                  <div className="text-4xl font-black text-yellow-600">{shapesWithVariants.length * shapeSizes.length * shapeColors.length + shapesWithoutVariants.length}</div>
                  <div className="text-base font-bold text-gray-600">Figuras</div>
                </div>
              </div>
              <div className="mt-6 text-center bg-gradient-to-r from-indigo-500 to-purple-500 rounded-xl p-4 text-white">
                <div className="text-5xl font-black">
                  {operators.length + numbers.length + animals.length + foods.length + carColors.length + (personGenders.length * personAges.length) + (shapesWithVariants.length * shapeSizes.length * shapeColors.length + shapesWithoutVariants.length)}
                </div>
                <div className="text-xl font-bold">CARTAS TOTALES</div>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}