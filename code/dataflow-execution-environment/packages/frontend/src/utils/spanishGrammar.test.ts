import { describe, expect, test } from 'bun:test';
import { conjugateColor, conjugateSize, describeCountedNoun, nounForm } from './spanishGrammar';

describe('nounForm', () => {
  test('usa la forma correcta con tilde en singular y plural', () => {
    expect(nounForm('triangulo', 1)).toBe('triángulo');
    expect(nounForm('triangulo', 3)).toBe('triángulos');
  });

  test('reconoce alias en plural que entrega el intérprete (peras, uvas)', () => {
    expect(nounForm('peras', 1)).toBe('pera');
    expect(nounForm('peras', 5)).toBe('peras');
    expect(nounForm('uvas', 2)).toBe('uvas');
  });
});

describe('conjugateColor / conjugateSize', () => {
  test('concuerda en género y número con el sustantivo', () => {
    expect(conjugateColor('rojo', 'f', 3)).toBe('rojas');
    expect(conjugateColor('rojo', 'm', 1)).toBe('rojo');
    expect(conjugateColor('azul', 'f', 3)).toBe('azules');
    expect(conjugateSize('grande', 'm', 3)).toBe('grandes');
    expect(conjugateSize('pequeño', 'f', 1)).toBe('pequeña');
  });
});

describe('describeCountedNoun', () => {
  test('corrige "3 manzanas rojo" a "3 manzanas rojas"', () => {
    expect(describeCountedNoun('manzana', 3, { color: 'rojo' })).toBe('manzanas rojas');
  });

  test('incluye tamaño y color juntos para figuras (bug de figuras sin tamaño)', () => {
    expect(describeCountedNoun('triangulo', 1, { size: 'grande', color: 'rojo' })).toBe(
      'triángulo grande rojo'
    );
    expect(describeCountedNoun('triangulo', 2, { size: 'grande', color: 'rojo' })).toBe(
      'triángulos grandes rojos'
    );
  });

  test('cubos/tapas/paletas concuerdan con el color', () => {
    expect(describeCountedNoun('montessori', 3, { color: 'azul' })).toBe('cubos azules');
    expect(describeCountedNoun('cap', 1, { color: 'blanco' })).toBe('tapa blanca');
  });
});
