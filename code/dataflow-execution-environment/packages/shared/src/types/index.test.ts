import { describe, it, expect } from 'bun:test';

import type {
  Natural,
  Integer,
  Decimal,
  Text,
  Boolean as Bool,
  Shape,
  Car,
  Food,
  Animal,
  Person,
  DataflowProgram,
  DataflowNode,
  DataSourceNode,
  TransformationNode,
  OutputNode,
  DataflowEdge,
  ValidationError,
  ValidationResult,
  ShapeType,
  Size,
  Color,
  Taste,
  AgeGroup,
  Gender,
  AnimalType,
} from '@dataflow/shared/types';

describe('Shared Types - Primitives', () => {
  it('should create valid Natural type', () => {
    const natural: Natural = { kind: 'natural', value: 5 };
    expect(natural.kind).toBe('natural');
    expect(natural.value).toBe(5);
  });

  it('should create valid Integer type', () => {
    const integer: Integer = { kind: 'integer', value: -3 };
    expect(integer.kind).toBe('integer');
    expect(typeof integer.value).toBe('number');
    expect(integer.value).toBe(-3);
  });

  it('should create valid Decimal type', () => {
    const decimal: Decimal = { kind: 'decimal', value: 3.14 };
    expect(decimal.kind).toBe('decimal');
    expect(typeof decimal.value).toBe('number');
    expect(decimal.value).toBe(3.14);
  });

  it('should create valid Text type', () => {
    const text: Text = { kind: 'text', value: 'hello' };
    expect(text.kind).toBe('text');
    expect(typeof text.value).toBe('string');
    expect(text.value).toBe('hello')
  });

  it('should create valid Boolean type', () => {
    const bool: Bool = { kind: 'boolean', value: true };
    expect(bool.kind).toBe('boolean');
    expect(typeof bool.value).toBe('boolean');
    expect(bool.value).toBe(true)
  });
});

describe('Shared Types - Curriculum', () => {
  it('should create valid Shape type', () => {
    const shape: Shape = {
      kind: 'shape',
      type: 'circle',
      size: 'large',
      color: 'red'
    };
    expect(shape.kind).toBe('shape');
    expect(['circle', 'triangle', 'square', 'rectangle'] satisfies ShapeType[]).toContain(shape.type);
    expect(['small', 'medium', 'large'] satisfies Size[]).toContain(shape.size);
    expect(['red', 'blue', 'yellow', 'green', 'orange', 'purple'] satisfies Color[]).toContain(shape.color);
  });

  it('should create valid Car type', () => {
    const car: Car = { kind: 'car', color: 'blue' };
    expect(car.kind).toBe('car');
    expect(['red', 'blue', 'yellow', 'green', 'orange', 'purple'] satisfies Color[]).toContain(car.color);
  });

  it('should create valid Food type', () => {
    const food: Food = {
      kind: 'food',
      taste: 'sweet',
      color: 'red'
    };
    expect(food.kind).toBe('food');
    expect(['sweet', 'salty', 'sour', 'bitter'] satisfies Taste[]).toContain(food.taste);
    expect(['red', 'blue', 'yellow', 'green', 'orange', 'purple'] satisfies Color[]).toContain(food.color);
  });

  it('should create valid Animal type', () => {
    const animal: Animal = {
      kind: 'animal',
      type: 'dog',
      color: 'purple'
    };
    expect(animal.kind).toBe('animal');
    expect(['dog', 'cat', 'bird', 'fish', 'rabbit', 'turtle'] as AnimalType[]).toContain(animal.type);
    expect(['red', 'blue', 'yellow', 'green', 'orange', 'purple'] satisfies Color[]).toContain(animal.color);
  });

  it('should create valid Person type', () => {
    const person: Person = {
      kind: 'person',
      ageGroup: 'child',
      gender: 'male'
    };
    expect(person.kind).toBe('person');
    expect(['child', 'teenager', 'adult', 'senior'] as AgeGroup[]).toContain(person.ageGroup);
    expect(['male', 'female'] as Gender[]).toContain(person.gender);
  });

  it('should have only 6 colors per spec', () => {
    const validColors: Color[] = ['red', 'blue', 'yellow', 'green', 'orange', 'purple'];
    expect(validColors).toHaveLength(6);

    const validColorValues: string[] = validColors;
    expect(validColorValues).not.toContain('white');
    expect(validColorValues).not.toContain('black');
  });
});

describe('Shared Types - Program Structure', () => {
  it('should create valid DataSourceNode', () => {
    const node: DataSourceNode = {
      id: 'n1',
      type: 'DataSource',
      dataType: 'natural',
      value: 5
    };
    expect(node.id).toBe('n1');
    expect(node.type).toBe('DataSource');
    expect(node.dataType).toBe('natural');
    expect(node.value).toBe(5);
  });

  it('should create valid TransformationNode', () => {
    const node: TransformationNode = {
      id: 'add',
      type: 'Transformation',
      dataType: 'natural',
      operation: 'ADD',
      inputs: ['n1', 'n2']
    };
    expect(node.id).toBe('add');
    expect(node.type).toBe('Transformation');
    expect(node.dataType).toBe('natural');
    expect(node.operation).toBe('ADD');
    expect(node.inputs).toHaveLength(2);
    expect(node.inputs).toContain('n1');
    expect(node.inputs).toContain('n2');
  });

  it('should create valid OutputNode', () => {
    const node: OutputNode = {
      id: 'output',
      type: 'Output',
      dataType: 'natural',
      input: 'add'
    };
    expect(node.id).toBe('output')
    expect(node.type).toBe('Output');
    expect(node.dataType).toBe('natural');
    expect(node.input).toBe('add');
  });

  it('should create valid DataflowEdge', () => {
    const edge: DataflowEdge = {
      id: 'e1',
      from: 'n1',
      to: 'add',
      toPort: 0
    };
    expect(edge.id).toBe('e1');
    expect(edge.from).toBe('n1');
    expect(edge.to).toBe('add');
    expect(edge.toPort).toBe(0);
  });

  it('should create valid DataflowProgram', () => {
    const program: DataflowProgram = {
      metadata: {
        programId: 'prog_001',
        activityId: 'activity_1',
        level: 1
      },
      graph: {
        nodes: [],
        edges: []
      }
    };
    expect(program.metadata.programId).toBe('prog_001');
    expect(program.metadata.activityId).toBe('activity_1');
    expect(program.metadata.level).toBe(1);
    expect(program.graph.nodes).toEqual([]);
    expect(program.graph.edges).toEqual([]);
  });
});

describe('Shared Types - Validation', () => {
  it('should create valid ValidationError', () => {
    const error: ValidationError = {
      code: 'CYCLE_DETECTED',
      message: 'Graph contains a cycle',
      childMessage: '⚠️ ¡Ups! Hay un ciclo en el programa.',
      nodeId: 'node_1',
      suggestion: 'Busca dónde un bloque apunta a sí mismo y desconecta una línea.',
      example: '[A] → [B] → [C] ❌\n[A] → [B] ✅'
    };
    expect(error.code).toBe('CYCLE_DETECTED');
    expect(error.nodeId).toBe('node_1');
    expect(error.childMessage).toBeDefined();
    expect(error.suggestion).toBeDefined();
    expect(error.example).toBeDefined();
    expect(error.message).toBeDefined();
  });

  it('should create valid ValidationResult with success', () => {
    const result: ValidationResult = {
      success: true,
      errors: [],
      warnings: []
    };
    expect(result.success).toBe(true);
    expect(result.errors).toHaveLength(0);
    expect(result.warnings).toHaveLength(0);
  });

  it('should create valid ValidationResult with errors', () => {
    const result: ValidationResult = {
      success: false,
      errors: [
        {
          code: 'WRONG_ARITY',
          message: 'ADD requires 2 inputs'
        }
      ],
      warnings: []
    };
    expect(result.success).toBe(false);
    expect(result.errors).toHaveLength(1);
    expect(result.warnings).toHaveLength(0);
  });
});
