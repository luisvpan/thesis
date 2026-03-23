export interface GeneratorFactory<T = unknown> {
  (): Generator<T>;
}

export const BUILTIN_GENERATORS: Record<string, GeneratorFactory> = {
  counter: function* () {
    let i = 0;
    while (true) {
      yield i++;
    }
  },

  range: function* () {
    for (let i = 0; i < 10; i++) {
      yield i;
    }
    while (true) {
      yield 9;
    }
  },

  constant: function* () {
    while (true) {
      yield 0;
    }
  },

  repeat: function* () {
    while (true) {
      yield 1;
    }
  },

  cycle: function* () {
    const items = [0, 1];
    let index = 0;
    while (true) {
      yield items[index];
      index = (index + 1) % items.length;
    }
  }
};

export function getGenerator(name: string): GeneratorFactory | undefined {
  return BUILTIN_GENERATORS[name];
}

export function hasGenerator(name: string): boolean {
  return name in BUILTIN_GENERATORS;
}
