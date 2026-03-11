export interface GeneratorFactory<T = unknown> {
  (): Generator<T>;
}

export const BUILTIN_GENERATORS: Record<string, GeneratorFactory> = {
  counter: function* () {
    let i = 0;
    while (true) {
      yield i++;
    }
  }
};

export function getGenerator(name: string): GeneratorFactory | undefined {
  return BUILTIN_GENERATORS[name];
}

export function hasGenerator(name: string): boolean {
  return name in BUILTIN_GENERATORS;
}
