// Errores de sintaxis — LANGUAGE_SPEC.md §4.1
//
// La gramática es su especificación, así que no se enumeran; lo que sí hace
// falta es reportarlos con su posición. El lexer y el parser producen los suyos;
// esta clase es para los que solo se ven al construir el AST (un grupo con un
// criterio dentro, una categoría CPA que no existe).

export class DataflowSyntaxError extends Error {
  readonly line?: number;
  readonly column?: number;

  constructor(message: string, position?: { line?: number; column?: number }) {
    super(message);
    this.name = "DataflowSyntaxError";
    this.line = position?.line;
    this.column = position?.column;
  }
}
