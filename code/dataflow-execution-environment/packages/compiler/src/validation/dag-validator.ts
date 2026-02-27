import type { DataflowNode, DataflowEdge } from "@dataflow/shared/types";
import type { ValidationError, ValidationResult } from "@dataflow/shared/types";
import type { Statement } from "../ast/ast-types.js";
import { OPERATION_REGISTRY } from "@dataflow/shared/operations";

export class DagValidator {
  validate(statements: Statement[]): ValidationResult {
    const errors: ValidationError[] = [];
    const defined = new Set<string>();

    for (const stmt of statements) {
      if (defined.has(stmt.id)) {
        errors.push({
          code: "DUPLICATE_IDENTIFIER",
          message: `Identifier '${stmt.id}' already defined`,
          nodeId: stmt.id,
          childMessage: `⚠️ ¡Ups! Ya tienes un bloque llamado "${stmt.id}". Cada bloque necesita un nombre único.`,
          suggestion: `Cambia el nombre de uno de los bloques "${stmt.id}"`,
          example: `[${stmt.id}_1] y [${stmt.id}_2] ✅`
        });
      }
      defined.add(stmt.id);
    }

    const hasCycle = this.detectCycles(statements);
    if (hasCycle) {
      errors.push({
        code: "CYCLE_DETECTED",
        message: "Graph contains a cycle",
        childMessage: "⚠️ ¡Ups! Hay un ciclo en el programa. Los bloques se conectan en círculo y no se puede calcular.",
        suggestion: "Busca dónde un bloque apunta a sí mismo y desconecta una línea.",
        example: "[A] → [B] → [C] ❌\n[A] → [B] ✅ (rompe el ciclo)"
      });
    }

    for (const stmt of statements) {
      if (stmt.type === "TransformStatement") {
        const signature = OPERATION_REGISTRY[stmt.operation];
        if (!signature) {
          errors.push({
            code: "UNKNOWN_OPERATION",
            message: `Unknown operation: ${stmt.operation}`,
            nodeId: stmt.id,
            childMessage: `⚠️ ¡Ups! No reconozco el bloque "${stmt.operation}".`,
            suggestion: `Revisa si el bloque está conectado correctamente.`,
            example: `Usa un bloque conocido como "+", "-", "×", "÷"`
          });
          continue;
        }

        if (stmt.inputs.length !== signature.arity) {
          errors.push({
            code: "WRONG_ARITY",
            message: `Operation ${stmt.operation} requires ${signature.arity} inputs, got ${stmt.inputs.length}`,
            nodeId: stmt.id,
            childMessage: `⚠️ ¡Ups! El bloque "${stmt.operation}" necesita ${signature.arity} ${signature.arity === 1 ? 'entrada' : 'entradas'}. Solo conectaste ${stmt.inputs.length}.`,
            suggestion: `Conecta ${signature.arity - stmt.inputs.length} bloque${signature.arity - stmt.inputs.length === 1 ? '' : 's'} más al bloque "${stmt.operation}".`,
            example: this.generateExample(stmt.operation, signature.arity)
          });
        }
      }

      if (stmt.type === "TransformStatement" || stmt.type === "OutputStatement") {
        const refs = stmt.type === "TransformStatement" ? stmt.inputs : [stmt.input];
        for (const ref of refs) {
          if (!defined.has(ref)) {
            errors.push({
              code: "UNDEFINED_IDENTIFIER",
              message: `Undefined identifier: ${ref}`,
              nodeId: stmt.id,
              childMessage: `⚠️ ¡Ups! No encuentro el bloque "${ref}".`,
              suggestion: "Asegúrate de que el bloque existe antes de conectarlo.",
              example: "Crea el bloque primero, luego conéctalo."
            });
          }
        }
      }
    }

    return {
      success: errors.length === 0,
      errors,
      warnings: []
    };
  }

  private detectCycles(statements: Statement[]): boolean {
    const adj = new Map<string, string[]>();
    const inDegree = new Map<string, number>();
    const allIds = new Set<string>();

    for (const stmt of statements) {
      allIds.add(stmt.id);
      inDegree.set(stmt.id, 0);
    }

    for (const stmt of statements) {
      if (stmt.type === "TransformStatement") {
        for (const input of stmt.inputs) {
          if (!adj.has(input)) {
            adj.set(input, []);
          }
          adj.get(input)!.push(stmt.id);
          inDegree.set(stmt.id, (inDegree.get(stmt.id) || 0) + 1);
        }
      } else if (stmt.type === "OutputStatement") {
        if (!adj.has(stmt.input)) {
          adj.set(stmt.input, []);
        }
        adj.get(stmt.input)!.push(stmt.id);
        inDegree.set(stmt.id, (inDegree.get(stmt.id) || 0) + 1);
      }
    }

    const queue: string[] = [];
    for (const [id, degree] of inDegree.entries()) {
      if (degree === 0) {
        queue.push(id);
      }
    }

    let visited = 0;
    while (queue.length > 0) {
      const node = queue.shift()!;
      visited++;

      if (adj.has(node)) {
        for (const neighbor of adj.get(node)!) {
          inDegree.set(neighbor, inDegree.get(neighbor)! - 1);
          if (inDegree.get(neighbor) === 0) {
            queue.push(neighbor);
          }
        }
      }
    }

    return visited !== allIds.size;
  }

  private generateExample(operation: string, arity: number): string {
    if (operation === "ADD") {
      return "[5] + [3] ✅";
    }
    if (operation === "FILTER") {
      return "[formas] 🎨 [rojo] ✅";
    }
    return `Conecta ${arity} bloques al bloque "${operation}" ✅`;
  }
}
