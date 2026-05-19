#!/usr/bin/env bun
// Dataflow REPL - Interactive interpreter for the dataflow language

import * as readline from "readline";
import { Interpreter } from "./interpreter";
import { formatValue } from "./formatter";

const VERSION = "1.0.0";

const HELP_TEXT = `
Dataflow REPL v${VERSION}

Comandos:
  .help   - Muestra esta ayuda
  .clear  - Limpia el estado y programa
  .vars   - Muestra variables definidas
  .exit   - Sale del REPL

Sintaxis:
  source <id> = <valor>;           - Define una fuente
  transform <id> = <op>(<args>);   - Aplica una transformación
  sink <id> = <source>;            - Define una salida (se muestra el resultado)

Operaciones: sum, substract, multiply, divide, filter, order_asc, order_desc

Ejemplo:
  source x = 5;
  source y = 3;
  transform suma = sum(x, y);
  sink resultado = suma;
`;

/**
 * Extracts the identifier from a statement line.
 * Returns null if the line is not a valid statement.
 */
function extractIdentifier(line: string): string | null {
  const match = line.trim().match(/^(?:source|transform|sink)\s+(\w+)\s*=/);
  return match ? match[1] : null;
}

async function main() {
  const interpreter = new Interpreter();
  // Use a Map to track statements by identifier (allows redefinition)
  const statements: Map<string, string> = new Map();
  const sinkNames: Set<string> = new Set();

  console.log(`Dataflow REPL v${VERSION} - Escribe .help para ver comandos\n`);

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: "dataflow> ",
  });

  rl.prompt();

  rl.on("line", async (line: string) => {
    const trimmed = line.trim();

    // Handle commands
    if (trimmed === ".help") {
      console.log(HELP_TEXT);
      rl.prompt();
      return;
    }

    if (trimmed === ".clear") {
      statements.clear();
      sinkNames.clear();
      interpreter.reset();
      console.log("Estado limpiado.\n");
      rl.prompt();
      return;
    }

    if (trimmed === ".vars") {
      if (statements.size === 0) {
        console.log("No hay variables definidas.\n");
      } else {
        console.log("Variables definidas:");
        for (const [id, stmt] of statements) {
          console.log(`  ${id}: ${stmt.trim()}`);
        }
        console.log();
      }
      rl.prompt();
      return;
    }

    if (trimmed === ".exit" || trimmed === ".quit") {
      console.log("¡Hasta luego!");
      rl.close();
      return;
    }

    // Skip empty lines
    if (trimmed === "") {
      rl.prompt();
      return;
    }

    // Extract identifier from the statement
    const identifier = extractIdentifier(trimmed);
    if (!identifier) {
      console.error("Error: Sintaxis inválida. Usa: source/transform/sink <id> = ...");
      rl.prompt();
      return;
    }

    // Check if this is a sink
    const isSink = trimmed.startsWith("sink ");

    // Store the old statement in case we need to revert
    const oldStatement = statements.get(identifier);
    const wasInSinks = sinkNames.has(identifier);

    // Update or add the statement
    statements.set(identifier, trimmed);
    if (isSink) {
      sinkNames.add(identifier);
    } else if (wasInSinks) {
      // If redefining a sink as source/transform, remove from sinks
      sinkNames.delete(identifier);
    }

    // Build the program from all statements
    const program = Array.from(statements.values()).join("\n");

    try {
      const result = await interpreter.execute(program);

      if (result.errors.length > 0) {
        // Show errors and revert the change
        for (const err of result.errors) {
          console.error(`Error: ${err.message}`);
        }

        // Revert to old state
        if (oldStatement) {
          statements.set(identifier, oldStatement);
        } else {
          statements.delete(identifier);
        }

        if (wasInSinks) {
          sinkNames.add(identifier);
        } else {
          sinkNames.delete(identifier);
        }
      } else {
        // Show all sink values (they might have changed due to dependency updates)
        for (const name of sinkNames) {
          const value = result.results.get(name);
          if (value !== undefined) {
            console.log(`${name} = ${formatValue(value)}`);
          }
        }
      }
    } catch (err) {
      console.error(`Error interno: ${err}`);

      // Revert to old state
      if (oldStatement) {
        statements.set(identifier, oldStatement);
      } else {
        statements.delete(identifier);
      }

      if (wasInSinks) {
        sinkNames.add(identifier);
      } else {
        sinkNames.delete(identifier);
      }
    }

    rl.prompt();
  });

  rl.on("close", () => {
    process.exit(0);
  });
}

main().catch(console.error);
