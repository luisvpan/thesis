import type { ValidationError } from "@dataflow/shared/types";

export const ERROR_MESSAGES = {
  DUPLICATE_IDENTIFIER: {
    message: (id: string) => `Identifier '${id}' already defined`,
    childMessage: (id: string) => `⚠️ ¡Ups! Ya tienes un bloque llamado "${id}". Cada bloque necesita un nombre único.`,
    suggestion: (id: string) => `Cambia el nombre de uno de los bloques "${id}"`,
    example: (id: string) => `[${id}_1] y [${id}_2] ✅`
  },
  UNDEFINED_IDENTIFIER: {
    message: (ref: string) => `Undefined identifier: ${ref}`,
    childMessage: (ref: string) => `⚠️ ¡Ups! No encuentro el bloque "${ref}".`,
    suggestion: () => "Asegúrate de que el bloque existe antes de conectarlo.",
    example: () => "Crea el bloque primero, luego conéctalo."
  },
  CYCLE_DETECTED: {
    message: () => "Graph contains a cycle",
    childMessage: () => "⚠️ ¡Ups! Hay un ciclo en el programa. Los bloques se conectan en círculo y no se puede calcular.",
    suggestion: () => "Busca dónde un bloque apunta a sí mismo y desconecta una línea.",
    example: () => "[A] → [B] → [C] ❌\n[A] → [B] ✅ (rompe el ciclo)"
  },
  UNKNOWN_OPERATION: {
    message: (operation: string) => `Unknown operation: ${operation}`,
    childMessage: (operation: string) => `⚠️ ¡Ups! No reconozco el bloque "${operation}".`,
    suggestion: () => `Revisa si el bloque está conectado correctamente.`,
    example: () => `Usa un bloque conocido como "+", "-", "×", "÷"`
  },
  WRONG_ARITY: {
    message: (operation: string, expectedArity: number, actualArity: number) => 
      `Operation ${operation} requires ${expectedArity} ${expectedArity === 1 ? 'input' : 'inputs'}, got ${actualArity}`,
    childMessage: (operation: string, expectedArity: number, actualArity: number) => 
      `⚠️ ¡Ups! El bloque "${operation}" necesita ${expectedArity} ${expectedArity === 1 ? 'entrada' : 'entradas'}. Solo conectaste ${actualArity}.`,
    suggestion: (operation: string, expectedArity: number, actualArity: number) => 
      `Conecta ${expectedArity - actualArity} bloque${expectedArity - actualArity === 1 ? '' : 's'} más al bloque "${operation}".`,
    example: (operation: string, arity: number) => {
      if (operation === "ADD") return "[5] + [3] ✅";
      if (operation === "FILTER") return "[formas] 🎨 [rojo] ✅";
      return `Conecta ${arity} bloques al bloque "${operation}" ✅`;
    }
  },
  TYPE_ERROR: {
    message: (operation: string, inputTypes: string[], property?: string) => 
      property 
        ? `No matching signature for operation ${operation} with input types [${inputTypes.join(", ")}] - missing '${property}' property`
        : `No matching signature for operation ${operation} with input types [${inputTypes.join(", ")}]`,
    childMessage: (operation: string, inputTypes: string[], signatures?: any) => {
      const numericOps = ["ADD", "SUBTRACT", "MULTIPLY", "DIVIDE"];
      const hasNumericInput = inputTypes.some(t => ["natural", "integer", "decimal", "fraction"].includes(t));
      const hasTextInput = inputTypes.includes("text");

      if (operation === "SORT") {
        return `⚠️ ¡Ups! SORT solo funciona con números (natural, integer, decimal, fraction).\nTu conjunto contiene "${inputTypes.join(", ")}".`;
      } else if (operation === "ALPHABETICAL_SORT") {
        return `⚠️ ¡Ups! ALPHABETICAL_SORT solo funciona con texto (text).\nTu conjunto contiene "${inputTypes.join(", ")}".`;
      } else if (numericOps.includes(operation) && hasNumericInput && hasTextInput) {
        return `⚠️ ¡Ups! El bloque "${operation}" necesita números, no palabras.`;
      } else if (operation.startsWith("FILTER_BY_") || operation.startsWith("COMPARE_BY_")) {
        const property = operation.split("_BY_")[1].toLowerCase();
        return `⚠️ ¡Ups! El bloque "${operation}" necesita elementos con propiedad '${property}'.`;
      }
      return `⚠️ ¡Ups! El bloque "${operation}" no funciona con estos tipos.`;
    },
    suggestion: (operation: string, inputTypes: string[], signatures?: any) => {
      if (operation === "SORT") {
        return `Conecta bloques de números al bloque SORT. Para ordenar palabras, usa ALPHABETICAL_SORT.`;
      } else if (operation === "ALPHABETICAL_SORT") {
        return `Conecta bloques de texto al bloque ALPHABETICAL_SORT. Para ordenar números, usa SORT.`;
      } else if (["ADD", "SUBTRACT", "MULTIPLY", "DIVIDE"].includes(operation)) {
        return `Conecta bloques de números (natural, integer, decimal, fraction) al bloque "${operation}".`;
      } else if (operation.startsWith("FILTER_BY_") || operation.startsWith("COMPARE_BY_")) {
        const property = operation.split("_BY_")[1].toLowerCase();
        return `Usa un tipo que tenga propiedad '${property}'.`;
      }
      return `Verifica que los tipos de entrada sean compatibles con el bloque "${operation}".`;
    },
    example: (operation: string, inputTypes: string[], signatures?: any) => {
      if (operation === "SORT") {
        return `[números 1, 5, 3] → SORT → [1, 3, 5] ✅`;
      } else if (operation === "ALPHABETICAL_SORT") {
        return `[texto "z", "a", "m"] → ALPHABETICAL_SORT → ["a", "m", "z"] ✅`;
      } else if (["ADD", "SUBTRACT", "MULTIPLY", "DIVIDE"].includes(operation)) {
        return `[número 5] + [número 3] ✅`;
      } else if (operation.startsWith("FILTER_BY_") || operation.startsWith("COMPARE_BY_")) {
        return `set<tipo con propiedad> → filtrados ✅`;
      }
      if (signatures) {
        return `Usa tipos compatibles: [${signatures.contracts.map((c: any) => c.inputTypes.join(", ")).join(" o ")}]`;
      }
      return `Usa tipos compatibles con el bloque "${operation}"`;
    }
  },
  TYPE_ERROR_INPUT: {
    childMessage: (operation: string, actualType: string, expectedType: string, inputId: string) => {
      const numericOps = ["ADD", "SUBTRACT", "MULTIPLY", "DIVIDE"];
      const isNumeric = ["natural", "integer", "decimal"].includes(actualType);
      const hasText = actualType === "text";

      if (operation === "ADD") {
        if (!isNumeric && hasText) {
          return `⚠️ ¡Ups! El bloque "${operation}" necesita números, no palabras. "${actualType}" no es un número.`;
        } else if (!isNumeric) {
          return `⚠️ ¡Ups! El bloque "${operation}" necesita ${expectedType}, recibiste "${actualType}".`;
        }
        return `⚠️ ¡Ups! El bloque "${operation}" recibió ${actualType} pero esperaba ${expectedType}. Verifica tus conexiones.`;
      } else if (operation === "SUBTRACT") {
        if (!isNumeric && hasText) {
          return `⚠️ ¡Ups! El bloque "${operation}" necesita números, no palabras. "${actualType}" no es un número.`;
        } else if (!isNumeric) {
          return `⚠️ ¡Ups! El bloque "${operation}" necesita ${expectedType}, recibiste "${actualType}".`;
        }
        return `⚠️ ¡Ups! El bloque "${operation}" recibió ${actualType} pero esperaba ${expectedType}. Verifica tus conexiones.`;
      } else if (operation === "MULTIPLY") {
        if (!isNumeric && hasText) {
          return `⚠️ ¡Ups! El bloque "${operation}" necesita números, no palabras. "${actualType}" no es un número.`;
        } else if (!isNumeric) {
          return `⚠️ ¡Ups! El bloque "${operation}" necesita ${expectedType}, recibiste "${actualType}".`;
        }
        return `⚠️ ¡Ups! El bloque "${operation}" recibió ${actualType} pero esperaba ${expectedType}. Verifica tus conexiones.`;
      } else if (operation === "DIVIDE") {
        if (!isNumeric && hasText) {
          return `⚠️ ¡Ups! El bloque "${operation}" necesita números, no palabras. "${actualType}" no es un número.`;
        } else if (!isNumeric) {
          return `⚠️ ¡Ups! El bloque "${operation}" necesita ${expectedType}, recibiste "${actualType}".`;
        }
        return `⚠️ ¡Ups! El bloque "${operation}" recibió ${actualType} pero esperaba ${expectedType}. Verifica tus conexiones.`;
      } else if (operation === "UNION") {
        const isSet = actualType.startsWith("set<");
        if (!isSet) {
          return `⚠️ ¡Ups! UNION solo funciona con conjuntos (set<...>).\nTu tipo es "${actualType}".`;
        }
        return `⚠️ ¡Ups! El bloque UNION necesita dos conjuntos del mismo tipo.`;
      } else if (operation === "INTERSECTION") {
        const isSet = actualType.startsWith("set<");
        if (!isSet) {
          return `⚠️ ¡Ups! INTERSECTION solo funciona con conjuntos (set<...>).\nTu tipo es "${actualType}".`;
        }
        return `⚠️ ¡Ups! El bloque INTERSECTION necesita dos conjuntos del mismo tipo.`;
      } else if (operation === "DIFFERENCE") {
        const isSet = actualType.startsWith("set<");
        if (!isSet) {
          return `⚠️ ¡Ups! DIFFERENCE solo funciona con conjuntos (set<...>).\nTu tipo es "${actualType}".`;
        }
        return `⚠️ ¡Ups! El bloque DIFFERENCE necesita dos conjuntos del mismo tipo.`;
      } else if (operation === "COMPLEMENT") {
        const isSet = actualType.startsWith("set<");
        if (!isSet) {
          return `⚠️ ¡Ups! COMPLEMENT solo funciona con conjuntos (set<...>).\nTu tipo es "${actualType}".`;
        }
        return `⚠️ ¡Ups! El bloque COMPLEMENT necesita un conjunto del tipo correcto.`;
      } else if (operation === "AND") {
        if (actualType !== "boolean") {
          return `⚠️ ¡Ups! El bloque AND solo funciona con valores verdadero/falso (boolean).\nTu tipo es "${actualType}".`;
        }
        return `⚠️ ¡Ups! El bloque AND recibió ${actualType} pero esperaba boolean.`;
      } else if (operation === "OR") {
        if (actualType !== "boolean") {
          return `⚠️ ¡Ups! El bloque OR solo funciona con valores verdadero/falso (boolean).\nTu tipo es "${actualType}".`;
        }
        return `⚠️ ¡Ups! El bloque OR recibió ${actualType} pero esperaba boolean.`;
      } else if (operation === "NOT") {
        if (actualType !== "boolean") {
          return `⚠️ ¡Ups! El bloque NOT solo funciona con valores verdadero/falso (boolean).\nTu tipo es "${actualType}".`;
        }
        return `⚠️ ¡Ups! El bloque NOT recibió ${actualType} pero esperaba boolean.`;
      } else if (operation === "NEXT") {
        const isStream = actualType.startsWith("stream<");
        if (!isStream) {
          return `⚠️ ¡Ups! NEXT solo funciona con flujos (stream<...>).\nTu tipo es "${actualType}".`;
        }
        return `⚠️ ¡Ups! El bloque NEXT recibió ${actualType} pero esperaba un flujo.`;
      } else if (operation === "FIRST") {
        const isStream = actualType.startsWith("stream<");
        if (!isStream) {
          return `⚠️ ¡Ups! FIRST solo funciona con flujos (stream<...>).\nTu tipo es "${actualType}".`;
        }
        return `⚠️ ¡Ups! El bloque FIRST recibió ${actualType} pero esperaba un flujo.`;
      } else if (operation === "ACCUMULATE") {
        const isStream = actualType.startsWith("stream<");
        if (!isStream) {
          return `⚠️ ¡Ups! ACCUMULATE solo funciona con flujos (stream<...>).\nTu tipo es "${actualType}".`;
        }
        return `⚠️ ¡Ups! El bloque ACCUMULATE recibió ${actualType} pero esperaba un flujo.`;
      } else if (operation === "FBY") {
        const isStream = actualType.startsWith("stream<");
        if (!isStream) {
          return `⚠️ ¡Ups! FBY necesita un valor inicial y un flujo (stream<...>).\nTu tipo es "${actualType}".`;
        }
        return `⚠️ ¡Ups! El bloque FBY necesita tipos correctos. Verifica los argumentos.`;
      } else if (operation === "SORT") {
        const isNumeric = ["natural", "integer", "decimal", "fraction"].includes(actualType);
        if (!isNumeric) {
          return `⚠️ ¡Ups! SORT solo funciona con números (natural, integer, decimal, fraction).\nTu conjunto contiene "${actualType}".`;
        }
        return `⚠️ ¡Ups! El bloque SORT recibió ${actualType} pero esperaba un número.`;
      } else if (operation === "ALPHABETICAL_SORT") {
        const isText = actualType === "text";
        if (!isText) {
          return `⚠️ ¡Ups! ALPHABETICAL_SORT solo funciona con texto (text).\nTu conjunto contiene "${actualType}".`;
        }
        return `⚠️ ¡Ups! El bloque ALPHABETICAL_SORT recibió ${actualType} pero esperaba texto.`;
      }
      return `⚠️ ¡Ups! El bloque "${operation}" espera ${expectedType}, recibiste "${actualType}".`;
    },
    suggestion: (operation: string, actualType: string, expectedType: string, inputId: string) => {
      const numericOps = ["ADD", "SUBTRACT", "MULTIPLY", "DIVIDE"];
      const isNumeric = ["natural", "integer", "decimal"].includes(actualType);
      const hasText = actualType === "text";

      if (operation === "ADD") {
        if (!isNumeric && hasText) {
          return `Conecta bloques de números (natural, integer, decimal) al bloque "${operation}".`;
        } else if (!isNumeric) {
          return `Conecta un bloque de tipo ${expectedType}.`;
        }
        return `Revisa que "${inputId}" está conectado a un bloque de tipo ${expectedType}.`;
      } else if (operation === "SUBTRACT") {
        if (!isNumeric && hasText) {
          return `Conecta bloques de números (natural, integer, decimal) al bloque "${operation}".`;
        } else if (!isNumeric) {
          return `Conecta un bloque de tipo ${expectedType}.`;
        }
        return `Revisa que "${inputId}" está conectado a un bloque de tipo ${expectedType}.`;
      } else if (operation === "MULTIPLY") {
        if (!isNumeric && hasText) {
          return `Conecta bloques de números (natural, integer, decimal) al bloque "${operation}".`;
        } else if (!isNumeric) {
          return `Conecta un bloque de tipo ${expectedType}.`;
        }
        return `Revisa que "${inputId}" está conectado a un bloque de tipo ${expectedType}.`;
      } else if (operation === "DIVIDE") {
        if (!isNumeric && hasText) {
          return `Conecta bloques de números (natural, integer, decimal) al bloque "${operation}".`;
        } else if (!isNumeric) {
          return `Conecta un bloque de tipo ${expectedType}.`;
        }
        return `Revisa que "${inputId}" está conectado a un bloque de tipo ${expectedType}.`;
      } else if (operation === "UNION") {
        const isSet = actualType.startsWith("set<");
        if (!isSet) {
          return `Conecta bloques de tipo conjunto al bloque UNION.`;
        }
        return `Asegúrate de que ambos conjuntos tengan el mismo tipo de elemento.`;
      } else if (operation === "INTERSECTION") {
        const isSet = actualType.startsWith("set<");
        if (!isSet) {
          return `Conecta bloques de tipo conjunto al bloque INTERSECTION.`;
        }
        return `Asegúrate de que ambos conjuntos tengan el mismo tipo de elemento.`;
      } else if (operation === "DIFFERENCE") {
        const isSet = actualType.startsWith("set<");
        if (!isSet) {
          return `Conecta bloques de tipo conjunto al bloque DIFFERENCE.`;
        }
        return `Asegúrate de que ambos conjuntos tengan el mismo tipo de elemento.`;
      } else if (operation === "COMPLEMENT") {
        const isSet = actualType.startsWith("set<");
        if (!isSet) {
          return `Conecta bloques de tipo conjunto al bloque COMPLEMENT.`;
        }
        return `Asegúrate de que el tipo sea correcto.`;
      } else if (operation === "AND" || operation === "OR" || operation === "NOT") {
        return `Conecta bloques de tipo boolean al bloque ${operation}.`;
      } else if (operation === "NEXT" || operation === "FIRST" || operation === "ACCUMULATE") {
        return `Conecta bloques de tipo flujo al bloque ${operation}.`;
      } else if (operation === "FBY") {
        return `El primer argumento debe ser un valor, el segundo debe ser un flujo.`;
      } else if (operation === "SORT") {
        return `Conecta bloques de números al bloque SORT. Para ordenar palabras, usa ALPHABETICAL_SORT.`;
      } else if (operation === "ALPHABETICAL_SORT") {
        return `Conecta bloques de texto al bloque ALPHABETICAL_SORT. Para ordenar números, usa SORT.`;
      }
      return `Revisa el tipo de "${inputId}". Debe ser ${expectedType}.`;
    },
    example: (operation: string, actualType: string, expectedType: string, inputId: string, arity?: number) => {
      if (operation === "ADD") {
        if (actualType === "text") {
          return `[número 5] + [número 3] ✅`;
        }
        return `[${expectedType} 5] + [${expectedType} 3] ✅`;
      } else if (operation === "SUBTRACT") {
        if (actualType === "text") {
          return `[número 5] - [número 3] ✅`;
        }
        return `[${expectedType} 5] - [${expectedType} 3] ✅`;
      } else if (operation === "MULTIPLY") {
        if (actualType === "text") {
          return `[número 5] × [número 3] ✅`;
        }
        return `[${expectedType} 5] × [${expectedType} 3] ✅`;
      } else if (operation === "DIVIDE") {
        if (actualType === "text") {
          return `[número 6] ÷ [número 2] ✅`;
        }
        return `[${expectedType} 6] ÷ [${expectedType} 2] ✅`;
      } else if (operation === "UNION") {
        return `[conjunto A] ∪ [conjunto B] → unión ✅`;
      } else if (operation === "INTERSECTION") {
        return `[conjunto A] ∩ [conjunto B] → intersección ✅`;
      } else if (operation === "DIFFERENCE") {
        return `[conjunto A] \\ [conjunto B] → diferencia ✅`;
      } else if (operation === "COMPLEMENT") {
        return `[conjunto] → COMPLEMENT → complemento ✅`;
      } else if (operation === "AND") {
        return `[verdadero] ∧ [verdadero] → verdadero ✅`;
      } else if (operation === "OR") {
        return `[verdadero] ∨ [falso] → verdadero ✅`;
      } else if (operation === "NOT") {
        return `¬ [verdadero] → falso ✅`;
      } else if (operation === "NEXT") {
        return `stream<número> → NEXT → siguiente valor ✅`;
      } else if (operation === "FIRST") {
        return `stream<número> → FIRST → primer valor ✅`;
      } else if (operation === "ACCUMULATE") {
        return `stream<número> → ACCUMULATE → acumulado ✅`;
      } else if (operation === "FBY") {
        return `[valor inicial] → FBY [flujo] → nuevo flujo ✅`;
      } else if (operation === "SORT") {
        return `[números 1, 5, 3] → SORT → [1, 3, 5] ✅`;
      } else if (operation === "ALPHABETICAL_SORT") {
        return `[texto "z", "a", "m"] → ALPHABETICAL_SORT → ["a", "m", "z"] ✅`;
      }
      if (arity) {
        if (operation === "ADD") return "[5] + [3] ✅";
        if (operation === "FILTER") return "[formas] 🎨 [rojo] ✅";
        return `Conecta ${arity} bloques al bloque "${operation}" ✅`;
      }
      return `${expectedType} → [${actualType}] ❌`;
    }
  },
  SET_HETEROGENEITY_ERROR: {
    message: () => "All elements in set must have same type",
    childMessage: () => "⚠️ ¡Ups! Un conjunto solo puede tener elementos del mismo tipo.",
    suggestion: (elementType: string) => `Asegúrate de que todos los elementos sean de tipo ${elementType}.`,
    example: (elementType: string) => `set<${elementType}> = {elemento 1, elemento 2} ✅`
  },
  LITERAL_TYPE_MISMATCH: {
    message: (index: number, valueType: string, elementType: string, value?: unknown) => 
      index !== undefined 
        ? `Set element at index ${index} has type ${valueType}, but set expects ${elementType}`
        : `Literal value ${JSON.stringify(value)} has type ${valueType}, but declared type is ${elementType}`,
    childMessage: (index: number, elementType: string) => 
      index !== undefined 
        ? `⚠️ ¡Ups! El elemento en posición ${index} no coincide con el tipo ${elementType}.`
        : "⚠️ ¡Ups! El valor no coincide con el tipo.",
    suggestion: (elementType: string) => `Cambia el elemento o usa el bloque de tipo correcto.`,
    example: (elementType: string) => `Para "set<${elementType}>", usa elementos de tipo ${elementType}`
  },
  MISSING_OUTPUT_NODE: {
    message: () => "Program must have at least one output node",
    childMessage: () => "⚠️ ¡Ups! Tu programa necesita al menos un bloque de salida (output).",
    suggestion: () => "Añade un bloque de salida para ver el resultado de tu programa.",
    example: () => "[resultado] → [output] ✅"
  },
  FILTER_COMPARE_PROPERTY_ERROR: {
    childMessage: (operation: string, property: string, elementType: string) => 
      `⚠️ ¡Ups! El bloque "${operation}" necesita elementos con propiedad "${property}". "${elementType}" no tiene "${property}".`,
    suggestion: (property: string, expectedTypeWithProperty: string) => 
      `Usa un tipo que tenga propiedad "${property}" (ej. ${expectedTypeWithProperty}).`,
    example: (expectedTypeWithProperty: string, property: string) => 
      `${expectedTypeWithProperty} [elemento con ${property}] → filtrados ✅`
  }
};
