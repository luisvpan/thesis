# REQUERIMIENTOS FUNCIONALES Y NO FUNCIONALES POR CONTENEDOR

---

## **CONTENEDOR 1: COMPUTER VISION MANAGER (CVM)**

### **Descripción**
Módulo de percepción del mundo físico que detecta bloques tangibles y trazos gestuales, los digitaliza y envía al DLEE para compilación.

### **Tecnologías**
- **Lenguaje:** Python 3.11+
- **Frameworks:** OpenCV 4.8+, NumPy, PyZMQ (WebSocket)
- **Hardware:** Kinect v2 SDK

---

### **REQUERIMIENTOS FUNCIONALES**

#### **RF-CVM-1: Detección de bloques tangibles**
**Descripción:** El sistema debe detectar bloques tangibles mediante marcadores ArUco.

**Criterios de aceptación:**
- Detectar bloques con marcadores ArUco de 6x6 bits
- Extraer ID, tipo, posición (x, y, z) y rotación del bloque
- Soportar hasta 20 bloques simultáneos en el campo visual
- Actualizar detecciones a mínimo 15 FPS

**Entrada:** Frame RGB-D del Kinect (1920x1080 @ 30 FPS)

**Salida:** 
```json
{
  "blocks": [
    {
      "id": "block_1",
      "type": "data_source",
      "subtype": "integer",
      "value": 3,
      "position": [100.5, 200.3, 10.2],
      "rotation": 45.0,
      "confidence": 0.98
    }
  ]
}
```

**Prioridad:** Alta

---

#### **RF-CVM-2: Reconocimiento de trazos gestuales**
**Descripción:** El sistema debe detectar trazos dibujados con el dedo sobre la superficie.

**Criterios de aceptación:**
- Detectar trazos mediante análisis de movimiento en espacio de color HSV
- Capturar path del trazo con mínimo 10 puntos por segundo
- Identificar bloque inicial y final del trazo mediante proximidad
- Filtrar ruido y trazos accidentales (< 5 cm de longitud)

**Entrada:** Frame RGB del Kinect con detección de movimiento

**Salida:**
```json
{
  "traces": [
    {
      "id": "trace_1",
      "path": [[100, 200], [150, 225], [200, 250]],
      "startBlockId": "block_1",
      "endBlockId": "block_2",
      "timestamp": "2026-01-24T10:30:15Z"
    }
  ]
}
```

**Prioridad:** Alta

---

#### **RF-CVM-3: Mapeo espacial**
**Descripción:** El sistema debe mapear coordenadas físicas 3D a coordenadas lógicas 2D.

**Criterios de aceptación:**
- Calibrar superficie de trabajo mediante 4 puntos de referencia
- Transformar coordenadas 3D (x, y, z) a 2D (x_logical, y_logical)
- Compensar distorsión de proyección y perspectiva
- Mantener precisión de ±2 cm en el mapeo

**Entrada:** Coordenadas físicas 3D + matriz de calibración

**Salida:** Coordenadas lógicas 2D en espacio normalizado [0, 1000] x [0, 1000]

**Prioridad:** Media

---

#### **RF-CVM-4: Serialización a JSON**
**Descripción:** El sistema debe serializar detecciones al formato JSON del programa dataflow.

**Criterios de aceptación:**
- Generar JSON conforme al esquema `DataflowProgram` v1.0
- Incluir metadatos: timestamp, userId, activityId
- Validar JSON contra esquema antes de enviar
- Comprimir JSON si excede 50 KB

**Entrada:** Bloques detectados + trazos + metadata

**Salida:** JSON serializado del programa dataflow completo

**Prioridad:** Alta

---

#### **RF-CVM-5: Comunicación con DLEE**
**Descripción:** El sistema debe enviar programas detectados al DLEE vía WebSocket.

**Criterios de aceptación:**
- Establecer conexión WebSocket persistente con DLEE
- Enviar mensajes JSON-RPC con método `program.compile`
- Recibir respuestas de compilación (errores/warnings)
- Reconectar automáticamente si conexión se pierde

**Entrada:** JSON del programa dataflow

**Salida:** Mensaje JSON-RPC enviado al DLEE

**Prioridad:** Alta

---

### **REQUERIMIENTOS NO FUNCIONALES**

#### **RNF-CVM-1: Latencia de detección**
El sistema debe procesar frames y detectar bloques/trazos con latencia máxima de 100ms desde captura hasta emisión JSON.

**Métrica:** Tiempo promedio de procesamiento ≤ 100ms

---

#### **RNF-CVM-2: Precisión de detección**
El sistema debe detectar bloques con precisión ≥ 95% bajo condiciones de iluminación controlada (500-1000 lux).

**Métrica:** 
- True Positive Rate ≥ 95%
- False Positive Rate ≤ 5%

---

#### **RNF-CVM-3: Robustez ante oclusión**
El sistema debe mantener detección de bloques con hasta 30% de oclusión parcial.

**Métrica:** Detección exitosa con oclusión ≤ 30% del marcador

---

#### **RNF-CVM-4: Escalabilidad de bloques**
El sistema debe soportar detección de 1 a 20 bloques sin degradación significativa de rendimiento.

**Métrica:** Latencia aumenta máximo 5ms por cada bloque adicional

---

#### **RNF-CVM-5: Disponibilidad**
El sistema debe estar disponible el 99% del tiempo de sesión de usuario.

**Métrica:** Uptime ≥ 99% durante sesiones de 30 minutos

---

#### **RNF-CVM-6: Configurabilidad**
El sistema debe permitir configurar parámetros de detección sin recompilar.

**Parámetros configurables:**
- Umbral de confianza mínimo (0.0 - 1.0)
- Tamaño mínimo de marcador (píxeles)
- Intervalo de actualización (FPS)
- Color de trazo gestual (HSV range)

---

---

## **CONTENEDOR 2: DATAFLOW LANGUAGE EXECUTION ENVIRONMENT (DLEE)**

### **Descripción**
Compilador y runtime del lenguaje dataflow. Parsea, valida, optimiza y ejecuta programas, reportando estado de ejecución al IDE.

### **Tecnologías**
- **Lenguaje:** TypeScript 5.0+ / Node.js 20+
- **Parser:** Custom recursive-descent parser
- **Comunicación:** WebSocket (ws library), LSP

---

### **REQUERIMIENTOS FUNCIONALES**

#### **RF-DLEE-1: Parsear JSON a AST**
**Descripción:** El sistema debe parsear JSON del programa dataflow a Abstract Syntax Tree (AST).

**Criterios de aceptación:**
- Soportar formato JSON conforme a esquema `DataflowProgram` v1.0
- Detectar errores de sintaxis JSON
- Construir AST con nodos: DataSource, Transformation, Output, Stream, Accumulator
- Preservar metadatos de posición y blockId

**Entrada:** JSON string del programa

**Salida:** Objeto `DataflowGraph` en memoria

**Prioridad:** Alta

---

#### **RF-DLEE-2: Validación semántica**
**Descripción:** El sistema debe validar semánticamente el programa dataflow.

**Criterios de aceptación:**
- **Validar DAG:** Detectar ciclos en el grafo, reportar nodos involucrados
- **Type checking:** Verificar compatibilidad de tipos entre nodos conectados
- **Arity checking:** Validar que transformaciones reciban número correcto de entradas
- **Connectivity:** Detectar nodos sin conexiones de entrada (excepto DataSource)

**Entrada:** `DataflowGraph`

**Salida:** 
```typescript
{
  valid: boolean,
  errors: Array<{
    type: "CycleError" | "TypeError" | "ArityError" | "ConnectivityError",
    message: string,
    nodeId?: string,
    edgeId?: string
  }>,
  warnings: Array<{...}>
}
```

**Prioridad:** Alta

---

#### **RF-DLEE-3: Optimización del IR**
**Descripción:** El sistema debe optimizar el IR mediante transformaciones.

**Criterios de aceptación:**
- **Constant Folding:** Evaluar operaciones con constantes en compile-time
- **Dead Code Elimination:** Eliminar nodos no alcanzables desde outputs
- **Common Subexpression Elimination:** Reutilizar cálculos duplicados

**Entrada:** `DataflowGraph` validado

**Salida:** `DataflowGraph` optimizado

**Prioridad:** Media

---

#### **RF-DLEE-4: Ordenamiento topológico**
**Descripción:** El sistema debe ordenar nodos para evaluación según dependencias.

**Criterios de aceptación:**
- Implementar algoritmo de ordenamiento topológico (Kahn o DFS)
- Garantizar que nodos se evalúen solo cuando sus dependencias estén listas
- Generar lista ordenada de nodos

**Entrada:** `DataflowGraph`

**Salida:** Array de `DataflowNode` ordenados

**Prioridad:** Alta

---

#### **RF-DLEE-5: Evaluación de programa**
**Descripción:** El sistema debe evaluar el programa dataflow y producir resultados.

**Criterios de aceptación:**
- Evaluar nodos en orden topológico
- Propagar datos a través de arcos
- Ejecutar operaciones: ADD, SUBTRACT, MULTIPLY, DIVIDE, COUNT, etc.
- Manejar tipos de datos: integer, float, boolean, set, sequence
- Trackear estado de cada nodo: waiting, processing, completed, error

**Entrada:** `DataflowGraph` optimizado

**Salida:** 
```typescript
{
  success: boolean,
  outputs: Map<string, any>,  // nodeId -> result
  trace: ExecutionTrace
}
```

**Prioridad:** Alta

---

#### **RF-DLEE-6: Gestión de streams (Nivel 5)**
**Descripción:** El sistema debe gestionar bloques especiales Stream y Accumulator.

**Criterios de aceptación:**
- **Stream:** Generar secuencias de valores según patrón (uniform, alternating, random)
- **Accumulator:** Mantener estado acumulativo entre evaluaciones
- **Timing:** Emitir valores de stream según intervalo configurado
- **Reset:** Limpiar estado de streams/accumulators al reiniciar programa

**Entrada:** Nodos Stream/Accumulator

**Salida:** Valores generados/acumulados

**Prioridad:** Baja (Nivel 5 es avanzado)

---

#### **RF-DLEE-7: State tracking**
**Descripción:** El sistema debe trackear estado de ejecución para visualización.

**Criterios de aceptación:**
- Trackear estado de cada nodo: waiting, processing, completed, error
- Trackear datos fluyendo por cada arco
- Generar trace de ejecución con timestamps
- Emitir eventos de cambio de estado

**Entrada:** Evaluación en progreso

**Salida:** Stream de eventos:
```typescript
{
  type: "node_state_change" | "data_flow",
  nodeId?: string,
  edgeId?: string,
  state?: NodeState,
  data?: any,
  timestamp: number
}
```

**Prioridad:** Alta

---

#### **RF-DLEE-8: Protocolo LSP**
**Descripción:** El sistema debe implementar Language Server Protocol para validación en tiempo real.

**Criterios de aceptación:**
- Implementar métodos LSP: `textDocument/didChange`, `textDocument/publishDiagnostics`
- Recibir programas incrementales desde IDE
- Validar y emitir diagnósticos (errores/warnings) al IDE
- Mantener sesión LSP persistente

**Entrada:** Mensajes LSP desde IDE

**Salida:** Diagnósticos LSP al IDE

**Prioridad:** Media

---

#### **RF-DLEE-9: API pública**
**Descripción:** El sistema debe exponer API pública para compilar y ejecutar programas.

**Métodos requeridos:**
```typescript
compile(source: string, format: "json" | "textual"): CompilationResult
execute(ir: DataflowGraph): ExecutionResult
run(source: string): ExecutionResult
validate(source: string): ValidationResult
step(ir: DataflowGraph): StepResult  // Ejecución paso a paso
```

**Prioridad:** Alta

---

### **REQUERIMIENTOS NO FUNCIONALES**

#### **RNF-DLEE-1: Latencia de compilación**
El sistema debe compilar programas con latencia ≤ 500ms para programas de hasta 20 nodos.

**Métrica:** Tiempo de compilación promedio ≤ 500ms

---

#### **RNF-DLEE-2: Latencia de evaluación**
El sistema debe evaluar programas con latencia ≤ 200ms para programas de hasta 20 nodos.

**Métrica:** Tiempo de evaluación promedio ≤ 200ms

---

#### **RNF-DLEE-3: Escalabilidad**
El sistema debe soportar programas de 1 a 50 nodos sin degradación exponencial de rendimiento.

**Métrica:** Complejidad temporal O(V + E) donde V = nodos, E = arcos

---

#### **RNF-DLEE-4: Correctitud**
El sistema debe producir resultados correctos para el 100% de programas válidos.

**Métrica:** 
- 0 resultados incorrectos en suite de pruebas
- Cobertura de código ≥ 90%

---

#### **RNF-DLEE-5: Manejo de errores**
El sistema debe detectar y reportar errores de forma clara y accionable.

**Criterios:**
- Mensajes de error en lenguaje natural apropiado para niños 8-9 años
- Indicar ubicación exacta del error (nodeId, edgeId)
- Sugerir solución cuando sea posible

---

#### **RNF-DLEE-6: Disponibilidad**
El sistema debe estar disponible el 99.5% del tiempo.

**Métrica:** Uptime ≥ 99.5%

---

#### **RNF-DLEE-7: Extensibilidad**
El sistema debe permitir agregar nuevas operaciones sin modificar core del compilador.

**Mecanismo:** Registro de operaciones mediante plugin system

---

---

## **CONTENEDOR 3: INTEGRATED DEVELOPMENT ENVIRONMENT (IDE)**

### **Descripción**
Interfaz gráfica principal que presenta visualización RA, gestiona actividades, muestra feedback y coordina interacción con otros componentes.

### **Tecnologías**
- **Frontend:** React 18+, TypeScript 5.0+
- **Rendering:** Canvas API, WebGL (Three.js para 3D)
- **Comunicación:** WebSocket, REST API (Axios)

---

### **REQUERIMIENTOS FUNCIONALES**

#### **RF-IDE-1: Menú principal**
**Descripción:** El sistema debe presentar menú principal con opciones de navegación.

**Criterios de aceptación:**
- Mostrar opciones: Actividades, Mi Progreso, Configuración, Salir
- Soportar navegación por teclado y táctil
- Animaciones suaves de transición (≤ 300ms)

**Prioridad:** Media

---

#### **RF-IDE-2: Selector de actividades**
**Descripción:** El sistema debe permitir seleccionar actividades disponibles.

**Criterios de aceptación:**
- Obtener lista de actividades desde Activities Database vía REST API
- Mostrar tarjetas con: título, nivel, conceptos, progreso
- Filtrar por nivel (1-5)
- Indicar actividades completadas vs. pendientes

**Entrada:** GET `/api/activities?level=2&userId=student_123`

**Salida:** Renderización de tarjetas de actividades

**Prioridad:** Alta

---

#### **RF-IDE-3: Workspace Editor**
**Descripción:** El sistema debe proporcionar canvas interactivo para construir redes.

**Criterios de aceptación:**
- Mostrar área delimitada para colocar bloques físicos
- Renderizar overlays visuales de bloques detectados por CVM
- Permitir trazar conexiones digitalmente (opcional: complementa gestos físicos)
- Mostrar instrucciones de la actividad actual

**Prioridad:** Alta

---

#### **RF-IDE-4: Renderizador de Realidad Aumentada**
**Descripción:** El sistema debe renderizar visualización RA proyectada sobre superficie.

**Criterios de aceptación:**
- Renderizar nodos (bloques) con iconografía y etiquetas
- Renderizar arcos (conexiones) con dirección indicada
- Animar flujo de datos durante ejecución (partículas moviéndose por arcos)
- Resaltar nodos según estado: esperando, procesando, completado, error
- Frame rate ≥ 30 FPS

**Entrada:** Estado de ejecución desde DLEE

**Salida:** Frame 1920x1080 enviado al proyector

**Prioridad:** Alta

---

#### **RF-IDE-5: Feedback Manager**
**Descripción:** El sistema debe mostrar retroalimentación visual y auditiva.

**Criterios de aceptación:**
- Mostrar errores de compilación como mensajes flotantes sobre nodos problemáticos
- Mostrar hints contextuales si niño lleva > 2 minutos sin progresar
- Reproducir sonidos: éxito, error, hint, progreso
- Usar lenguaje apropiado para edad 6-9 años

**Entrada:** Diagnósticos LSP desde DLEE

**Salida:** Mensajes visuales + audio

**Prioridad:** Alta

---

#### **RF-IDE-6: Execution Visualizer**
**Descripción:** El sistema debe animar ejecución paso a paso del programa.

**Criterios de aceptación:**
- Mostrar animación de datos fluyendo por arcos
- Resaltar nodo actualmente procesándose
- Mostrar resultado parcial en cada nodo
- Controles: Pausar, Continuar, Reiniciar, Velocidad (0.5x, 1x, 2x)

**Entrada:** Stream de eventos de ejecución desde DLEE

**Salida:** Animaciones sincronizadas con ejecución

**Prioridad:** Media

---

#### **RF-IDE-7: Teacher Dashboard**
**Descripción:** El sistema debe proporcionar panel para docentes.

**Criterios de aceptación:**
- Mostrar lista de estudiantes activos
- Mostrar progreso individual: actividades completadas, intentos, score
- Permitir crear/editar actividades
- Exportar reportes en CSV

**Prioridad:** Baja (para MVP)

---

#### **RF-IDE-8: Guardar progreso**
**Descripción:** El sistema debe guardar progreso del estudiante automáticamente.

**Criterios de aceptación:**
- Guardar programa construido cada 30 segundos (autosave)
- Guardar resultado de actividad al completarla
- Enviar datos a Activities Database vía POST `/api/progress`
- Mostrar indicador de "guardando..."

**Prioridad:** Alta

---

### **REQUERIMIENTOS NO FUNCIONALES**

#### **RNF-IDE-1: Latencia de renderizado**
El sistema debe renderizar frames de RA con latencia ≤ 33ms (30 FPS mínimo).

**Métrica:** Frame time promedio ≤ 33ms

---

#### **RNF-IDE-2: Responsividad de UI**
El sistema debe responder a interacciones del usuario en ≤ 100ms.

**Métrica:** Tiempo desde input hasta feedback visual ≤ 100ms

---

#### **RNF-IDE-3: Usabilidad para niños**
El sistema debe ser usable por niños de 6-9 años sin asistencia del docente.

**Criterios:**
- Iconografía clara y grande (≥ 64x64 px)
- Texto mínimo, pictórico cuando sea posible
- Retroalimentación multimodal (visual + audio)
- Tolerancia a errores (permitir deshacer)

**Métrica:** 80% de niños completan primera actividad sin ayuda

---

#### **RNF-IDE-4: Accesibilidad**
El sistema debe cumplir con WCAG 2.1 nivel AA para accesibilidad.

**Criterios:**
- Contraste de color ≥ 4.5:1
- Tamaño de texto ajustable
- Soporte para navegación por teclado

---

#### **RNF-IDE-5: Performance**
El sistema debe funcionar fluidamente en hardware modesto.

**Hardware mínimo:**
- CPU: Intel i5 (8va gen) o equivalente
- RAM: 8 GB
- GPU: Integrada con soporte WebGL 2.0

**Métrica:** FPS ≥ 30 en hardware mínimo

---

#### **RNF-IDE-6: Compatibilidad**
El sistema debe funcionar en navegadores modernos.

**Navegadores soportados:**
- Chrome 120+
- Firefox 120+
- Edge 120+

---

---

## **CONTENEDOR 4: ACTIVITIES DATABASE (ADB)**

### **Descripción**
Base de datos relacional que almacena actividades pedagógicas, progreso de usuarios y configuraciones del sistema.

### **Tecnologías**
- **Base de datos:** PostgreSQL 15+
- **API:** Node.js + Express + TypeORM
- **Autenticación:** JWT

---

### **REQUERIMIENTOS FUNCIONALES**

#### **RF-ADB-1: Almacenar actividades**
**Descripción:** El sistema debe almacenar actividades pedagógicas.

**Esquema de tabla `activities`:**
```sql
CREATE TABLE activities (
  id UUID PRIMARY KEY,
  title VARCHAR(255) NOT NULL,
  description TEXT,
  level INTEGER NOT NULL CHECK (level BETWEEN 1 AND 5),
  concepts TEXT[],  -- Array de conceptos PC
  required_blocks JSONB,  -- Bloques necesarios
  expected_output JSONB,  -- Resultado esperado
  validation_rules JSONB,  -- Reglas de validación
  created_at TIMESTAMP DEFAULT NOW(),
  created_by UUID REFERENCES users(id)
);
```

**Prioridad:** Alta

---

#### **RF-ADB-2: Almacenar progreso de usuarios**
**Descripción:** El sistema debe almacenar progreso de cada estudiante.

**Esquema de tabla `progress`:**
```sql
CREATE TABLE progress (
  id UUID PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  activity_id UUID REFERENCES activities(id),
  program_json JSONB,  -- Programa construido
  completed BOOLEAN DEFAULT FALSE,
  score INTEGER CHECK (score BETWEEN 0 AND 100),
  attempts INTEGER DEFAULT 0,
  time_spent_seconds INTEGER,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  UNIQUE (user_id, activity_id)
);
```

**Prioridad:** Alta

---

#### **RF-ADB-3: API REST - Obtener actividades**
**Descripción:** El sistema debe exponer endpoint para obtener actividades.

**Endpoint:** `GET /api/activities`

**Query params:**
- `level` (opcional): Filtrar por nivel
- `concept` (opcional): Filtrar por concepto PC
- `userId` (opcional): Incluir progreso del usuario

**Respuesta:**
```json
{
  "activities": [
    {
      "id": "uuid",
      "title": "Sumar frutas",
      "level": 2,
      "concepts": ["transformación", "fusión de flujos"],
      "userProgress": {
        "completed": true,
        "score": 85
      }
    }
  ]
}
```

**Prioridad:** Alta

---

#### **RF-ADB-4: API REST - Guardar progreso**
**Descripción:** El sistema debe exponer endpoint para guardar progreso.

**Endpoint:** `POST /api/progress`

**Body:**
```json
{
  "userId": "uuid",
  "activityId": "uuid",
  "programJson": {...},
  "completed": true,
  "score": 90,
  "timeSpentSeconds": 180
}
```

**Respuesta:**
```json
{
  "success": true,
  "progress": {
    "id": "uuid",
    "attempts": 3,
    "bestScore": 90
  }
}
```

**Prioridad:** Alta

---

#### **RF-ADB-5: API REST - Obtener progreso de usuario**
**Descripción:** El sistema debe exponer endpoint para obtener progreso completo de un usuario.

**Endpoint:** `GET /api/progress/:userId`

**Respuesta:**
```json
{
  "userId": "uuid",
  "totalActivities": 25,
  "completedActivities": 12,
  "averageScore": 78,
  "progress": [
    {
      "activityId": "uuid",
      "activityTitle": "Sumar frutas",
      "completed": true,
      "score": 85,
      "attempts": 2,
      "lastAttempt": "2026-01-24T10:30:00Z"
    }
  ]
}
```

**Prioridad:** Media

---

#### **RF-ADB-6: Autenticación**
**Descripción:** El sistema debe autenticar usuarios (estudiantes y docentes).

**Criterios de aceptación:**
- Soportar login con usuario/contraseña
- Emitir JWT válido por 24 horas
- Refrescar token automáticamente

**Endpoints:**
- `POST /api/auth/login`
- `POST /api/auth/refresh`

**Prioridad:** Media

---

### **REQUERIMIENTOS NO FUNCIONALES**

#### **RNF-ADB-1: Latencia de queries**
El sistema debe responder queries en ≤ 200ms para el 95% de requests.

**Métrica:** P95 latency ≤ 200ms

---

#### **RNF-ADB-2: Throughput**
El sistema debe soportar ≥ 100 requests/segundo concurrentes.

**Métrica:** Throughput ≥ 100 req/s sin degradación

---

#### **RNF-ADB-3: Disponibilidad**
El sistema debe estar disponible el 99.9% del tiempo.

**Métrica:** Uptime ≥ 99.9%

---

#### **RNF-ADB-4: Integridad de datos**
El sistema debe garantizar integridad referencial y consistencia de datos.

**Mecanismos:**
- Foreign keys habilitadas
- Transacciones ACID
- Backups diarios automáticos

---

#### **RNF-ADB-5: Escalabilidad**
El sistema debe soportar hasta 1000 usuarios concurrentes sin degradación.

**Métrica:** Performance estable con 1-1000 usuarios

---

#### **RNF-ADB-6: Seguridad**
El sistema debe proteger datos sensibles de usuarios.

**Criterios:**
- Contraseñas hasheadas con bcrypt (cost factor ≥ 10)
- Comunicación HTTPS obligatoria
- Tokens JWT firmados con RS256
- SQL injection prevention (parametrized queries)

---

#### **RNF-ADB-7: Auditabilidad**
El sistema debe registrar todas las operaciones de escritura.

**Mecanismo:** Tabla de auditoría con: usuario, acción, timestamp, IP

---

---

## **RESUMEN DE PRIORIDADES**

### **Alta prioridad (MVP):**
- CVM: RF-1, RF-2, RF-4, RF-5
- DLEE: RF-1, RF-2, RF-4, RF-5, RF-7, RF-9
- IDE: RF-2, RF-3, RF-4, RF-5, RF-8
- ADB: RF-1, RF-2, RF-3, RF-4

### **Media prioridad (Post-MVP):**
- CVM: RF-3
- DLEE: RF-3, RF-8
- IDE: RF-6, RF-7
- ADB: RF-5, RF-6

### **Baja prioridad (Futuro):**
- DLEE: RF-6 (Streams - Nivel 5)
- IDE: RF-7 (Teacher Dashboard completo)

---

## **MÉTRICAS CLAVE DEL SISTEMA**

| Métrica | Objetivo | Prioridad |
|---------|----------|-----------|
| Latencia end-to-end (gesto → visualización) | ≤ 500ms | Alta |
| FPS de visualización RA | ≥ 30 FPS | Alta |
| Precisión de detección de bloques | ≥ 95% | Alta |
| Tiempo de compilación | ≤ 500ms | Alta |
| Uptime del sistema | ≥ 99% | Media |
| Usuarios concurrentes soportados | 100 | Media |

---

**Documento versión 1.0**  
**Fecha:** 2026-01-24  
**Autores:** Equipo de Desarrollo
