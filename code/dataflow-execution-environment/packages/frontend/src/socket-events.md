# Eventos Socket.io (backend → frontend)

El frontend se conecta con `socket.io-client` a la URL definida en `VITE_SOCKET_URL` (o `window.location.origin` si no existe). El **backend** debe emitir los siguientes eventos para que la UI reaccione sin usar mouse.

---

## 1. Agregar nodos y edges al IDE (canvas)

### `addNode`

Agrega una carta/nodo al tablero de React Flow.

**Payload:**

```ts
{
  id: string;           // ej: "num-5-1234567890"
  type: "number" | "operator";
  position: { x: number; y: number };
  data: {
    value?: number;     // para type "number"
    operator?: "adicion" | "sustraccion";  // para type "operator"
    result?: number;    // opcional, se calcula en frontend
  };
}
```

**Ejemplo:**

```js
socket.emit("addNode", {
  id: "num-7-" + Date.now(),
  type: "number",
  position: { x: 100, y: 80 },
  data: { value: 7 },
});
```

### `addEdge`

Agrega una conexión entre dos nodos.

**Payload:**

```ts
{
  id?: string;           // opcional; si no se envía se genera uno
  source: string;        // id del nodo origen
  target: string;        // id del nodo destino
  sourceHandle?: string; // opcional
  targetHandle?: "a" | "b";  // para operadores: "a" o "b" (entradas)
}
```

**Ejemplo:**

```js
socket.emit("addEdge", {
  source: "num-3-1",
  target: "op-sum-1",
  targetHandle: "a",
});
```

---

## 2. Navegación (sin mouse)

El frontend escucha el evento **`navigate`** y cambia de ruta según el payload. Así los botones físicos (Menu, Configuración, Atrás, Jugar, Mundos, Niveles) pueden enviar un evento desde el backend y la app navega.

**Payload (uno de los dos):**

- **Ruta directa:** `{ path: string }`  
  Ej: `{ path: "/" }`, `{ path: "/juego/1" }`, `{ path: "/ide/sandbox" }`

- **Acción nombrada:** `{ to: string, worldId?: string }`

| `to`             | Acción        | Ruta resultante   |
|------------------|---------------|-------------------|
| `menu`           | Menú principal| `/`               |
| `configuracion`  | Configuración | `/configuracion`  |
| `atras`          | Atrás         | `navigate(-1)`    |
| `jugar`          | Jugar         | `/juego`          |
| `mundos`         | Mundos        | `/juego`          |
| `niveles`        | Niveles       | `/juego/:worldId` (requiere `worldId`) |

**Ejemplos:**

```js
// Ir al menú
socket.emit("navigate", { to: "menu" });
// o
socket.emit("navigate", { path: "/" });

// Ir a configuración
socket.emit("navigate", { to: "configuracion" });

// Atrás (historial)
socket.emit("navigate", { to: "atras" });

// Ir a Jugar / lista de mundos
socket.emit("navigate", { to: "jugar" });

// Ir a niveles del Mundo 1
socket.emit("navigate", { to: "niveles", worldId: "1" });

// Ruta directa al IDE sandbox
socket.emit("navigate", { path: "/ide/sandbox" });
```

---

## Nota

- Los eventos **`addNode`** y **`addEdge`** solo tienen efecto cuando la pantalla visible es el **IDE** (Dataflow). Si el usuario está en menú o configuración, los nodos/edges se agregan al estado y se verán al entrar al IDE.
- La conexión socket se inicia al cargar la app; el backend debe aceptar conexiones en la misma URL (o en `VITE_SOCKET_URL` si se configura).
