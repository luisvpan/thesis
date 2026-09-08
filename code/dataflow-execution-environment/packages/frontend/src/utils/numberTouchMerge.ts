/**
 * Fusión de cartas numéricas cuyas áreas (AABB 208×208) se tocan en el lienzo.
 * Los dígitos se concatenan de izquierda a derecha: 1 + 2 → 12, + 3 → 123.
 */

import type { SourceFlowNodeData } from "@/components/dataflow";
import type { DataflowNode } from "@/contexts/node/types";
import {
  doAxisAlignedBoundsOverlap,
  getFlowCardBounds,
} from "./arrayZoneGeometry";

/** SourceFlowNodeData con variant: 'number' */
type NumberSourceData = Extract<SourceFlowNodeData, { variant: 'number' }>;

export type NumberTouchGroup = {
  /** Nodo más a la izquierda; identificador canónico para el programa y header visible. */
  primaryId: string;
  /** Nodo más a la derecha; único con handle de salida visible. */
  tailId: string;
  memberIds: string[];
  mergedValue: number;
};

function isNumberSourceNode(n: DataflowNode): n is DataflowNode & { type: "source" } {
  return n.type === "source" && (n.data as SourceFlowNodeData).variant === "number";
}

function readDigitValue(data: NumberSourceData): number {
  const d = data.digitValue ?? data.value ?? 0;
  return Math.max(0, Math.min(9, Math.trunc(d)));
}

/** Grupos de cartas numéricas con áreas que se tocan (componentes conexas). */
export function getNumberTouchGroups(nodes: DataflowNode[]): NumberTouchGroup[] {
  const numberNodes = nodes.filter(isNumberSourceNode);
  if (numberNodes.length === 0) return [];

  const parent = new Map<string, string>();
  const find = (id: string): string => {
    let root = id;
    while (parent.get(root) !== root) {
      root = parent.get(root)!;
    }
    let cur = id;
    while (parent.get(cur) !== root) {
      const next = parent.get(cur)!;
      parent.set(cur, root);
      cur = next;
    }
    return root;
  };
  const unite = (a: string, b: string) => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent.set(rb, ra);
  };

  for (const n of numberNodes) {
    parent.set(n.id, n.id);
  }

  for (let i = 0; i < numberNodes.length; i++) {
    const bi = getFlowCardBounds(numberNodes[i]);
    for (let j = i + 1; j < numberNodes.length; j++) {
      if (doAxisAlignedBoundsOverlap(bi, getFlowCardBounds(numberNodes[j]))) {
        unite(numberNodes[i].id, numberNodes[j].id);
      }
    }
  }

  const byRoot = new Map<string, DataflowNode[]>();
  for (const n of numberNodes) {
    const root = find(n.id);
    const list = byRoot.get(root) ?? [];
    list.push(n);
    byRoot.set(root, list);
  }

  const groups: NumberTouchGroup[] = [];
  for (const members of byRoot.values()) {
    const sorted = [...members].sort((a, b) => a.position.x - b.position.x);
    const digits = sorted.map((n) =>
      readDigitValue(n.data as NumberSourceData)
    );
    const mergedValue = Number.parseInt(
      digits.map(String).join(""),
      10
    );
    groups.push({
      primaryId: sorted[0].id,
      tailId: sorted[sorted.length - 1].id,
      memberIds: sorted.map((n) => n.id),
      mergedValue: Number.isNaN(mergedValue) ? digits[0] : mergedValue,
    });
  }

  return groups;
}

export function buildNumberTouchLookup(
  groups: NumberTouchGroup[]
): {
  primaryByMemberId: Map<string, string>;
  mergedValueByMemberId: Map<string, number>;
} {
  const primaryByMemberId = new Map<string, string>();
  const mergedValueByMemberId = new Map<string, number>();
  for (const g of groups) {
    for (const id of g.memberIds) {
      primaryByMemberId.set(id, g.primaryId);
      mergedValueByMemberId.set(id, g.mergedValue);
    }
  }
  return { primaryByMemberId, mergedValueByMemberId };
}

/** Valor mostrado / enviado al intérprete para un nodo numérico. */
export function getEffectiveNumberValue(
  nodeId: string,
  data: NumberSourceData,
  nodes: DataflowNode[]
): number {
  const { mergedValueByMemberId } = buildNumberTouchLookup(
    getNumberTouchGroups(nodes)
  );
  return mergedValueByMemberId.get(nodeId) ?? data.value ?? 0;
}

/** ID de fuente canónica (evita duplicar literales en el programa). */
export function resolveNumberSourceId(
  nodeId: string,
  nodes: DataflowNode[]
): string {
  const { primaryByMemberId } = buildNumberTouchLookup(
    getNumberTouchGroups(nodes)
  );
  return primaryByMemberId.get(nodeId) ?? nodeId;
}

export function isNumberMergeHead(
  nodeId: string,
  data: SourceFlowNodeData
): boolean {
  if (data.variant !== "number") return true;
  return !data.numberMergePrimaryId || data.numberMergePrimaryId === nodeId;
}

export function isNumberMergeTail(
  nodeId: string,
  data: SourceFlowNodeData
): boolean {
  if (data.variant !== "number") return true;
  return !data.numberMergeTailId || data.numberMergeTailId === nodeId;
}

function numberDataEqual(a: NumberSourceData, b: NumberSourceData): boolean {
  return (
    a.value === b.value &&
    a.digitValue === b.digitValue &&
    a.numberMergePrimaryId === b.numberMergePrimaryId &&
    a.numberMergeTailId === b.numberMergeTailId
  );
}

/**
 * Actualiza `value` (mostrar/enviar) y metadatos de grupo en nodos numéricos.
 * `digitValue` conserva el dígito de cada carta física.
 */
export function applyNumberTouchMerge(nodes: DataflowNode[]): DataflowNode[] {
  const groups = getNumberTouchGroups(nodes);
  const groupByMember = new Map<string, NumberTouchGroup>();
  for (const g of groups) {
    for (const id of g.memberIds) {
      groupByMember.set(id, g);
    }
  }

  let changed = false;
  const next = nodes.map((node) => {
    if (!isNumberSourceNode(node)) return node;

    const prev = node.data as NumberSourceData;
    const digitValue = readDigitValue(prev);
    const group = groupByMember.get(node.id);
    const mergedValue = group?.mergedValue ?? digitValue;
    const primaryId = group?.primaryId ?? node.id;
    const tailId = group?.tailId ?? node.id;
    const inGroup = group !== undefined && group.memberIds.length > 1;

    const updated: NumberSourceData = {
      ...prev,
      digitValue,
      value: mergedValue,
      numberMergePrimaryId: inGroup ? primaryId : undefined,
      numberMergeTailId: inGroup ? tailId : undefined,
    };

    if (!numberDataEqual(prev, updated)) {
      changed = true;
      return { ...node, data: updated };
    }
    return node;
  });

  return changed ? next : nodes;
}

/** Solo emite SourceStatement para el primario de cada grupo táctil. */
export function shouldEmitNumberSource(
  nodeId: string,
  nodes: DataflowNode[]
): boolean {
  return resolveNumberSourceId(nodeId, nodes) === nodeId;
}
