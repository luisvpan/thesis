import { describe, it, expect } from "bun:test";
import { LRUCache } from "../utils/lru-cache";

describe("LRUCache", () => {
  it("should get and set values", () => {
    const cache = new LRUCache<string, number>(3);
    cache.set("a", 1);
    cache.set("b", 2);
    cache.set("c", 3);

    expect(cache.get("a")).toBe(1);
    expect(cache.get("b")).toBe(2);
    expect(cache.get("c")).toBe(3);
  });

  it("should evict least recently used entry when at capacity", () => {
    const cache = new LRUCache<string, number>(3);
    cache.set("a", 1);
    cache.set("b", 2);
    cache.set("c", 3);
    cache.set("d", 4);

    expect(cache.get("a")).toBeUndefined();
    expect(cache.get("b")).toBe(2);
    expect(cache.get("c")).toBe(3);
    expect(cache.get("d")).toBe(4);
  });

  it("should move accessed entry to most recently used", () => {
    const cache = new LRUCache<string, number>(3);
    cache.set("a", 1);
    cache.set("b", 2);
    cache.set("c", 3);
    cache.get("a");
    cache.set("d", 4);

    expect(cache.get("a")).toBe(1);
    expect(cache.get("b")).toBeUndefined();
    expect(cache.get("c")).toBe(3);
    expect(cache.get("d")).toBe(4);
  });

  it("should delete entries", () => {
    const cache = new LRUCache<string, number>(3);
    cache.set("a", 1);
    cache.set("b", 2);
    cache.delete("a");

    expect(cache.get("a")).toBeUndefined();
    expect(cache.get("b")).toBe(2);
    expect(cache.size).toBe(1);
  });

  it("should clear all entries", () => {
    const cache = new LRUCache<string, number>(3);
    cache.set("a", 1);
    cache.set("b", 2);
    cache.set("c", 3);
    cache.clear();

    expect(cache.size).toBe(0);
    expect(cache.get("a")).toBeUndefined();
    expect(cache.get("b")).toBeUndefined();
    expect(cache.get("c")).toBeUndefined();
  });

  it("should handle complex values", () => {
    const cache = new LRUCache<string, Map<number, unknown>>(3);
    const map1 = new Map([[1, "value1"]]);
    const map2 = new Map([[2, "value2"]]);

    cache.set("node1", map1);
    cache.set("node2", map2);

    const retrieved1 = cache.get("node1");
    const retrieved2 = cache.get("node2");

    expect(retrieved1).toBe(map1);
    expect(retrieved2).toBe(map2);
    expect(retrieved1?.get(1)).toBe("value1");
    expect(retrieved2?.get(2)).toBe("value2");
  });

  it("should return correct size", () => {
    const cache = new LRUCache<string, number>(5);
    expect(cache.size).toBe(0);

    cache.set("a", 1);
    cache.set("b", 2);
    cache.set("c", 3);
    expect(cache.size).toBe(3);

    cache.delete("a");
    expect(cache.size).toBe(2);
  });

  it("should update existing entry", () => {
    const cache = new LRUCache<string, number>(3);
    cache.set("a", 1);
    cache.set("a", 10);

    expect(cache.get("a")).toBe(10);
    expect(cache.size).toBe(1);
  });

  it("should handle has check", () => {
    const cache = new LRUCache<string, number>(3);
    cache.set("a", 1);

    expect(cache.has("a")).toBe(true);
    expect(cache.has("b")).toBe(false);
  });
});
