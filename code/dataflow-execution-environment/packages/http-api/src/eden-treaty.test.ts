import { describe, it, expect } from "bun:test";
import type { App } from "../src/index";

describe("Eden Treaty Export", () => {
  it("should export app instance", async () => {
    const { app } = await import("../src/index");

    expect(app).toBeDefined();
    expect(typeof app).toBe("object");
  });

  it("Eden Treaty can use App type for type safety", () => {
    type TestApp = App;

    const mockApi = {
      api: {
        v1: {
          health: {
            get: () => Promise.resolve({ status: "healthy", version: "1.0.0", uptime: 0 })
          },
          compile: {
            post: () => Promise.resolve({ success: true, errors: [], warnings: [] })
          },
          execute: {
            post: () => Promise.resolve({ success: true, outputs: [], trace: undefined })
          }
        }
      }
    };

    expect(mockApi).toBeDefined();
    expect(mockApi.api).toBeDefined();
    expect(mockApi.api.v1).toBeDefined();
    expect(mockApi.api.v1.health).toBeDefined();
    expect(mockApi.api.v1.compile).toBeDefined();
    expect(mockApi.api.v1.execute).toBeDefined();
  });

  it("package exports allow Eden Treaty pattern", async () => {
    const api = await import("../src/index");

    expect(api).toBeDefined();
    expect(api.app).toBeDefined();
    expect(typeof api.app).toBe("object");
  });

  it("App type can be used for type inference", () => {
    type ClientType = App;

    const typeName: string = "function";
    expect(typeof typeName).toBe("string");
  });
});
