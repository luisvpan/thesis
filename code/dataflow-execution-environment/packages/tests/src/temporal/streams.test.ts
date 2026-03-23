import { describe, it, expect } from "bun:test";
import { Compiler } from "@dataflow/compiler";
import { Runtime } from "@dataflow/runtime";

describe("Integration Tests - Temporal", () => {
  describe("Test 5.2: FIRST from Stream", () => {
    it("should extract first value from stream", () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source numbers: stream<natural> = stream<natural>(generator(counter));

        transform first_number: natural = FIRST(numbers);

        output result: natural = first_number;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);
      const outputs = runtime.execute();

      expect(outputs).toHaveLength(1);
      expect(outputs[0]).toEqual({ kind: "natural", value: 0 });
    });

    it("should always return the same first value", () => {
      const compiler = new Compiler();
      const runtime = new Runtime();

      const source = `
        source numbers: stream<natural> = stream<natural>(generator(counter));

        transform first_number: natural = FIRST(numbers);

        output result: natural = first_number;
      `;

      const compileResult = compiler.compile(source);
      expect(compileResult.success).toBe(true);

      runtime.loadProgram(compileResult.program!);

      const output1 = runtime.execute(0);
      const output2 = runtime.execute(1);
      const output3 = runtime.execute(5);

      expect(output1[0]).toEqual({ kind: "natural", value: 0 });
      expect(output2[0]).toEqual({ kind: "natural", value: 0 });
      expect(output3[0]).toEqual({ kind: "natural", value: 0 });
    });
  });
});
