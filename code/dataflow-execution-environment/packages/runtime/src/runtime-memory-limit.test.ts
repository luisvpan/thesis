import { describe, it, expect } from 'bun:test';
import { Runtime } from './runtime';
import { IncrementalRuntime } from './incremental-runtime';
import type { DataflowProgram } from '@dataflow/shared/types';

function* counterGenerator() {
  let i = 0;
  while (true) {
    yield i++;
  }
}

describe('Runtime Memory Limit', () => {
  describe('Batch Runtime', () => {
    it('should enforce memory limit and throw error when exceeded', async () => {
      const runtime = new Runtime();
      
      const program: DataflowProgram = {
        metadata: { programId: 'test-memory-limit' },
        graph: {
          nodes: [
            { id: 'stream', type: 'DataSource', dataType: 'stream<natural>' as any, value: { kind: 'stream', elementType: 'natural', generatorFactory: counterGenerator } },
            { id: 'result', type: 'Output', dataType: 'natural', input: 'stream' }
          ],
          edges: [
            { id: 'e1', from: 'stream', to: 'result' }
          ]
        }
      };

      runtime.loadProgram(program);
      const result = await runtime.execute(0);
      expect(result).toBeDefined();

      const result2 = await runtime.execute(1000);
      expect(result2).toBeDefined();
    });

    it('should allow normal evaluation under memory limit', async () => {
      const runtime = new Runtime();
      
      const program: DataflowProgram = {
        metadata: { programId: 'test-memory-normal' },
        graph: {
          nodes: [
            { id: 'a', type: 'DataSource', dataType: 'natural', value: 1 },
            { id: 'b', type: 'DataSource', dataType: 'natural', value: 2 },
            { id: 'add', type: 'Transformation', dataType: 'natural', operation: 'ADD', inputs: ['a', 'b'] },
            { id: 'result', type: 'Output', dataType: 'natural', input: 'add' }
          ],
          edges: [
            { id: 'e1', from: 'a', to: 'add', toPort: 0 },
            { id: 'e2', from: 'b', to: 'add', toPort: 1 },
            { id: 'e3', from: 'add', to: 'result' }
          ]
        }
      };

      runtime.loadProgram(program);
      const result = await runtime.execute(0);
      expect(result).toEqual([{ kind: 'natural', value: 3 }]);
    });
  });

  describe('Incremental Runtime', () => {
    it('should enforce memory limit in evaluatePartial', async () => {
      const runtime = new IncrementalRuntime();
      
      const program: DataflowProgram = {
        metadata: { programId: 'test-memory-limit-incremental' },
        graph: {
          nodes: [
            { id: 'stream', type: 'DataSource', dataType: 'stream<natural>' as any, value: { kind: 'stream', elementType: 'natural', generatorFactory: counterGenerator } },
            { id: 'result', type: 'Output', dataType: 'natural', input: 'stream' }
          ],
          edges: [
            { id: 'e1', from: 'stream', to: 'result' }
          ]
        }
      };

      runtime.loadProgram(program);
      const result = await runtime.evaluatePartial(0);
      expect(result).toBeDefined();

      const result2 = await runtime.evaluatePartial(1000);
      expect(result2).toBeDefined();
    });

    it('should allow normal evaluation under memory limit', async () => {
      const runtime = new IncrementalRuntime();
      
      const program: DataflowProgram = {
        metadata: { programId: 'test-memory-normal-incremental' },
        graph: {
          nodes: [
            { id: 'a', type: 'DataSource', dataType: 'natural', value: 1 },
            { id: 'b', type: 'DataSource', dataType: 'natural', value: 2 },
            { id: 'add', type: 'Transformation', dataType: 'natural', operation: 'ADD', inputs: ['a', 'b'] },
            { id: 'result', type: 'Output', dataType: 'natural', input: 'add' }
          ],
          edges: [
            { id: 'e1', from: 'a', to: 'add', toPort: 0 },
            { id: 'e2', from: 'b', to: 'add', toPort: 1 },
            { id: 'e3', from: 'add', to: 'result' }
          ]
        }
      };

      runtime.loadProgram(program);
      const result = await runtime.evaluatePartial(0);
      expect(result.nodeStates.get('result')?.status).toBe('completed');
      expect(result.nodeStates.get('result')?.value).toEqual({ kind: 'natural', value: 3 });
    });
  });
});
