import { describe, it, expect, beforeEach } from 'bun:test';
import { IncrementalRuntime } from '@dataflow/runtime';
import type { DataflowProgram } from '@dataflow/shared/types';

describe('IncrementalRuntime - Subscriptions', () => {
  let runtime: IncrementalRuntime;
  let subscriptionCallbackCalls: any[] = [];

  beforeEach(() => {
    runtime = new IncrementalRuntime();
    subscriptionCallbackCalls = [];
  });

  it('should register subscription to a node', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-subscription' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 5 }
        ],
        edges: []
      }
    };

    runtime.loadProgram(program);
    runtime.subscribe('a', (state) => {
      subscriptionCallbackCalls.push(state);
    });

    runtime.evaluatePartial(0);

    expect(subscriptionCallbackCalls).toHaveLength(1);
    expect(subscriptionCallbackCalls[0]).toEqual({
      status: 'completed',
      value: { kind: 'natural', value: 5 }
    });
  });

  it('should unregister subscription from a node', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-unsubscription' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 5 }
        ],
        edges: []
      }
    };

    runtime.loadProgram(program);
    
    const callback = (state: any) => {
      subscriptionCallbackCalls.push(state);
    };
    
    runtime.subscribe('a', callback);
    runtime.unsubscribe('a', callback);
    runtime.evaluatePartial(0);

    expect(subscriptionCallbackCalls).toHaveLength(0);
  });

  it('should notify subscribers on value changes', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-notification' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 5 },
          { id: 'c', type: 'DataSource', dataType: 'natural', value: 3 },
          { id: 'b', type: 'Transformation', dataType: 'natural', operation: 'ADD', inputs: ['a', 'c'] }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'b', toPort: 0 },
          { id: 'e2', from: 'c', to: 'b', toPort: 1 }
        ]
      }
    };

    runtime.loadProgram(program);
    
    let callbackCallCount = 0;
    runtime.subscribe('b', (state) => {
      callbackCallCount++;
      subscriptionCallbackCalls.push(state);
    });

    runtime.evaluatePartial(0);

    expect(callbackCallCount).toBe(1);
    expect(subscriptionCallbackCalls[0]).toEqual({
      status: 'completed',
      value: { kind: 'natural', value: 8 }
    });
  });

  it('should handle multiple subscribers to same node', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-multiple-subscribers' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 5 }
        ],
        edges: []
      }
    };

    runtime.loadProgram(program);
    
    const callback1 = (state: any) => subscriptionCallbackCalls.push({ callback: 1, state });
    const callback2 = (state: any) => subscriptionCallbackCalls.push({ callback: 2, state });
    const callback3 = (state: any) => subscriptionCallbackCalls.push({ callback: 3, state });
    
    runtime.subscribe('a', callback1);
    runtime.subscribe('a', callback2);
    runtime.subscribe('a', callback3);
    
    runtime.evaluatePartial(0);

    expect(subscriptionCallbackCalls).toHaveLength(3);
    expect(subscriptionCallbackCalls[0].callback).toBe(1);
    expect(subscriptionCallbackCalls[1].callback).toBe(2);
    expect(subscriptionCallbackCalls[2].callback).toBe(3);
    expect(subscriptionCallbackCalls[0].state).toEqual({
      status: 'completed',
      value: { kind: 'natural', value: 5 }
    });
    expect(subscriptionCallbackCalls[1].state).toEqual({
      status: 'completed',
      value: { kind: 'natural', value: 5 }
    });
    expect(subscriptionCallbackCalls[2].state).toEqual({
      status: 'completed',
      value: { kind: 'natural', value: 5 }
    });
  });

  it('should clean up when no subscribers remain', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-cleanup' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 5 }
        ],
        edges: []
      }
    };

    runtime.loadProgram(program);
    
    const callback = (state: any) => {
      subscriptionCallbackCalls.push(state);
    };
    
    runtime.subscribe('a', callback);
    runtime.evaluatePartial(0);
    
    expect(subscriptionCallbackCalls).toHaveLength(1);
    
    runtime.unsubscribe('a', callback);
    runtime.evaluatePartial(1);
    
    expect(subscriptionCallbackCalls).toHaveLength(1);
  });
});
