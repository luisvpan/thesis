import { describe, it, expect, beforeEach } from 'bun:test';
import { IncrementalRuntime } from '@dataflow/runtime';
import type { DataflowProgram } from '@dataflow/shared/types';

function isCompletedState(state: any): state is { status: "completed"; value: any } {
  return state && state.status === "completed";
}

describe('IncrementalRuntime - Notifications', () => {
  let runtime: IncrementalRuntime;
  let notificationCallbackCalls: any[] = [];

  beforeEach(() => {
    runtime = new IncrementalRuntime();
    notificationCallbackCalls = [];
  });

  it('should push node_state_changed events', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-node-state-changed' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 5 },
          { id: 'result', type: 'Output', dataType: 'natural', input: 'a' }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'result' }
        ]
      }
    };

    runtime.loadProgram(program);
    
    runtime.subscribe('a', (state) => {
      notificationCallbackCalls.push({ nodeId: 'a', state });
    });

    runtime.evaluatePartial(0);

    expect(notificationCallbackCalls).toHaveLength(1);
    expect(notificationCallbackCalls[0].nodeId).toBe('a');
    expect(notificationCallbackCalls[0].state.status).toBe('completed');
  });

  it('should include correct node data in notifications', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-correct-node-data' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 10 },
          { id: 'b', type: 'DataSource', dataType: 'natural', value: 5 },
          { id: 'add', type: 'Transformation', dataType: 'natural', operation: 'ADD', inputs: ['a', 'b'] }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'add', toPort: 0 },
          { id: 'e2', from: 'b', to: 'add', toPort: 1 }
        ]
      }
    };

    runtime.loadProgram(program);
    
    runtime.subscribe('add', (state) => {
      notificationCallbackCalls.push(state);
    });

    runtime.evaluatePartial(0);

    expect(notificationCallbackCalls).toHaveLength(1);
    if (isCompletedState(notificationCallbackCalls[0])) {
      expect(notificationCallbackCalls[0].value).toEqual({ kind: 'natural', value: 15 });
    }
  });

  it('should handle multiple changes in single evaluation', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-multiple-changes' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 5 },
          { id: 'b', type: 'DataSource', dataType: 'natural', value: 3 },
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
    
    let aCallCount = 0;
    let addCallCount = 0;
    let resultCallCount = 0;
    
    runtime.subscribe('a', (state) => { aCallCount++; });
    runtime.subscribe('add', (state) => { addCallCount++; });
    runtime.subscribe('result', (state) => { resultCallCount++; });

    runtime.evaluatePartial(0);

    // All subscribed nodes should receive notifications in a single evaluation
    expect(aCallCount).toBe(1);
    expect(addCallCount).toBe(1);
    expect(resultCallCount).toBe(1);
  });

  it('should send notifications on node updates', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-update-notifications' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 5 },
          { id: 'result', type: 'Output', dataType: 'natural', input: 'a' }
        ],
        edges: [
          { id: 'e1', from: 'a', to: 'result' }
        ]
      }
    };

    runtime.loadProgram(program);
    
    runtime.subscribe('a', (state) => {
      notificationCallbackCalls.push(state);
    });

    // First evaluation
    runtime.evaluatePartial(0);
    expect(notificationCallbackCalls).toHaveLength(1);
    if (isCompletedState(notificationCallbackCalls[0])) {
      expect(notificationCallbackCalls[0].value).toEqual({ kind: 'natural', value: 5 });
    }

    // Update node and re-evaluate
    runtime.updateGraph({
      addedNodes: [
        { id: 'a', type: 'DataSource', dataType: 'natural', value: 15 }
      ]
    });

    notificationCallbackCalls = [];
    runtime.evaluatePartial(0);
    
    // Should receive notification with updated value
    expect(notificationCallbackCalls).toHaveLength(1);
    if (isCompletedState(notificationCallbackCalls[0])) {
      expect(notificationCallbackCalls[0].value).toEqual({ kind: 'natural', value: 15 });
    }
  });

  it('should notify all subscribers of changed nodes', () => {
    const program: DataflowProgram = {
      metadata: { programId: 'test-all-subscribers' },
      graph: {
        nodes: [
          { id: 'a', type: 'DataSource', dataType: 'natural', value: 5 }
        ],
        edges: []
      }
    };

    runtime.loadProgram(program);
    
    const callback1Calls: any[] = [];
    const callback2Calls: any[] = [];
    const callback3Calls: any[] = [];
    
    runtime.subscribe('a', (state) => { callback1Calls.push(state); });
    runtime.subscribe('a', (state) => { callback2Calls.push(state); });
    runtime.subscribe('a', (state) => { callback3Calls.push(state); });

    runtime.evaluatePartial(0);

    // All subscribers should receive notifications
    expect(callback1Calls).toHaveLength(1);
    expect(callback2Calls).toHaveLength(1);
    expect(callback3Calls).toHaveLength(1);
    
    // All should receive the same state
    expect(callback1Calls[0]).toEqual(callback2Calls[0]);
    expect(callback2Calls[0]).toEqual(callback3Calls[0]);
  });
});
