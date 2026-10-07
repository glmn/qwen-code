/**
 * @license
 * Copyright 2025 Qwen Team
 * SPDX-License-Identifier: Apache-2.0
 */

import { EventEmitter } from 'node:events';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Config, MCPServerConfig } from '../config/config.js';
import type { PromptRegistry } from '../prompts/prompt-registry.js';
import type { ResourceRegistry } from '../resources/resource-registry.js';
import type { WorkspaceContext } from '../utils/workspaceContext.js';
import { McpClientManager } from './mcp-client-manager.js';
import type { PoolEvent } from './mcp-pool-events.js';
import type { DiscoveredMCPTool } from './mcp-tool.js';
import { McpTransportPool } from './mcp-transport-pool.js';
import type { ToolRegistry } from './tool-registry.js';

// A stdio MCP server that lists `first`. Once the trigger file exists it also
// lists `second` and sends `notifications/tools/list_changed`.
const SERVER = `
  import { existsSync } from 'node:fs';
  import readline from 'node:readline';
  const trigger = process.argv[1];
  const tool = (name) => ({
    name,
    description: name,
    inputSchema: { type: 'object', properties: {} },
  });
  let tools = [tool('first')];
  const send = (message) =>
    process.stdout.write(JSON.stringify({ jsonrpc: '2.0', ...message }) + '\\n');
  const timer = setInterval(() => {
    if (!existsSync(trigger)) return;
    clearInterval(timer);
    tools = [...tools, tool('second')];
    send({ method: 'notifications/tools/list_changed' });
  }, 10);
  const lines = readline.createInterface({ input: process.stdin });
  lines.on('close', () => process.exit(0));
  lines.on('line', (line) => {
    const request = JSON.parse(line);
    if (request.id === undefined) return;
    if (request.method === 'initialize') {
      send({
        id: request.id,
        result: {
          protocolVersion: request.params.protocolVersion,
          capabilities: { tools: { listChanged: true } },
          serverInfo: { name: 'list-changed', version: '1.0.0' },
        },
      });
    } else if (request.method === 'tools/list') {
      send({ id: request.id, result: { tools } });
    } else {
      send({ id: request.id, error: { code: -32601, message: 'Method not found' } });
    }
  });
`;

/** Tool registry holding real registrations, keyed by tool name. */
function mkToolRegistry() {
  const tools = new Map<string, DiscoveredMCPTool>();
  const registry = {
    registerTool: (tool: DiscoveredMCPTool) => tools.set(tool.name, tool),
    removeMcpToolsByServer: (serverName: string) => {
      for (const [name, tool] of tools) {
        if (tool.serverName === serverName) tools.delete(name);
      }
    },
    refreshMcpToolsByServer: (_serverName: string, reregister: () => void) =>
      reregister(),
    getToolsByServer: (serverName: string) =>
      [...tools.values()].filter((tool) => tool.serverName === serverName),
  };
  const names = () =>
    [...tools.values()].map((tool) => tool.serverToolName).sort();
  return { names, registry: registry as unknown as ToolRegistry };
}

const promptRegistry = {
  registerPrompt: vi.fn(),
  removePromptsByServer: vi.fn(),
} as unknown as PromptRegistry;
const resourceRegistry = {
  registerResource: vi.fn(),
  removeResourcesByServer: vi.fn(),
} as unknown as ResourceRegistry;
const workspaceContext = {
  getDirectories: () => [],
  onDirectoriesChanged: () => () => {},
} as unknown as WorkspaceContext;

async function until(check: () => boolean): Promise<void> {
  await vi.waitFor(() => expect(check()).toBe(true), {
    timeout: 10_000,
    interval: 20,
  });
}

describe('notifications/tools/list_changed', () => {
  let dir: string;
  let trigger: string;
  let serverConfig: MCPServerConfig;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'qwen-mcp-list-changed-'));
    trigger = join(dir, 'add-second-tool');
    serverConfig = {
      command: process.execPath,
      args: ['--input-type=module', '--eval', SERVER, trigger],
    } as MCPServerConfig;
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  it('re-registers the tools of a standalone (non-pooled) server', async () => {
    const { names, registry } = mkToolRegistry();
    const config = {
      getMcpServers: () => ({ srv: serverConfig }),
      getMcpServerCommand: () => undefined,
      getTargetDir: () => dir,
      getResourceRegistry: () => resourceRegistry,
      getPromptRegistry: () => promptRegistry,
      getWorkspaceContext: () => workspaceContext,
      getDebugMode: () => false,
    } as unknown as Config;
    const eventEmitter = new EventEmitter();
    const updates = vi.fn();
    eventEmitter.on('mcp-client-update', updates);
    const manager = new McpClientManager(config, registry, {
      eventEmitter,
      budgetConfig: { budgetMode: 'off' },
    });
    try {
      await manager.discoverMcpToolsForServer('srv', config);
      expect(names()).toEqual(['first']);
      updates.mockClear();

      writeFileSync(trigger, '');

      await until(() => names().includes('second'));
      expect(names()).toEqual(['first', 'second']);
      expect(updates).toHaveBeenCalled();
    } finally {
      await manager.stop();
    }
  }, 20_000);

  it('refreshes a pooled server and emits toolsChanged to every session', async () => {
    const pool = new McpTransportPool({} as Config, {
      workspaceContext,
      debugMode: false,
      drainDelayMs: 1_000,
    });
    const a = mkToolRegistry();
    const b = mkToolRegistry();
    try {
      const connA = await pool.acquire(
        'srv',
        serverConfig,
        'session-a',
        a.registry,
        promptRegistry,
        resourceRegistry,
      );
      await pool.acquire(
        'srv',
        serverConfig,
        'session-b',
        b.registry,
        promptRegistry,
        resourceRegistry,
      );
      const events: PoolEvent[] = [];
      connA.on('event', (event) => events.push(event));
      expect(a.names()).toEqual(['first']);

      writeFileSync(trigger, '');

      await until(() => events.some((e) => e.kind === 'toolsChanged'));
      const changed = events.find((e) => e.kind === 'toolsChanged');
      expect(
        changed?.kind === 'toolsChanged' &&
          changed.snapshot.map((tool) => tool.serverToolName),
      ).toEqual(['first', 'second']);
      expect(connA.toolsSnapshot.map((tool) => tool.serverToolName)).toEqual([
        'first',
        'second',
      ]);
      for (const session of [a, b]) {
        expect(session.names()).toEqual(['first', 'second']);
      }
    } finally {
      await pool.drainAll();
    }
  }, 20_000);
});
