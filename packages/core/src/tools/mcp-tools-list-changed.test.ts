/**
 * @license
 * Copyright 2025 Qwen Team
 * SPDX-License-Identifier: Apache-2.0
 */

import { EventEmitter } from 'node:events';
import {
  existsSync,
  mkdtempSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Config, MCPServerConfig } from '../config/config.js';
import type { PromptRegistry } from '../prompts/prompt-registry.js';
import type { ResourceRegistry } from '../resources/resource-registry.js';
import type { WorkspaceContext } from '../utils/workspaceContext.js';
import { McpClientManager } from './mcp-client-manager.js';
import { TOOLS_REFRESH_RETRY_MS } from './mcp-client.js';
import type { PoolEvent } from './mcp-pool-events.js';
import type { DiscoveredMCPTool } from './mcp-tool.js';
import { McpTransportPool } from './mcp-transport-pool.js';
import type { ToolRegistry } from './tool-registry.js';

// A stdio MCP server driven by a control file. It starts listing `first` and
// `gone`. Each time the control file's `seq` grows it takes the new state and
// sends `notify` (default 1) `notifications/tools/list_changed`:
//   tools      – what tools/list returns from now on
//   failList   – tools/list answers with an error while true
//   responses  – per-request overrides for the next tools/list requests,
//                in order: { tools?, delayMs? }
// tools/call answers `called <name>` for a listed tool and an error for any
// other. Every tools/list request is appended to the log file.
const SERVER = `
  import { appendFileSync, readFileSync } from 'node:fs';
  import readline from 'node:readline';
  const [control, log] = process.argv.slice(1);
  const tool = (name) => ({
    name,
    description: name,
    inputSchema: { type: 'object', properties: {} },
  });
  let seq = 0;
  let tools = ['first', 'gone'];
  let failList = false;
  let responses = [];
  const send = (message) =>
    process.stdout.write(JSON.stringify({ jsonrpc: '2.0', ...message }) + '\\n');
  const timer = setInterval(() => {
    let next;
    try {
      next = JSON.parse(readFileSync(control, 'utf8'));
    } catch {
      return;
    }
    if (next.seq <= seq) return;
    seq = next.seq;
    tools = next.tools ?? tools;
    failList = next.failList ?? false;
    responses = next.responses ?? [];
    for (let i = 0; i < (next.notify ?? 1); i++) {
      send({ method: 'notifications/tools/list_changed' });
    }
  }, 10);
  const lines = readline.createInterface({ input: process.stdin });
  lines.on('close', () => {
    clearInterval(timer);
    process.exit(0);
  });
  lines.on('line', (line) => {
    const request = JSON.parse(line);
    if (request.id === undefined) return;
    const reply = (body) => send({ id: request.id, ...body });
    if (request.method === 'initialize') {
      reply({
        result: {
          protocolVersion: request.params.protocolVersion,
          capabilities: { tools: { listChanged: true } },
          serverInfo: { name: 'list-changed', version: '1.0.0' },
        },
      });
    } else if (request.method === 'tools/list') {
      appendFileSync(log, 'list\\n');
      if (failList) {
        reply({ error: { code: -32603, message: 'catalog unavailable' } });
        return;
      }
      const override = responses.shift() ?? {};
      const result = { tools: (override.tools ?? tools).map(tool) };
      if (override.delayMs) setTimeout(() => reply({ result }), override.delayMs);
      else reply({ result });
    } else if (request.method === 'tools/call') {
      const name = request.params.name;
      if (tools.includes(name)) {
        reply({ result: { content: [{ type: 'text', text: 'called ' + name }] } });
      } else {
        reply({ error: { code: -32602, message: 'Unknown tool: ' + name } });
      }
    } else {
      reply({ error: { code: -32601, message: 'Method not found' } });
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
  /** Calls a registered tool the way the scheduler does. */
  const call = async (serverToolName: string): Promise<string> => {
    const tool = [...tools.values()].find(
      (t) => t.serverToolName === serverToolName,
    );
    if (!tool) throw new Error(`'${serverToolName}' is not registered`);
    const result = await tool.build({}).execute(new AbortController().signal);
    return JSON.stringify(result.llmContent);
  };
  return { names, call, registry: registry as unknown as ToolRegistry };
}
type Session = ReturnType<typeof mkToolRegistry>;

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

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** The server's tools as seen through one path. */
interface Harness {
  /** One registry per session (standalone: one; pooled: two). */
  sessions: Session[];
  /** Pooled only: the pool's events, as a subscriber sees them. */
  events: PoolEvent[];
  stop(): Promise<void>;
}

const STALE = /reported that its tool list changed/;

describe.each(['standalone', 'pooled'] as const)(
  'notifications/tools/list_changed (%s)',
  (mode) => {
    let dir: string;
    let control: string;
    let log: string;
    let seq = 0;
    let harness: Harness;

    /** Changes the server's state and makes it send list_changed. */
    function serverChanges(next: Record<string, unknown>): void {
      seq += 1;
      const tmp = `${control}.tmp`;
      writeFileSync(tmp, JSON.stringify({ seq, ...next }));
      renameSync(tmp, control);
    }
    const listRequests = () =>
      existsSync(log)
        ? readFileSync(log, 'utf8').split('\n').filter(Boolean).length
        : 0;
    const allNames = () => harness.sessions.map((s) => s.names());
    const everywhere = (names: string[]) =>
      allNames().every((n) => JSON.stringify(n) === JSON.stringify(names));

    async function start(): Promise<Harness> {
      const serverConfig = {
        command: process.execPath,
        args: ['--input-type=module', '--eval', SERVER, control, log],
      } as MCPServerConfig;
      if (mode === 'standalone') {
        const session = mkToolRegistry();
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
        const manager = new McpClientManager(config, session.registry, {
          eventEmitter,
          budgetConfig: { budgetMode: 'off' },
        });
        await manager.discoverMcpToolsForServer('srv', config);
        return {
          sessions: [session],
          events: [],
          stop: () => manager.stop(),
        };
      }
      const pool = new McpTransportPool({} as Config, {
        workspaceContext,
        debugMode: false,
        drainDelayMs: 1_000,
      });
      const sessions = [mkToolRegistry(), mkToolRegistry()];
      const events: PoolEvent[] = [];
      for (const [i, session] of sessions.entries()) {
        const conn = await pool.acquire(
          'srv',
          serverConfig,
          `session-${i}`,
          session.registry,
          promptRegistry,
          resourceRegistry,
        );
        if (i === 0) conn.on('event', (event) => events.push(event));
      }
      return {
        sessions,
        events,
        stop: async () => {
          await pool.drainAll();
        },
      };
    }

    beforeEach(async () => {
      dir = mkdtempSync(join(tmpdir(), 'qwen-mcp-list-changed-'));
      control = join(dir, 'control.json');
      log = join(dir, 'requests.log');
      seq = 0;
      harness = await start();
      expect(everywhere(['first', 'gone'])).toBe(true);
    });

    afterEach(async () => {
      await harness.stop();
      rmSync(dir, { recursive: true, force: true });
    });

    it('positive: a successful refresh keeps listed tools callable and adds new ones', async () => {
      serverChanges({ tools: ['first', 'gone', 'added'] });

      await until(() => everywhere(['added', 'first', 'gone']));
      for (const session of harness.sessions) {
        expect(await session.call('first')).toContain('called first');
        expect(await session.call('added')).toContain('called added');
      }
      if (mode === 'pooled') {
        const changed = harness.events.find((e) => e.kind === 'toolsChanged');
        expect(
          changed?.kind === 'toolsChanged' &&
            changed.snapshot.map((tool) => tool.serverToolName),
        ).toEqual(['first', 'gone', 'added']);
      }
    }, 20_000);

    it('removal: a successful refresh without a tool removes it from every session', async () => {
      serverChanges({ tools: ['first'] });

      await until(() => everywhere(['first']));
      for (const session of harness.sessions) {
        await expect(session.call('gone')).rejects.toThrow(/is not registered/);
        expect(await session.call('first')).toContain('called first');
      }
    }, 20_000);

    it('negative: while the refresh fails, the old tools are refused, then work again after a successful refresh', async () => {
      const before = listRequests();
      serverChanges({ failList: true });

      await until(() => listRequests() > before);
      // Still registered from the old listing, but refused rather than run.
      for (const session of harness.sessions) {
        expect(session.names()).toEqual(['first', 'gone']);
        await expect(session.call('first')).rejects.toThrow(STALE);
      }
      // One retry follows; with the list still failing, the gate holds.
      const afterFirstAttempt = listRequests();
      await until(() => listRequests() > afterFirstAttempt);
      await sleep(TOOLS_REFRESH_RETRY_MS + 500);
      const afterRetry = listRequests();
      await sleep(TOOLS_REFRESH_RETRY_MS + 500);
      expect(listRequests()).toBe(afterRetry);
      for (const session of harness.sessions) {
        await expect(session.call('first')).rejects.toThrow(STALE);
      }

      serverChanges({ failList: false, tools: ['first'] });

      await until(() => everywhere(['first']));
      for (const session of harness.sessions) {
        expect(await session.call('first')).toContain('called first');
      }
    }, 30_000);

    it('ordering: an older response cannot restore a tool a newer revision removed', async () => {
      const before = listRequests();
      // Two notifications; the first re-list is answered late with the old
      // list (still holding `gone`), the second at once with the new one.
      serverChanges({
        tools: ['first'],
        notify: 2,
        responses: [{ tools: ['first', 'gone'], delayMs: 600 }],
      });

      await until(() => listRequests() >= before + 2);
      await until(() => everywhere(['first']));
      await sleep(1_000); // the late response has arrived and been dropped
      expect(everywhere(['first'])).toBe(true);
      for (const session of harness.sessions) {
        expect(await session.call('first')).toContain('called first');
      }
    }, 20_000);
  },
);
