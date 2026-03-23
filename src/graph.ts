/**
 * Network / graph statistics.
 *
 * - **Graph construction** — adjacency list representation, from edge list or adjacency matrix.
 * - **Centrality measures** — degree, betweenness, closeness, eigenvector, PageRank.
 * - **Community detection** — label propagation, modularity computation.
 * - **Path algorithms** — BFS shortest path, connected components.
 * - **Graph properties** — density, clustering coefficient, diameter.
 */

// ── Graph Data Structure ──────────────────────────────────────────────────

export class Graph {
  /** Adjacency list: node → [neighbour, ...]. */
  private _adj: Map<number, Set<number>>;
  /** Edge weights. Key = "i,j" (ordered). */
  private _weights: Map<string, number>;
  /** Whether the graph is directed. */
  readonly directed: boolean;

  constructor(directed = false) {
    this._adj = new Map();
    this._weights = new Map();
    this.directed = directed;
  }

  // ── Factories ───────────────────────────────────────────────────────────

  /**
   * Create from an edge list: [[source, target, weight?], ...].
   */
  static fromEdgeList(
    edges: [number, number, number?][],
    directed = false,
  ): Graph {
    const g = new Graph(directed);
    for (const [u, v, w] of edges) {
      g.addEdge(u, v, w ?? 1);
    }
    return g;
  }

  /**
   * Create from an adjacency matrix (0 = no edge).
   * Diagonal is ignored.
   */
  static fromAdjacencyMatrix(matrix: number[][], directed = false): Graph {
    const n = matrix.length;
    const g = new Graph(directed);
    for (let i = 0; i < n; i++) g.addNode(i);
    for (let i = 0; i < n; i++) {
      const jStart = directed ? 0 : i + 1;
      for (let j = jStart; j < n; j++) {
        if (i !== j && matrix[i][j] !== 0) {
          g.addEdge(i, j, matrix[i][j]);
        }
      }
    }
    return g;
  }

  // ── Mutation ────────────────────────────────────────────────────────────

  addNode(node: number): void {
    if (!this._adj.has(node)) this._adj.set(node, new Set());
  }

  addEdge(u: number, v: number, weight = 1): void {
    this.addNode(u);
    this.addNode(v);
    this._adj.get(u)!.add(v);
    this._weights.set(edgeKey(u, v), weight);
    if (!this.directed) {
      this._adj.get(v)!.add(u);
      this._weights.set(edgeKey(v, u), weight);
    }
  }

  // ── Properties ──────────────────────────────────────────────────────────

  get nodes(): number[] {
    return [...this._adj.keys()].sort((a, b) => a - b);
  }

  get nodeCount(): number {
    return this._adj.size;
  }

  get edgeCount(): number {
    let count = 0;
    for (const [, neighbors] of this._adj) count += neighbors.size;
    return this.directed ? count : count / 2;
  }

  neighbours(node: number): number[] {
    return [...(this._adj.get(node) ?? [])];
  }

  degree(node: number): number {
    return this._adj.get(node)?.size ?? 0;
  }

  weight(u: number, v: number): number {
    return this._weights.get(edgeKey(u, v)) ?? 0;
  }

  hasEdge(u: number, v: number): boolean {
    return this._adj.get(u)?.has(v) ?? false;
  }

  // ── Graph metrics ───────────────────────────────────────────────────────

  /**
   * Graph density: |E| / (|V|(|V|−1)/k), k = 1 for directed, 2 for undirected.
   */
  density(): number {
    const n = this.nodeCount;
    if (n < 2) return 0;
    const maxEdges = this.directed ? n * (n - 1) : (n * (n - 1)) / 2;
    return this.edgeCount / maxEdges;
  }

  /**
   * Global clustering coefficient (transitivity):
   * 3 × triangles / connected triples.
   */
  clusteringCoefficient(): number {
    const nodeList = this.nodes;
    let triangles = 0;
    let triples = 0;

    for (const v of nodeList) {
      const nbrs = this.neighbours(v);
      const k = nbrs.length;
      triples += (k * (k - 1)) / 2;
      for (let i = 0; i < nbrs.length; i++) {
        for (let j = i + 1; j < nbrs.length; j++) {
          if (this.hasEdge(nbrs[i], nbrs[j])) triangles++;
        }
      }
    }
    return triples > 0 ? triangles / triples : 0;
  }

  // ── Paths ───────────────────────────────────────────────────────────────

  /**
   * BFS shortest path (unweighted) from source.
   * Returns distances (−1 = unreachable).
   */
  bfsDistances(source: number): Map<number, number> {
    const dist = new Map<number, number>();
    for (const n of this.nodes) dist.set(n, -1);
    dist.set(source, 0);
    const queue = [source];
    let head = 0;

    while (head < queue.length) {
      const u = queue[head++];
      const d = dist.get(u)!;
      for (const v of this.neighbours(u)) {
        if (dist.get(v) === -1) {
          dist.set(v, d + 1);
          queue.push(v);
        }
      }
    }
    return dist;
  }

  /**
   * Connected components (undirected).
   * Returns array of component arrays.
   */
  connectedComponents(): number[][] {
    const visited = new Set<number>();
    const components: number[][] = [];

    for (const node of this.nodes) {
      if (visited.has(node)) continue;
      const component: number[] = [];
      const queue = [node];
      visited.add(node);

      while (queue.length > 0) {
        const u = queue.shift()!;
        component.push(u);
        for (const v of this.neighbours(u)) {
          if (!visited.has(v)) {
            visited.add(v);
            queue.push(v);
          }
        }
      }
      components.push(component);
    }
    return components;
  }

  /**
   * Graph diameter (longest shortest path).
   * Returns Infinity if the graph is disconnected.
   */
  diameter(): number {
    let maxDist = 0;
    for (const node of this.nodes) {
      const dists = this.bfsDistances(node);
      for (const [, d] of dists) {
        if (d === -1) return Infinity;
        if (d > maxDist) maxDist = d;
      }
    }
    return maxDist;
  }
}

// ── Centrality Measures ───────────────────────────────────────────────────

export interface CentralityResult {
  /** Map from node ID to centrality value. */
  values: Map<number, number>;
}

/**
 * Degree centrality: degree(v) / (n − 1).
 */
export function degreeCentrality(g: Graph): CentralityResult {
  const n = g.nodeCount;
  const values = new Map<number, number>();
  for (const node of g.nodes) {
    values.set(node, n > 1 ? g.degree(node) / (n - 1) : 0);
  }
  return { values };
}

/**
 * Closeness centrality: (n − 1) / Σ d(v, u).
 */
export function closenessCentrality(g: Graph): CentralityResult {
  const n = g.nodeCount;
  const values = new Map<number, number>();

  for (const node of g.nodes) {
    const dists = g.bfsDistances(node);
    let sum = 0;
    let reachable = 0;
    for (const [, d] of dists) {
      if (d > 0) {
        sum += d;
        reachable++;
      }
    }
    values.set(node, sum > 0 ? reachable / sum : 0);
  }
  return { values };
}

/**
 * Betweenness centrality (Brandes' algorithm).
 * Normalised by 2/((n−1)(n−2)) for undirected graphs.
 */
export function betweennessCentrality(g: Graph): CentralityResult {
  const nodeList = g.nodes;
  const n = nodeList.length;
  const cb = new Map<number, number>();
  for (const v of nodeList) cb.set(v, 0);

  for (const s of nodeList) {
    // BFS
    const stack: number[] = [];
    const pred = new Map<number, number[]>();
    const sigma = new Map<number, number>();
    const dist = new Map<number, number>();
    const delta = new Map<number, number>();

    for (const v of nodeList) {
      pred.set(v, []);
      sigma.set(v, 0);
      dist.set(v, -1);
      delta.set(v, 0);
    }
    sigma.set(s, 1);
    dist.set(s, 0);
    const queue = [s];
    let head = 0;

    while (head < queue.length) {
      const v = queue[head++];
      stack.push(v);
      for (const w of g.neighbours(v)) {
        if (dist.get(w) === -1) {
          dist.set(w, dist.get(v)! + 1);
          queue.push(w);
        }
        if (dist.get(w) === dist.get(v)! + 1) {
          sigma.set(w, sigma.get(w)! + sigma.get(v)!);
          pred.get(w)!.push(v);
        }
      }
    }

    while (stack.length > 0) {
      const w = stack.pop()!;
      for (const v of pred.get(w)!) {
        const d = (sigma.get(v)! / sigma.get(w)!) * (1 + delta.get(w)!);
        delta.set(v, delta.get(v)! + d);
      }
      if (w !== s) {
        cb.set(w, cb.get(w)! + delta.get(w)!);
      }
    }
  }

  // Normalise
  const norm = g.directed ? 1 / ((n - 1) * (n - 2)) : 2 / ((n - 1) * (n - 2));
  if (n > 2) {
    for (const v of nodeList) {
      cb.set(v, cb.get(v)! * norm);
    }
  }

  return { values: cb };
}

/**
 * Eigenvector centrality via power iteration.
 */
export function eigenvectorCentrality(
  g: Graph,
  options: { maxIterations?: number; tolerance?: number } = {},
): CentralityResult {
  const { maxIterations = 100, tolerance = 1e-8 } = options;
  const nodeList = g.nodes;
  const n = nodeList.length;
  const nodeIdx = new Map<number, number>();
  for (let i = 0; i < n; i++) nodeIdx.set(nodeList[i], i);

  let x = new Array(n).fill(1 / Math.sqrt(n));

  for (let iter = 0; iter < maxIterations; iter++) {
    const xNew = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (const nbr of g.neighbours(nodeList[i])) {
        xNew[i] += x[nodeIdx.get(nbr)!];
      }
    }
    // Normalise
    let norm = 0;
    for (let i = 0; i < n; i++) norm += xNew[i] * xNew[i];
    norm = Math.sqrt(norm);
    if (norm === 0) break;
    for (let i = 0; i < n; i++) xNew[i] /= norm;

    // Check convergence
    let maxDiff = 0;
    for (let i = 0; i < n; i++) maxDiff = Math.max(maxDiff, Math.abs(xNew[i] - x[i]));
    x = xNew;
    if (maxDiff < tolerance) break;
  }

  const values = new Map<number, number>();
  for (let i = 0; i < n; i++) values.set(nodeList[i], x[i]);
  return { values };
}

/**
 * PageRank centrality.
 *
 * @param g  Graph.
 * @param options.damping  Damping factor (default 0.85).
 * @param options.maxIterations  Max iterations (default 100).
 * @param options.tolerance  Convergence tolerance (default 1e-8).
 */
export function pageRank(
  g: Graph,
  options: { damping?: number; maxIterations?: number; tolerance?: number } = {},
): CentralityResult {
  const { damping = 0.85, maxIterations = 100, tolerance = 1e-8 } = options;
  const nodeList = g.nodes;
  const n = nodeList.length;
  const nodeIdx = new Map<number, number>();
  for (let i = 0; i < n; i++) nodeIdx.set(nodeList[i], i);

  let pr = new Array(n).fill(1 / n);

  for (let iter = 0; iter < maxIterations; iter++) {
    const prNew = new Array(n).fill((1 - damping) / n);

    for (let i = 0; i < n; i++) {
      const nbrs = g.neighbours(nodeList[i]);
      if (nbrs.length === 0) {
        // Dangling node: distribute evenly
        for (let j = 0; j < n; j++) prNew[j] += damping * pr[i] / n;
      } else {
        const share = damping * pr[i] / nbrs.length;
        for (const nbr of nbrs) prNew[nodeIdx.get(nbr)!] += share;
      }
    }

    let maxDiff = 0;
    for (let i = 0; i < n; i++) maxDiff = Math.max(maxDiff, Math.abs(prNew[i] - pr[i]));
    pr = prNew;
    if (maxDiff < tolerance) break;
  }

  const values = new Map<number, number>();
  for (let i = 0; i < n; i++) values.set(nodeList[i], pr[i]);
  return { values };
}

// ── Community Detection ───────────────────────────────────────────────────

export interface CommunityResult {
  /** Map from node to community label. */
  communities: Map<number, number>;
  /** Number of communities. */
  nCommunities: number;
  /** Modularity of the partition. */
  modularity: number;
}

/**
 * Community detection via label propagation.
 *
 * Each node adopts the most frequent label among its neighbours.
 * Ties are broken randomly. Runs until convergence or maxIterations.
 */
export function labelPropagation(
  g: Graph,
  maxIterations = 100,
): CommunityResult {
  const nodeList = g.nodes;
  const labels = new Map<number, number>();
  for (const node of nodeList) labels.set(node, node);

  for (let iter = 0; iter < maxIterations; iter++) {
    let changed = false;

    // Process nodes in random-ish order (rotate by iteration)
    const order = [...nodeList];
    for (let i = order.length - 1; i > 0; i--) {
      const j = (i + iter) % (i + 1);
      [order[i], order[j]] = [order[j], order[i]];
    }

    for (const node of order) {
      const nbrs = g.neighbours(node);
      if (nbrs.length === 0) continue;

      // Count neighbour labels
      const counts = new Map<number, number>();
      for (const nbr of nbrs) {
        const l = labels.get(nbr)!;
        counts.set(l, (counts.get(l) ?? 0) + 1);
      }

      // Find most frequent
      let maxCount = 0;
      let bestLabel = labels.get(node)!;
      for (const [label, count] of counts) {
        if (count > maxCount) {
          maxCount = count;
          bestLabel = label;
        }
      }

      if (bestLabel !== labels.get(node)) {
        labels.set(node, bestLabel);
        changed = true;
      }
    }

    if (!changed) break;
  }

  const nCommunities = new Set(labels.values()).size;
  const mod = modularity(g, labels);

  return { communities: labels, nCommunities, modularity: mod };
}

/**
 * Compute modularity Q for a given partition.
 *
 * Q = (1 / 2m) Σᵢⱼ [Aᵢⱼ − kᵢkⱼ/(2m)] δ(cᵢ, cⱼ)
 */
export function modularity(g: Graph, communities: Map<number, number>): number {
  const m = g.edgeCount;
  if (m === 0) return 0;

  const nodeList = g.nodes;
  let Q = 0;

  for (const i of nodeList) {
    for (const j of nodeList) {
      if (communities.get(i) !== communities.get(j)) continue;
      const aij = g.hasEdge(i, j) ? 1 : 0;
      Q += aij - (g.degree(i) * g.degree(j)) / (2 * m);
    }
  }

  return Q / (2 * m);
}

// ── Helpers ───────────────────────────────────────────────────────────────

function edgeKey(u: number, v: number): string {
  return `${u},${v}`;
}
