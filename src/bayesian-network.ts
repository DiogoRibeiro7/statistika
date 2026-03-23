/**
 * Bayesian Networks — DAG-based probabilistic models.
 *
 * - **DAG construction** — add nodes, edges, validate acyclicity.
 * - **Conditional probability tables** — discrete BN parameterisation.
 * - **Inference** — variable elimination, marginal probabilities.
 * - **Structure learning** — K2 algorithm (greedy score-based).
 * - **d-separation** — test conditional independence in the DAG.
 * - **Topological sort** — for efficient forward/backward passes.
 */

// ── DAG Data Structure ────────────────────────────────────────────────────

/**
 * A Bayesian Network represented as a directed acyclic graph (DAG).
 *
 * Provides methods for constructing the graph (adding nodes and edges),
 * setting conditional probability tables, performing inference via
 * enumeration, testing d-separation, and computing topological orderings.
 *
 * @example
 * ```ts
 * const bn = new BayesianNetwork();
 * bn.addNode("Rain");
 * bn.addNode("Sprinkler");
 * bn.addNode("WetGrass");
 * bn.addEdge("Rain", "WetGrass");
 * bn.addEdge("Sprinkler", "WetGrass");
 * bn.setCPT("Rain", { probabilities: [0.8, 0.2], parentValues: [] });
 * ```
 */
export class BayesianNetwork {
  private _parents: Map<string, string[]>;
  private _children: Map<string, string[]>;
  private _cpts: Map<string, ConditionalProbabilityTable>;
  private _nodeOrder: string[];

  /** Create an empty Bayesian Network with no nodes or edges. */
  constructor() {
    this._parents = new Map();
    this._children = new Map();
    this._cpts = new Map();
    this._nodeOrder = [];
  }

  // ── Construction ────────────────────────────────────────────────────────

  /**
   * Add a node to the network. If the node already exists, this is a no-op.
   *
   * @param name - Unique name for the node
   */
  addNode(name: string): void {
    if (this._parents.has(name)) return;
    this._parents.set(name, []);
    this._children.set(name, []);
    this._nodeOrder.push(name);
  }

  /**
   * Add a directed edge from parent to child.
   *
   * Automatically creates the parent and child nodes if they do not exist.
   * Validates that the new edge does not introduce a cycle in the DAG.
   *
   * @param parent - Name of the parent node
   * @param child - Name of the child node
   * @throws {Error} If parent === child (self-loops not allowed)
   * @throws {Error} If the edge would create a cycle in the DAG
   */
  addEdge(parent: string, child: string): void {
    this.addNode(parent);
    this.addNode(child);

    // Check for cycles
    if (parent === child) throw new Error("Self-loops not allowed");
    if (this._isAncestor(child, parent)) {
      throw new Error(`Adding ${parent} → ${child} would create a cycle`);
    }

    this._parents.get(child)!.push(parent);
    this._children.get(parent)!.push(child);
  }

  /**
   * Set the conditional probability table (CPT) for a node.
   *
   * For root nodes (no parents), provide probabilities as a flat array
   * where probabilities[i] = P(node = i). For nodes with parents,
   * provide probabilities as a 2D array where each row corresponds
   * to a parent value combination listed in parentValues.
   *
   * @param node - Node name (must already exist in the network)
   * @param cpt - Conditional probability table (see {@link ConditionalProbabilityTable})
   * @throws {Error} If the node does not exist in the network
   */
  setCPT(node: string, cpt: ConditionalProbabilityTable): void {
    if (!this._parents.has(node)) throw new Error(`Node "${node}" not found`);
    this._cpts.set(node, cpt);
  }

  // ── Properties ──────────────────────────────────────────────────────────

  /** All node names in insertion order. */
  get nodes(): string[] {
    return [...this._nodeOrder];
  }

  /** Total number of nodes in the network. */
  get nodeCount(): number {
    return this._parents.size;
  }

  /** Total number of directed edges in the network. */
  get edgeCount(): number {
    let count = 0;
    for (const [, children] of this._children) count += children.length;
    return count;
  }

  /**
   * Get the parent nodes of a given node.
   *
   * @param node - Node name
   * @returns Array of parent node names (empty if root or node not found)
   */
  parents(node: string): string[] {
    return [...(this._parents.get(node) ?? [])];
  }

  /**
   * Get the child nodes of a given node.
   *
   * @param node - Node name
   * @returns Array of child node names (empty if leaf or node not found)
   */
  children(node: string): string[] {
    return [...(this._children.get(node) ?? [])];
  }

  /**
   * Check whether a directed edge from parent to child exists.
   *
   * @param parent - Parent node name
   * @param child - Child node name
   * @returns True if the edge exists, false otherwise
   */
  hasEdge(parent: string, child: string): boolean {
    return this._parents.get(child)?.includes(parent) ?? false;
  }

  /**
   * Get the conditional probability table for a node.
   *
   * @param node - Node name
   * @returns The CPT if set, or undefined if no CPT has been assigned
   */
  getCPT(node: string): ConditionalProbabilityTable | undefined {
    return this._cpts.get(node);
  }

  // ── Topological Sort ────────────────────────────────────────────────────

  /**
   * Return nodes in topological order (parents before children).
   *
   * Uses depth-first search. The resulting order guarantees that for every
   * edge (u, v) in the DAG, u appears before v in the returned array.
   *
   * @returns Array of node names in topological order
   * @throws {Error} If a cycle is detected (should not happen if edges were validated)
   */
  topologicalSort(): string[] {
    const visited = new Set<string>();
    const result: string[] = [];
    const visiting = new Set<string>();

    const visit = (node: string) => {
      if (visited.has(node)) return;
      if (visiting.has(node)) throw new Error("Cycle detected");
      visiting.add(node);
      for (const parent of this._parents.get(node) ?? []) {
        visit(parent);
      }
      visiting.delete(node);
      visited.add(node);
      result.push(node);
    };

    for (const node of this._nodeOrder) visit(node);
    return result;
  }

  // ── d-Separation ────────────────────────────────────────────────────────

  /**
   * Test d-separation: are X and Y conditionally independent given Z?
   *
   * Uses the Bayes-Ball algorithm to determine whether all paths between
   * x and y are blocked by the conditioning set z. Two nodes are d-separated
   * given Z if no active path connects them (i.e., the information flow is
   * blocked by observations or lack thereof at intermediate nodes).
   *
   * @param x - Source node name
   * @param y - Target node name
   * @param z - Set of conditioning (observed) node names
   * @returns True if x and y are d-separated given z (conditionally independent),
   *   false otherwise
   *
   * @example
   * ```ts
   * // A -> B -> C: A and C are d-separated given B
   * bn.dSeparated("A", "C", new Set(["B"])); // true
   * bn.dSeparated("A", "C", new Set());       // false
   * ```
   */
  dSeparated(x: string, y: string, z: Set<string>): boolean {
    // BFS-like reachability using Bayes-Ball rules
    const visited = new Set<string>();
    // Queue items: [node, direction] where direction = "up" (from child) or "down" (from parent)
    const queue: [string, "up" | "down"][] = [[x, "up"]];
    const seen = new Set<string>();

    while (queue.length > 0) {
      const [node, dir] = queue.shift()!;
      const key = `${node},${dir}`;
      if (seen.has(key)) continue;
      seen.add(key);

      if (node === y) return false; // reachable → not d-separated

      const isObserved = z.has(node);

      if (dir === "up" && !isObserved) {
        // Visited from child, node not observed: can go to parents and children
        for (const parent of this._parents.get(node) ?? []) {
          queue.push([parent, "up"]);
        }
        for (const child of this._children.get(node) ?? []) {
          queue.push([child, "down"]);
        }
      } else if (dir === "down" && !isObserved) {
        // Visited from parent, node not observed: can go to children
        for (const child of this._children.get(node) ?? []) {
          queue.push([child, "down"]);
        }
      } else if (dir === "down" && isObserved) {
        // Visited from parent, node observed: can go to parents (explaining away)
        for (const parent of this._parents.get(node) ?? []) {
          queue.push([parent, "up"]);
        }
      }
      // dir === "up" && isObserved: blocked
    }

    return true; // not reachable → d-separated
  }

  // ── Inference: Variable Elimination ─────────────────────────────────────

  /**
   * Compute marginal probability distribution P(query | evidence) via enumeration.
   *
   * Enumerates all assignments to hidden (non-evidence, non-query) variables,
   * computing the joint probability for each and marginalizing. Suitable for
   * small discrete networks; complexity is exponential in the number of hidden nodes.
   *
   * @param query - Name of the query variable
   * @param evidence - Map of observed variable name to observed integer state value
   * @returns Array of probabilities where result[i] = P(query = i | evidence),
   *   normalized to sum to 1
   * @throws {Error} If the query node does not exist in the network
   * @throws {Error} If no CPT has been set for the query node
   *
   * @example
   * ```ts
   * const probs = bn.infer("WetGrass", new Map([["Rain", 1]]));
   * console.log(probs); // [P(WetGrass=0|Rain=1), P(WetGrass=1|Rain=1)]
   * ```
   */
  infer(
    query: string,
    evidence: Map<string, number> = new Map(),
  ): number[] {
    if (!this._parents.has(query)) throw new Error(`Node "${query}" not found`);

    const cpt = this._cpts.get(query);
    if (!cpt) throw new Error(`No CPT for node "${query}"`);

    const topoOrder = this.topologicalSort();
    const nStates = cpt.probabilities.length > 0
      ? (Array.isArray(cpt.probabilities[0]) ? (cpt.probabilities[0] as number[]).length : cpt.probabilities.length)
      : 0;

    // Simple forward sampling / enumeration for small networks
    // Enumerate all combinations of hidden variables
    const hiddenNodes = topoOrder.filter(
      (n) => n !== query && !evidence.has(n),
    );

    // Get number of states per node
    const nodeStates = new Map<string, number>();
    for (const node of topoOrder) {
      const nodeCpt = this._cpts.get(node);
      if (nodeCpt) {
        const probs = nodeCpt.probabilities;
        if (probs.length > 0 && Array.isArray(probs[0])) {
          nodeStates.set(node, (probs[0] as number[]).length);
        } else {
          nodeStates.set(node, probs.length);
        }
      } else {
        nodeStates.set(node, 2); // default binary
      }
    }

    const queryStates = nodeStates.get(query) ?? nStates;
    const marginal = new Array(queryStates).fill(0);

    // For small networks, enumerate
    const enumerate = (
      idx: number,
      assignment: Map<string, number>,
    ) => {
      if (idx === hiddenNodes.length) {
        // Compute joint probability for each query state
        for (let qv = 0; qv < queryStates; qv++) {
          assignment.set(query, qv);
          let prob = 1;
          for (const node of topoOrder) {
            const p = this._lookupCPT(node, assignment);
            prob *= p;
          }
          marginal[qv] += prob;
        }
        return;
      }

      const node = hiddenNodes[idx];
      const nS = nodeStates.get(node) ?? 2;
      for (let v = 0; v < nS; v++) {
        assignment.set(node, v);
        enumerate(idx + 1, assignment);
      }
      assignment.delete(node);
    };

    const assignment = new Map<string, number>(evidence);
    enumerate(0, assignment);

    // Normalise
    let sum = 0;
    for (const p of marginal) sum += p;
    if (sum > 0) {
      for (let i = 0; i < marginal.length; i++) marginal[i] /= sum;
    }

    return marginal;
  }

  // ── Internal ────────────────────────────────────────────────────────────

  private _isAncestor(potentialAncestor: string, node: string): boolean {
    const visited = new Set<string>();
    const queue = [potentialAncestor];
    while (queue.length > 0) {
      const current = queue.shift()!;
      if (visited.has(current)) continue;
      visited.add(current);
      for (const child of this._children.get(current) ?? []) {
        if (child === node) return true;
        queue.push(child);
      }
    }
    return false;
  }

  private _lookupCPT(node: string, assignment: Map<string, number>): number {
    const cpt = this._cpts.get(node);
    if (!cpt) return 1;

    const nodeVal = assignment.get(node) ?? 0;
    const parentNames = this._parents.get(node) ?? [];

    if (parentNames.length === 0) {
      // Root node: simple probability table
      const probs = cpt.probabilities as number[];
      return probs[nodeVal] ?? 0;
    }

    // Find row matching parent assignment
    const parentVals = parentNames.map((p) => assignment.get(p) ?? 0);

    for (let r = 0; r < cpt.parentValues.length; r++) {
      const row = cpt.parentValues[r];
      let match = true;
      for (let i = 0; i < parentNames.length; i++) {
        if (row[i] !== parentVals[i]) { match = false; break; }
      }
      if (match) {
        const probs = cpt.probabilities[r] as number[];
        return probs[nodeVal] ?? 0;
      }
    }

    return 0;
  }
}

// ── Conditional Probability Table ─────────────────────────────────────────

/**
 * A conditional probability table (CPT) for a discrete random variable.
 *
 * Encodes P(node | parents) as a table of probabilities indexed by
 * parent value combinations.
 */
export interface ConditionalProbabilityTable {
  /**
   * For root nodes: probabilities[i] = P(node = i).
   * For nodes with parents: probabilities[row][i] = P(node = i | parents = parentValues[row]).
   */
  probabilities: number[] | number[][];
  /** Parent value combinations. parentValues[row] = [val_parent1, val_parent2, ...]. */
  parentValues: number[][];
}

// ── Structure Learning ────────────────────────────────────────────────────

/**
 * Result of Bayesian network structure learning.
 *
 * Contains the learned network (DAG with edges but no CPTs) and the
 * overall BIC score of the learned structure.
 */
export interface StructureLearningResult {
  /** Learned Bayesian network. */
  network: BayesianNetwork;
  /** BIC/MDL score of the learned structure. */
  score: number;
}

/**
 * Learn Bayesian Network structure using the K2 algorithm (greedy score-based).
 *
 * Requires a topological node ordering (causal order). For each node in order,
 * greedily adds the parent from earlier nodes that most improves the BIC score,
 * up to the maximum number of parents allowed.
 *
 * The BIC local score for a node with parent set Pa is:
 *   Score = LL - 0.5 * k * ln(n)
 * where LL is the log-likelihood, k is the number of free parameters, and n is
 * the sample size.
 *
 * @param data - Discrete data as a record mapping variable names to integer-coded
 *   observation arrays. All arrays must have the same length.
 * @param nodeOrder - Ordered variable names. Parents of a node can only come from
 *   earlier positions in this ordering.
 * @param maxParents - Maximum number of parents per node (default 3)
 * @returns A {@link StructureLearningResult} containing the learned network and total BIC score
 *
 * @example
 * ```ts
 * const data = { A: [0,1,0,1], B: [0,0,1,1], C: [0,1,1,1] };
 * const result = k2StructureLearning(data, ["A", "B", "C"], 2);
 * console.log(result.network.nodes);  // ["A", "B", "C"]
 * console.log(result.score);          // BIC score
 * ```
 */
export function k2StructureLearning(
  data: Record<string, number[]>,
  nodeOrder: string[],
  maxParents = 3,
): StructureLearningResult {
  const n = data[nodeOrder[0]].length;
  const bn = new BayesianNetwork();
  let totalScore = 0;

  for (const node of nodeOrder) bn.addNode(node);

  // For each node, greedily add best parent
  for (let idx = 0; idx < nodeOrder.length; idx++) {
    const node = nodeOrder[idx];
    const candidates = nodeOrder.slice(0, idx);
    const currentParents: string[] = [];

    let bestScore = computeLocalScore(data, node, currentParents, n);

    for (let p = 0; p < maxParents && candidates.length > 0; p++) {
      let bestCandidate: string | null = null;
      let bestNewScore = bestScore;

      for (const cand of candidates) {
        if (currentParents.includes(cand)) continue;
        const testParents = [...currentParents, cand];
        const score = computeLocalScore(data, node, testParents, n);
        if (score > bestNewScore) {
          bestNewScore = score;
          bestCandidate = cand;
        }
      }

      if (bestCandidate) {
        currentParents.push(bestCandidate);
        bestScore = bestNewScore;
      } else {
        break;
      }
    }

    for (const parent of currentParents) {
      bn.addEdge(parent, node);
    }
    totalScore += bestScore;
  }

  return { network: bn, score: totalScore };
}

// ── Helpers ───────────────────────────────────────────────────────────────

/** BIC-based local score for a node given its parents. */
function computeLocalScore(
  data: Record<string, number[]>,
  node: string,
  parents: string[],
  n: number,
): number {
  const nodeData = data[node];
  const nodeStates = new Set(nodeData).size;

  if (parents.length === 0) {
    // Count frequencies
    const counts = new Map<number, number>();
    for (const v of nodeData) counts.set(v, (counts.get(v) ?? 0) + 1);

    let ll = 0;
    for (const [, count] of counts) {
      if (count > 0) ll += count * Math.log(count / n);
    }
    // BIC penalty
    const nParams = nodeStates - 1;
    return ll - 0.5 * nParams * Math.log(n);
  }

  // Group by parent configuration
  const groups = new Map<string, Map<number, number>>();
  const groupCounts = new Map<string, number>();

  for (let i = 0; i < n; i++) {
    const key = parents.map((p) => data[p][i]).join(",");
    if (!groups.has(key)) {
      groups.set(key, new Map());
      groupCounts.set(key, 0);
    }
    const nodeVal = nodeData[i];
    const group = groups.get(key)!;
    group.set(nodeVal, (group.get(nodeVal) ?? 0) + 1);
    groupCounts.set(key, groupCounts.get(key)! + 1);
  }

  let ll = 0;
  for (const [key, group] of groups) {
    const total = groupCounts.get(key)!;
    for (const [, count] of group) {
      if (count > 0) ll += count * Math.log(count / total);
    }
  }

  const parentConfigs = groups.size;
  const nParams = parentConfigs * (nodeStates - 1);
  return ll - 0.5 * nParams * Math.log(n);
}
