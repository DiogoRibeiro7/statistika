import {
  Graph,
  degreeCentrality,
  closenessCentrality,
  betweennessCentrality,
  eigenvectorCentrality,
  pageRank,
  labelPropagation,
  modularity,
} from "../src/graph";

// ── Graph construction ────────────────────────────────────────────────────

describe("Graph construction", () => {
  it("creates from edge list", () => {
    const g = Graph.fromEdgeList([[0, 1], [1, 2], [2, 3]]);
    expect(g.nodeCount).toBe(4);
    expect(g.edgeCount).toBe(3);
    expect(g.directed).toBe(false);
  });

  it("creates directed graph", () => {
    const g = Graph.fromEdgeList([[0, 1], [1, 2]], true);
    expect(g.directed).toBe(true);
    expect(g.hasEdge(0, 1)).toBe(true);
    expect(g.hasEdge(1, 0)).toBe(false);
  });

  it("creates from adjacency matrix", () => {
    const g = Graph.fromAdjacencyMatrix([
      [0, 1, 0],
      [1, 0, 1],
      [0, 1, 0],
    ]);
    expect(g.nodeCount).toBe(3);
    expect(g.edgeCount).toBe(2);
    expect(g.hasEdge(0, 1)).toBe(true);
    expect(g.hasEdge(0, 2)).toBe(false);
  });

  it("tracks node degrees", () => {
    const g = Graph.fromEdgeList([[0, 1], [0, 2], [0, 3]]);
    expect(g.degree(0)).toBe(3);
    expect(g.degree(1)).toBe(1);
  });

  it("tracks edge weights", () => {
    const g = Graph.fromEdgeList([[0, 1, 5], [1, 2, 3]]);
    expect(g.weight(0, 1)).toBe(5);
    expect(g.weight(1, 2)).toBe(3);
    expect(g.weight(0, 2)).toBe(0); // no edge
  });
});

// ── Graph properties ──────────────────────────────────────────────────────

describe("Graph properties", () => {
  it("density of complete graph = 1", () => {
    // K4: all 6 edges
    const g = Graph.fromEdgeList([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]);
    expect(g.density()).toBeCloseTo(1, 8);
  });

  it("density of sparse graph < 1", () => {
    const g = Graph.fromEdgeList([[0, 1], [2, 3]]);
    expect(g.density()).toBeLessThan(1);
  });

  it("clustering coefficient of complete graph = 1", () => {
    const g = Graph.fromEdgeList([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]);
    expect(g.clusteringCoefficient()).toBeCloseTo(1, 8);
  });

  it("clustering coefficient of tree = 0", () => {
    // Star graph
    const g = Graph.fromEdgeList([[0, 1], [0, 2], [0, 3], [0, 4]]);
    expect(g.clusteringCoefficient()).toBe(0);
  });

  it("connected components", () => {
    const g = Graph.fromEdgeList([[0, 1], [1, 2], [3, 4]]);
    const comps = g.connectedComponents();
    expect(comps.length).toBe(2);
  });

  it("diameter of path graph", () => {
    const g = Graph.fromEdgeList([[0, 1], [1, 2], [2, 3]]);
    expect(g.diameter()).toBe(3);
  });

  it("diameter of disconnected graph = Infinity", () => {
    const g = Graph.fromEdgeList([[0, 1], [2, 3]]);
    expect(g.diameter()).toBe(Infinity);
  });
});

// ── BFS ───────────────────────────────────────────────────────────────────

describe("BFS distances", () => {
  it("computes shortest distances from source", () => {
    const g = Graph.fromEdgeList([[0, 1], [1, 2], [2, 3]]);
    const dists = g.bfsDistances(0);
    expect(dists.get(0)).toBe(0);
    expect(dists.get(1)).toBe(1);
    expect(dists.get(2)).toBe(2);
    expect(dists.get(3)).toBe(3);
  });

  it("unreachable nodes have distance -1", () => {
    const g = Graph.fromEdgeList([[0, 1], [2, 3]]);
    const dists = g.bfsDistances(0);
    expect(dists.get(2)).toBe(-1);
  });
});

// ── Centrality ────────────────────────────────────────────────────────────

describe("degreeCentrality", () => {
  it("hub has highest centrality in star graph", () => {
    const g = Graph.fromEdgeList([[0, 1], [0, 2], [0, 3], [0, 4]]);
    const dc = degreeCentrality(g);
    expect(dc.values.get(0)).toBeCloseTo(1, 8); // connected to all
    expect(dc.values.get(1)).toBeCloseTo(0.25, 8);
  });
});

describe("closenessCentrality", () => {
  it("centre of path graph has highest closeness", () => {
    const g = Graph.fromEdgeList([[0, 1], [1, 2], [2, 3], [3, 4]]);
    const cc = closenessCentrality(g);
    // Node 2 is in the centre
    expect(cc.values.get(2)!).toBeGreaterThan(cc.values.get(0)!);
    expect(cc.values.get(2)!).toBeGreaterThan(cc.values.get(4)!);
  });
});

describe("betweennessCentrality", () => {
  it("bridge node has high betweenness", () => {
    // Two triangles connected by a bridge node 2
    const g = Graph.fromEdgeList([
      [0, 1], [1, 2], [0, 2],
      [2, 3], [3, 4], [2, 4],
    ]);
    const bc = betweennessCentrality(g);
    expect(bc.values.get(2)!).toBeGreaterThan(bc.values.get(0)!);
    expect(bc.values.get(2)!).toBeGreaterThan(bc.values.get(4)!);
  });

  it("leaf nodes have zero betweenness", () => {
    const g = Graph.fromEdgeList([[0, 1], [1, 2], [1, 3]]);
    const bc = betweennessCentrality(g);
    expect(bc.values.get(0)).toBeCloseTo(0, 8);
    expect(bc.values.get(2)).toBeCloseTo(0, 8);
    expect(bc.values.get(3)).toBeCloseTo(0, 8);
  });
});

describe("eigenvectorCentrality", () => {
  it("connected nodes in dense subgraph have higher centrality", () => {
    // Triangle + pendant
    const g = Graph.fromEdgeList([[0, 1], [1, 2], [0, 2], [2, 3]]);
    const ec = eigenvectorCentrality(g);
    // Nodes in the triangle should have higher centrality than the pendant
    expect(ec.values.get(0)!).toBeGreaterThan(ec.values.get(3)!);
  });

  it("values are normalised", () => {
    const g = Graph.fromEdgeList([[0, 1], [1, 2], [2, 0]]);
    const ec = eigenvectorCentrality(g);
    let sumSq = 0;
    for (const [, v] of ec.values) sumSq += v * v;
    expect(sumSq).toBeCloseTo(1, 4);
  });
});

describe("pageRank", () => {
  it("sums to approximately 1", () => {
    const g = Graph.fromEdgeList([[0, 1], [1, 2], [2, 0], [2, 3]]);
    const pr = pageRank(g);
    let sum = 0;
    for (const [, v] of pr.values) sum += v;
    expect(sum).toBeCloseTo(1, 4);
  });

  it("nodes with more incoming links rank higher", () => {
    // Node 2 receives from 0, 1, and 3. Node 4 receives only from 2.
    const g = Graph.fromEdgeList([[0, 2], [1, 2], [3, 2], [2, 4], [4, 3]], true);
    const pr = pageRank(g);
    expect(pr.values.get(2)!).toBeGreaterThan(pr.values.get(4)!);
  });

  it("handles dangling nodes", () => {
    const g = new Graph(true);
    g.addEdge(0, 1);
    g.addNode(2); // dangling
    const pr = pageRank(g);
    let sum = 0;
    for (const [, v] of pr.values) sum += v;
    expect(sum).toBeCloseTo(1, 4);
  });
});

// ── Community Detection ───────────────────────────────────────────────────

describe("labelPropagation", () => {
  it("detects communities in well-separated clusters", () => {
    // Two cliques connected by a single bridge
    const g = Graph.fromEdgeList([
      [0, 1], [0, 2], [1, 2], // clique 1
      [3, 4], [3, 5], [4, 5], // clique 2
      [2, 3],                 // bridge
    ]);

    const result = labelPropagation(g);
    expect(result.nCommunities).toBeLessThanOrEqual(3);
    // Nodes in same clique should share a label
    expect(result.communities.get(0)).toBe(result.communities.get(1));
    expect(result.communities.get(3)).toBe(result.communities.get(4));
  });

  it("single clique forms one community", () => {
    const g = Graph.fromEdgeList([[0, 1], [1, 2], [0, 2]]);
    const result = labelPropagation(g);
    expect(result.nCommunities).toBe(1);
  });

  it("computes modularity", () => {
    const g = Graph.fromEdgeList([[0, 1], [1, 2], [3, 4]]);
    const result = labelPropagation(g);
    expect(Number.isFinite(result.modularity)).toBe(true);
  });
});

describe("modularity", () => {
  it("higher modularity for good partitions", () => {
    const g = Graph.fromEdgeList([
      [0, 1], [0, 2], [1, 2],
      [3, 4], [3, 5], [4, 5],
      [2, 3],
    ]);

    // Good partition: {0,1,2}, {3,4,5}
    const good = new Map([[0, 0], [1, 0], [2, 0], [3, 1], [4, 1], [5, 1]]);
    // Bad partition: {0,3,4}, {1,2,5}
    const bad = new Map([[0, 0], [3, 0], [4, 0], [1, 1], [2, 1], [5, 1]]);

    expect(modularity(g, good)).toBeGreaterThan(modularity(g, bad));
  });
});
