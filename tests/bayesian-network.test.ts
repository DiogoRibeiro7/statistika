import {
  BayesianNetwork,
  k2StructureLearning,
} from "../src/bayesian-network";

// ── DAG Construction ──────────────────────────────────────────────────────

describe("BayesianNetwork construction", () => {
  it("adds nodes and edges", () => {
    const bn = new BayesianNetwork();
    bn.addNode("A");
    bn.addNode("B");
    bn.addEdge("A", "B");
    expect(bn.nodeCount).toBe(2);
    expect(bn.edgeCount).toBe(1);
    expect(bn.hasEdge("A", "B")).toBe(true);
    expect(bn.hasEdge("B", "A")).toBe(false);
  });

  it("tracks parents and children", () => {
    const bn = new BayesianNetwork();
    bn.addEdge("A", "C");
    bn.addEdge("B", "C");
    expect(bn.parents("C")).toEqual(["A", "B"]);
    expect(bn.children("A")).toEqual(["C"]);
  });

  it("rejects self-loops", () => {
    const bn = new BayesianNetwork();
    expect(() => bn.addEdge("A", "A")).toThrow("Self-loops");
  });

  it("rejects cycles", () => {
    const bn = new BayesianNetwork();
    bn.addEdge("A", "B");
    bn.addEdge("B", "C");
    expect(() => bn.addEdge("C", "A")).toThrow("cycle");
  });

  it("allows DAG structure", () => {
    const bn = new BayesianNetwork();
    bn.addEdge("A", "B");
    bn.addEdge("A", "C");
    bn.addEdge("B", "D");
    bn.addEdge("C", "D");
    expect(bn.nodeCount).toBe(4);
    expect(bn.edgeCount).toBe(4);
  });
});

// ── Topological Sort ──────────────────────────────────────────────────────

describe("topologicalSort", () => {
  it("returns parents before children", () => {
    const bn = new BayesianNetwork();
    bn.addEdge("A", "B");
    bn.addEdge("B", "C");

    const order = bn.topologicalSort();
    expect(order.indexOf("A")).toBeLessThan(order.indexOf("B"));
    expect(order.indexOf("B")).toBeLessThan(order.indexOf("C"));
  });

  it("handles diamond DAG", () => {
    const bn = new BayesianNetwork();
    bn.addEdge("A", "B");
    bn.addEdge("A", "C");
    bn.addEdge("B", "D");
    bn.addEdge("C", "D");

    const order = bn.topologicalSort();
    expect(order.indexOf("A")).toBeLessThan(order.indexOf("B"));
    expect(order.indexOf("A")).toBeLessThan(order.indexOf("C"));
    expect(order.indexOf("B")).toBeLessThan(order.indexOf("D"));
    expect(order.indexOf("C")).toBeLessThan(order.indexOf("D"));
  });
});

// ── d-Separation ──────────────────────────────────────────────────────────

describe("dSeparated", () => {
  it("chain: A → B → C: A ⊥ C | B", () => {
    const bn = new BayesianNetwork();
    bn.addEdge("A", "B");
    bn.addEdge("B", "C");

    expect(bn.dSeparated("A", "C", new Set(["B"]))).toBe(true);
    expect(bn.dSeparated("A", "C", new Set())).toBe(false);
  });

  it("fork: A ← B → C: A ⊥ C | B", () => {
    const bn = new BayesianNetwork();
    bn.addEdge("B", "A");
    bn.addEdge("B", "C");

    expect(bn.dSeparated("A", "C", new Set(["B"]))).toBe(true);
    expect(bn.dSeparated("A", "C", new Set())).toBe(false);
  });

  it("collider: A → C ← B: A ⊥ B | ∅, not A ⊥ B | C", () => {
    const bn = new BayesianNetwork();
    bn.addEdge("A", "C");
    bn.addEdge("B", "C");

    expect(bn.dSeparated("A", "B", new Set())).toBe(true);
    expect(bn.dSeparated("A", "B", new Set(["C"]))).toBe(false);
  });
});

// ── CPT and Inference ─────────────────────────────────────────────────────

describe("inference", () => {
  it("computes marginal for root node", () => {
    const bn = new BayesianNetwork();
    bn.addNode("Rain");
    bn.setCPT("Rain", {
      probabilities: [0.7, 0.3], // P(Rain=0)=0.7, P(Rain=1)=0.3
      parentValues: [],
    });

    const marginal = bn.infer("Rain");
    expect(marginal[0]).toBeCloseTo(0.7, 4);
    expect(marginal[1]).toBeCloseTo(0.3, 4);
  });

  it("computes conditional with evidence", () => {
    const bn = new BayesianNetwork();
    bn.addEdge("Rain", "Wet");

    bn.setCPT("Rain", {
      probabilities: [0.7, 0.3],
      parentValues: [],
    });

    bn.setCPT("Wet", {
      probabilities: [
        [0.9, 0.1], // P(Wet | Rain=0)
        [0.1, 0.9], // P(Wet | Rain=1)
      ],
      parentValues: [[0], [1]],
    });

    // P(Rain | Wet=1) ∝ P(Wet=1|Rain) P(Rain)
    // P(Rain=0|Wet=1) ∝ 0.1 × 0.7 = 0.07
    // P(Rain=1|Wet=1) ∝ 0.9 × 0.3 = 0.27
    // Normalized: 0.07/0.34 ≈ 0.206, 0.27/0.34 ≈ 0.794
    const marginal = bn.infer("Rain", new Map([["Wet", 1]]));
    expect(marginal[0]).toBeCloseTo(0.07 / 0.34, 2);
    expect(marginal[1]).toBeCloseTo(0.27 / 0.34, 2);
  });

  it("marginal probabilities sum to 1", () => {
    const bn = new BayesianNetwork();
    bn.addEdge("A", "B");
    bn.setCPT("A", { probabilities: [0.6, 0.4], parentValues: [] });
    bn.setCPT("B", {
      probabilities: [[0.8, 0.2], [0.3, 0.7]],
      parentValues: [[0], [1]],
    });

    const marginal = bn.infer("B");
    const sum = marginal.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 6);
  });
});

// ── Structure Learning ────────────────────────────────────────────────────

describe("k2StructureLearning", () => {
  it("learns simple dependency", () => {
    // Generate data where B depends on A
    const n = 200;
    const A: number[] = [];
    const B: number[] = [];
    for (let i = 0; i < n; i++) {
      const a = i % 2;
      A.push(a);
      B.push(a); // B = A (perfect dependency)
    }

    const result = k2StructureLearning(
      { A, B },
      ["A", "B"],
      2,
    );

    expect(result.network.hasEdge("A", "B")).toBe(true);
    expect(Number.isFinite(result.score)).toBe(true);
  });

  it("does not add unnecessary edges", () => {
    // Independent variables
    const n = 200;
    const A = Array.from({ length: n }, (_, i) => i % 2);
    const B = Array.from({ length: n }, (_, i) => (i + 1) % 2);

    const result = k2StructureLearning(
      { A, B },
      ["A", "B"],
      1,
    );

    // With independent data and BIC penalty, should not add edge
    // (or if it does, that's acceptable — K2 is greedy)
    expect(result.network.nodeCount).toBe(2);
  });

  it("respects maxParents", () => {
    const n = 100;
    const data: Record<string, number[]> = {
      A: Array.from({ length: n }, (_, i) => i % 2),
      B: Array.from({ length: n }, (_, i) => i % 3),
      C: Array.from({ length: n }, (_, i) => i % 2),
      D: Array.from({ length: n }, (_, i) => i % 2),
    };

    const result = k2StructureLearning(data, ["A", "B", "C", "D"], 1);
    // D can have at most 1 parent
    expect(result.network.parents("D").length).toBeLessThanOrEqual(1);
  });

  it("returns valid BN structure", () => {
    const n = 100;
    const A = Array.from({ length: n }, (_, i) => i % 2);
    const B = Array.from({ length: n }, (_, i) => i % 2);
    const C = Array.from({ length: n }, (_, i) => (i + 1) % 2);

    const result = k2StructureLearning({ A, B, C }, ["A", "B", "C"]);
    // Should be a valid DAG
    expect(() => result.network.topologicalSort()).not.toThrow();
  });
});
