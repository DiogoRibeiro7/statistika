import { DataFrame } from "../src/dataframe";
import { LazyDataFrame } from "../src/lazy-dataframe";

const sampleDf = () =>
  DataFrame.create({
    name: ["Alice", "Bob", "Carol", "Dave", "Eve"],
    age: [30, 25, 35, 28, 22],
    score: [85, 92, 78, 95, 88],
    dept: ["eng", "sales", "eng", "sales", "eng"],
  });

describe("LazyDataFrame", () => {
  it("collects unchanged DataFrame", () => {
    const df = sampleDf();
    const result = LazyDataFrame.from(df).collect();
    expect(result.nRows).toBe(5);
    expect(result.columnNames).toEqual(["name", "age", "score", "dept"]);
  });

  it("defers and applies select", () => {
    const df = sampleDf();
    const result = LazyDataFrame.from(df).select(["name", "score"]).collect();
    expect(result.columnNames).toEqual(["name", "score"]);
    expect(result.nRows).toBe(5);
  });

  it("defers and applies filter", () => {
    const df = sampleDf();
    const result = LazyDataFrame.from(df)
      .filter((row) => (row.age as number) > 25)
      .collect();
    expect(result.nRows).toBe(3);
    expect(result.column("name")).toEqual(["Alice", "Carol", "Dave"]);
  });

  it("defers and applies mutate", () => {
    const df = sampleDf();
    const result = LazyDataFrame.from(df)
      .mutate("doubleScore", (row) => (row.score as number) * 2)
      .collect();
    expect(result.columnNames).toContain("doubleScore");
    expect(result.column("doubleScore")).toEqual([170, 184, 156, 190, 176]);
  });

  it("defers and applies sort ascending", () => {
    const df = sampleDf();
    const result = LazyDataFrame.from(df).sort("score", "asc").collect();
    expect(result.column("score")).toEqual([78, 85, 88, 92, 95]);
  });

  it("defers and applies sort descending", () => {
    const df = sampleDf();
    const result = LazyDataFrame.from(df).sort("score", "desc").collect();
    expect(result.column("score")).toEqual([95, 92, 88, 85, 78]);
  });

  it("defers and applies head", () => {
    const df = sampleDf();
    const result = LazyDataFrame.from(df).head(3).collect();
    expect(result.nRows).toBe(3);
    expect(result.column("name")).toEqual(["Alice", "Bob", "Carol"]);
  });

  it("defers and applies tail", () => {
    const df = sampleDf();
    const result = LazyDataFrame.from(df).tail(2).collect();
    expect(result.nRows).toBe(2);
    expect(result.column("name")).toEqual(["Dave", "Eve"]);
  });

  it("chains multiple operations", () => {
    const df = sampleDf();
    const result = LazyDataFrame.from(df)
      .filter((row) => (row.age as number) > 24)
      .select(["name", "score"])
      .sort("score", "desc")
      .head(2)
      .collect();
    expect(result.nRows).toBe(2);
    expect(result.columnNames).toEqual(["name", "score"]);
    expect(result.column("score")).toEqual([95, 92]);
  });

  it("collectRecords returns array of objects", () => {
    const df = sampleDf();
    const records = LazyDataFrame.from(df).head(2).collectRecords();
    expect(records).toEqual([
      { name: "Alice", age: 30, score: 85, dept: "eng" },
      { name: "Bob", age: 25, score: 92, dept: "sales" },
    ]);
  });

  it("explain returns a query plan string", () => {
    const df = sampleDf();
    const plan = LazyDataFrame.from(df)
      .filter((row) => (row.age as number) > 25)
      .sort("score", "desc")
      .head(3)
      .explain();
    expect(plan).toContain("LazyDataFrame Query Plan:");
    expect(plan).toContain("Filter");
    expect(plan).toContain("Sort");
    expect(plan).toContain("Head");
  });

  // ── Optimization tests ──────────────────────────────────────────────────

  it("combines consecutive selects via intersection", () => {
    const df = sampleDf();
    const result = LazyDataFrame.from(df)
      .select(["name", "age", "score"])
      .select(["name", "score"])
      .collect();
    expect(result.columnNames).toEqual(["name", "score"]);
  });

  it("pushes filter before sort for optimization", () => {
    const df = sampleDf();
    // filter after sort should produce same result as filter before sort
    const result = LazyDataFrame.from(df)
      .sort("score", "desc")
      .filter((row) => (row.age as number) > 25)
      .collect();
    expect(result.nRows).toBe(3);
    // Result should be sorted descending by score
    const scores = result.column("score") as number[];
    for (let i = 1; i < scores.length; i++) {
      expect(scores[i]).toBeLessThanOrEqual(scores[i - 1]);
    }
  });

  it("is immutable — chaining returns new instances", () => {
    const df = sampleDf();
    const lazy1 = LazyDataFrame.from(df);
    const lazy2 = lazy1.filter((row) => (row.age as number) > 25);
    const lazy3 = lazy1.head(2);

    // lazy1 should still collect the full DataFrame
    expect(lazy1.collect().nRows).toBe(5);
    expect(lazy2.collect().nRows).toBe(3);
    expect(lazy3.collect().nRows).toBe(2);
  });
});
