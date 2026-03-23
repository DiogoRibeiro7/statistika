import { DataFrame } from "../src/dataframe";

// ── Creation ──────────────────────────────────────────────────────────────

describe("DataFrame creation", () => {
  it("creates from column map", () => {
    const df = DataFrame.create({
      name: ["Alice", "Bob"],
      age: [30, 25],
    });
    expect(df.nRows).toBe(2);
    expect(df.nCols).toBe(2);
    expect(df.columnNames).toEqual(["name", "age"]);
  });

  it("creates from records", () => {
    const df = DataFrame.fromRecords([
      { x: 1, y: "a" },
      { x: 2, y: "b" },
    ]);
    expect(df.nRows).toBe(2);
    expect(df.column("x")).toEqual([1, 2]);
  });

  it("creates from CSV", () => {
    const csv = "name,age\nAlice,30\nBob,25";
    const df = DataFrame.fromCSV(csv);
    expect(df.nRows).toBe(2);
    expect(df.column("name")).toEqual(["Alice", "Bob"]);
    expect(df.column("age")).toEqual([30, 25]); // auto-typed
  });

  it("creates from JSON", () => {
    const json = '[{"x": 1, "y": true}, {"x": 2, "y": false}]';
    const df = DataFrame.fromJSON(json);
    expect(df.nRows).toBe(2);
    expect(df.column("y")).toEqual([true, false]);
  });

  it("throws for mismatched column lengths", () => {
    expect(() =>
      DataFrame.create({ a: [1, 2], b: [1] }),
    ).toThrow("rows");
  });
});

// ── Column access ─────────────────────────────────────────────────────────

describe("column access", () => {
  const df = DataFrame.create({ x: [1, 2, 3], y: ["a", "b", "c"] });

  it("column returns a copy", () => {
    const col = df.column("x");
    col[0] = 999;
    expect(df.column("x")[0]).toBe(1);
  });

  it("numericColumn returns number[]", () => {
    const nums = df.numericColumn("x");
    expect(nums).toEqual([1, 2, 3]);
  });

  it("numericColumn throws for non-numeric", () => {
    expect(() => df.numericColumn("y")).toThrow("not a number");
  });

  it("column throws for unknown name", () => {
    expect(() => df.column("z")).toThrow("not found");
  });
});

// ── Selection & Filtering ─────────────────────────────────────────────────

describe("selection and filtering", () => {
  const df = DataFrame.create({
    name: ["Alice", "Bob", "Carol"],
    age: [30, 25, 35],
    city: ["NYC", "LA", "NYC"],
  });

  it("select picks columns", () => {
    const sub = df.select("name", "age");
    expect(sub.nCols).toBe(2);
    expect(sub.columnNames).toEqual(["name", "age"]);
  });

  it("filter rows by predicate", () => {
    const nyc = df.filter((row) => row.city === "NYC");
    expect(nyc.nRows).toBe(2);
    expect(nyc.column("name")).toEqual(["Alice", "Carol"]);
  });

  it("sort by column", () => {
    const sorted = df.sort("age");
    expect(sorted.column("name")).toEqual(["Bob", "Alice", "Carol"]);
  });

  it("sort descending", () => {
    const sorted = df.sort("age", false);
    expect(sorted.column("name")).toEqual(["Carol", "Alice", "Bob"]);
  });

  it("head returns first n rows", () => {
    expect(df.head(2).nRows).toBe(2);
    expect(df.head(2).column("name")).toEqual(["Alice", "Bob"]);
  });

  it("tail returns last n rows", () => {
    expect(df.tail(1).column("name")).toEqual(["Carol"]);
  });

  it("slice returns range", () => {
    const sliced = df.slice(1, 3);
    expect(sliced.nRows).toBe(2);
    expect(sliced.column("name")).toEqual(["Bob", "Carol"]);
  });
});

// ── Mutation ──────────────────────────────────────────────────────────────

describe("mutation", () => {
  const df = DataFrame.create({ x: [1, 2, 3] });

  it("addColumn adds a new column", () => {
    const df2 = df.addColumn("y", [10, 20, 30]);
    expect(df2.nCols).toBe(2);
    expect(df2.column("y")).toEqual([10, 20, 30]);
  });

  it("mutate creates column from function", () => {
    const df2 = df.mutate("x2", (row) => (row.x as number) * 2);
    expect(df2.column("x2")).toEqual([2, 4, 6]);
  });

  it("dropColumn removes a column", () => {
    const df2 = df.addColumn("y", [4, 5, 6]).dropColumn("x");
    expect(df2.columnNames).toEqual(["y"]);
  });

  it("renameColumn renames", () => {
    const df2 = df.renameColumn("x", "value");
    expect(df2.columnNames).toEqual(["value"]);
    expect(df2.column("value")).toEqual([1, 2, 3]);
  });

  it("addColumn throws for wrong length", () => {
    expect(() => df.addColumn("y", [1, 2])).toThrow("doesn't match");
  });
});

// ── Group-by ──────────────────────────────────────────────────────────────

describe("groupBy", () => {
  const df = DataFrame.create({
    group: ["A", "A", "B", "B", "B"],
    value: [10, 20, 30, 40, 50],
  });

  it("computes sum per group", () => {
    const result = df.groupBy(["group"], {
      total: { column: "value", fn: "sum" },
    });
    const records = result.toRecords();
    const a = records.find((r) => r.group === "A");
    const b = records.find((r) => r.group === "B");
    expect(a!.total).toBe(30);
    expect(b!.total).toBe(120);
  });

  it("computes mean per group", () => {
    const result = df.groupBy(["group"], {
      avg: { column: "value", fn: "mean" },
    });
    const records = result.toRecords();
    const a = records.find((r) => r.group === "A");
    expect(a!.avg).toBe(15);
  });

  it("computes count per group", () => {
    const result = df.groupBy(["group"], {
      n: { column: "value", fn: "count" },
    });
    const records = result.toRecords();
    const b = records.find((r) => r.group === "B");
    expect(b!.n).toBe(3);
  });

  it("supports custom aggregation function", () => {
    const result = df.groupBy(["group"], {
      range: {
        column: "value",
        fn: (vals) => {
          const nums = vals as number[];
          return Math.max(...nums) - Math.min(...nums);
        },
      },
    });
    const records = result.toRecords();
    const b = records.find((r) => r.group === "B");
    expect(b!.range).toBe(20);
  });
});

// ── Joins ─────────────────────────────────────────────────────────────────

describe("join", () => {
  const left = DataFrame.create({
    id: [1, 2, 3],
    name: ["Alice", "Bob", "Carol"],
  });
  const right = DataFrame.create({
    id: [2, 3, 4],
    score: [85, 92, 78],
  });

  it("inner join keeps matching rows", () => {
    const result = left.join(right, "id", "inner");
    expect(result.nRows).toBe(2);
    expect(result.column("id")).toEqual([2, 3]);
    expect(result.column("name")).toEqual(["Bob", "Carol"]);
    expect(result.column("score")).toEqual([85, 92]);
  });

  it("left join keeps all left rows", () => {
    const result = left.join(right, "id", "left");
    expect(result.nRows).toBe(3);
    expect(result.column("id")).toEqual([1, 2, 3]);
    expect(result.column("score")).toEqual([null, 85, 92]);
  });

  it("right join keeps all right rows", () => {
    const result = left.join(right, "id", "right");
    expect(result.nRows).toBe(3);
    expect(result.column("score")).toEqual([85, 92, 78]);
  });

  it("outer join keeps all rows", () => {
    const result = left.join(right, "id", "outer");
    expect(result.nRows).toBe(4); // ids 1,2,3,4
  });

  it("handles column name collisions", () => {
    const r2 = DataFrame.create({ id: [1], name: ["X"] });
    const result = left.join(r2, "id", "inner");
    expect(result.columnNames).toContain("name");
    expect(result.columnNames).toContain("name_right");
  });
});

// ── I/O ───────────────────────────────────────────────────────────────────

describe("I/O", () => {
  it("roundtrips through CSV", () => {
    const df = DataFrame.create({ x: [1, 2], y: ["a", "b"] });
    const csv = df.toCSV();
    const df2 = DataFrame.fromCSV(csv);
    expect(df2.column("x")).toEqual([1, 2]);
    expect(df2.column("y")).toEqual(["a", "b"]);
  });

  it("roundtrips through JSON", () => {
    const df = DataFrame.create({ x: [1, 2], y: [true, false] });
    const json = df.toJSON();
    const df2 = DataFrame.fromJSON(json);
    expect(df2.column("x")).toEqual([1, 2]);
    expect(df2.column("y")).toEqual([true, false]);
  });

  it("handles CSV with quoted fields", () => {
    const csv = 'name,note\n"Smith, John","has ""quotes"""\nJane,simple';
    const df = DataFrame.fromCSV(csv);
    expect(df.column("name")).toEqual(["Smith, John", "Jane"]);
    expect(df.column("note")).toEqual(['has "quotes"', "simple"]);
  });

  it("describe returns shape info", () => {
    const df = DataFrame.create({ a: [1], b: [2] });
    const info = df.describe();
    expect(info.nRows).toBe(1);
    expect(info.nCols).toBe(2);
    expect(info.columns).toEqual(["a", "b"]);
  });

  it("auto-types CSV values", () => {
    const csv = "a,b,c\n1,true,hello\n2,false,null";
    const df = DataFrame.fromCSV(csv);
    expect(df.column("a")).toEqual([1, 2]);
    expect(df.column("b")).toEqual([true, false]);
    expect(df.column("c")).toEqual(["hello", null]);
  });
});
