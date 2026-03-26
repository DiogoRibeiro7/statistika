import { VonMises } from "../../../src/distributions/continuous/von-mises";

describe("VonMises distribution", () => {
  const vm = new VonMises(0, 2); // mu=0, kappa=2

  it("has correct name", () => {
    expect(vm.name).toBe("VonMises(0, 2)");
  });

  it("throws on negative kappa", () => {
    expect(() => new VonMises(0, -1)).toThrow();
  });

  it("allows kappa = 0 (uniform on circle)", () => {
    const vmUniform = new VonMises(0, 0);
    expect(vmUniform.name).toBe("VonMises(0, 0)");
  });

  it("mean equals mu", () => {
    expect(vm.mean()).toBe(0);
    const vm2 = new VonMises(1.5, 3);
    expect(vm2.mean()).toBe(1.5);
  });

  it("circular variance is 1 for kappa=0 (uniform)", () => {
    const vmUniform = new VonMises(0, 0);
    expect(vmUniform.variance()).toBe(1);
  });

  it("circular variance is between 0 and 1", () => {
    expect(vm.variance()).toBeGreaterThan(0);
    expect(vm.variance()).toBeLessThan(1);
  });

  it("circular variance decreases with increasing kappa", () => {
    const vm1 = new VonMises(0, 1);
    const vm5 = new VonMises(0, 5);
    const vm10 = new VonMises(0, 10);
    expect(vm1.variance()).toBeGreaterThan(vm5.variance());
    expect(vm5.variance()).toBeGreaterThan(vm10.variance());
  });

  it("pdf is maximum at mu", () => {
    const pdfAtMu = vm.pdf(0);
    expect(pdfAtMu).toBeGreaterThan(vm.pdf(0.5));
    expect(pdfAtMu).toBeGreaterThan(vm.pdf(-0.5));
    expect(pdfAtMu).toBeGreaterThan(vm.pdf(1));
  });

  it("pdf is symmetric around mu", () => {
    expect(vm.pdf(0.5)).toBeCloseTo(vm.pdf(-0.5), 10);
    expect(vm.pdf(1)).toBeCloseTo(vm.pdf(-1), 10);
  });

  it("pdf for kappa=0 is uniform 1/(2*pi)", () => {
    const vmUniform = new VonMises(0, 0);
    const expected = 1 / (2 * Math.PI);
    expect(vmUniform.pdf(0)).toBeCloseTo(expected, 10);
    expect(vmUniform.pdf(1)).toBeCloseTo(expected, 10);
    expect(vmUniform.pdf(-2)).toBeCloseTo(expected, 10);
  });

  it("pdf at known values", () => {
    // f(0; 0, 2) = exp(2*cos(0)) / (2*pi*I0(2))
    // = exp(2) / (2*pi*I0(2))
    // We test relative values instead of absolute to avoid I0 lookup
    const pdfAt0 = vm.pdf(0);
    const pdfAtPi = vm.pdf(Math.PI);
    // ratio = exp(kappa*cos(0)) / exp(kappa*cos(pi)) = exp(2)/exp(-2) = exp(4)
    expect(pdfAt0 / pdfAtPi).toBeCloseTo(Math.exp(4), 1);
  });

  it("pdf is always positive", () => {
    for (let x = -Math.PI; x <= Math.PI; x += 0.3) {
      expect(vm.pdf(x)).toBeGreaterThan(0);
    }
  });

  it("cdf is 0 at -pi", () => {
    expect(vm.cdf(-Math.PI)).toBe(0);
  });

  it("cdf is 1 at pi", () => {
    expect(vm.cdf(Math.PI)).toBe(1);
  });

  it("cdf is 0.5 at mu (by symmetry)", () => {
    expect(vm.cdf(0)).toBeCloseTo(0.5, 2);
  });

  it("cdf increases with x on [-pi, pi]", () => {
    expect(vm.cdf(-1)).toBeLessThan(vm.cdf(0));
    expect(vm.cdf(0)).toBeLessThan(vm.cdf(1));
  });

  it("cdf and quantile are inverses", () => {
    for (const p of [0.1, 0.25, 0.5, 0.75, 0.9]) {
      const q = vm.quantile(p);
      expect(vm.cdf(q)).toBeCloseTo(p, 1);
    }
  });

  it("quantile(0) = -pi", () => {
    expect(vm.quantile(0)).toBe(-Math.PI);
  });

  it("quantile(1) = pi", () => {
    expect(vm.quantile(1)).toBe(Math.PI);
  });

  it("quantile(0.5) is close to mu", () => {
    expect(vm.quantile(0.5)).toBeCloseTo(0, 1);
  });

  it("quantile throws on out-of-range p", () => {
    expect(() => vm.quantile(-0.1)).toThrow();
    expect(() => vm.quantile(1.1)).toThrow();
  });

  it("sample returns values in [-pi, pi]", () => {
    for (let i = 0; i < 100; i++) {
      const s = vm.sample();
      expect(s).toBeGreaterThanOrEqual(-Math.PI);
      expect(s).toBeLessThanOrEqual(Math.PI);
    }
  });

  it("sampleN returns correct number of samples", () => {
    expect(vm.sampleN(10).length).toBe(10);
  });

  it("sample for kappa=0 returns values in [-pi, pi]", () => {
    const vmUniform = new VonMises(0, 0);
    for (let i = 0; i < 50; i++) {
      const s = vmUniform.sample();
      expect(s).toBeGreaterThanOrEqual(-Math.PI);
      expect(s).toBeLessThanOrEqual(Math.PI);
    }
  });

  it("sample circular mean converges to mu", () => {
    const samples = vm.sampleN(5000);
    // Circular mean: atan2(mean(sin), mean(cos))
    let sinSum = 0;
    let cosSum = 0;
    for (const s of samples) {
      sinSum += Math.sin(s);
      cosSum += Math.cos(s);
    }
    const circularMean = Math.atan2(sinSum / samples.length, cosSum / samples.length);
    expect(circularMean).toBeCloseTo(0, 0);
  });

  it("works with non-standard parameters", () => {
    const vm2 = new VonMises(1, 5);
    expect(vm2.name).toBe("VonMises(1, 5)");
    expect(vm2.mean()).toBe(1);
    // PDF should peak at mu=1
    expect(vm2.pdf(1)).toBeGreaterThan(vm2.pdf(0));
    expect(vm2.pdf(1)).toBeGreaterThan(vm2.pdf(2));
  });

  it("high kappa gives narrow distribution", () => {
    const vmHigh = new VonMises(0, 100);
    // Almost all density should be near mu=0
    expect(vmHigh.pdf(0)).toBeGreaterThan(1);
    expect(vmHigh.pdf(1)).toBeLessThan(0.01);
    expect(vmHigh.variance()).toBeLessThan(0.02);
  });
});
