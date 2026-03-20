import {
  gammaLn,
  gamma,
  factorial,
  logFactorial,
  binomialCoeff,
  betaFn,
  erf,
  erfc,
  regularizedGammaP,
  regularizedBeta,
} from "../../src/utils/math";

describe("gammaLn / gamma", () => {
  it("gamma(1) = 1", () => {
    expect(gamma(1)).toBeCloseTo(1, 10);
  });

  it("gamma(0.5) = sqrt(pi)", () => {
    expect(gamma(0.5)).toBeCloseTo(Math.sqrt(Math.PI), 10);
  });

  it("gamma(5) = 4! = 24", () => {
    expect(gamma(5)).toBeCloseTo(24, 8);
  });

  it("gammaLn(10) matches log(gamma(10))", () => {
    expect(gammaLn(10)).toBeCloseTo(Math.log(gamma(10)), 8);
  });

  it("throws for non-positive integers", () => {
    expect(() => gammaLn(0)).toThrow();
    expect(() => gammaLn(-1)).toThrow();
  });
});

describe("factorial / logFactorial", () => {
  it("0! = 1", () => expect(factorial(0)).toBeCloseTo(1, 10));
  it("5! = 120", () => expect(factorial(5)).toBeCloseTo(120));
  it("10! = 3628800", () => expect(factorial(10)).toBeCloseTo(3628800));
  it("logFactorial(10) = ln(3628800)", () => {
    expect(logFactorial(10)).toBeCloseTo(Math.log(3628800), 8);
  });
  it("factorial(171) = Infinity", () => expect(factorial(171)).toBe(Infinity));
});

describe("binomialCoeff", () => {
  it("C(5,2) = 10", () => expect(binomialCoeff(5, 2)).toBeCloseTo(10));
  it("C(10,0) = 1", () => expect(binomialCoeff(10, 0)).toBeCloseTo(1));
  it("C(10,10) = 1", () => expect(binomialCoeff(10, 10)).toBeCloseTo(1));
  it("C(5,6) = 0", () => expect(binomialCoeff(5, 6)).toBe(0));
});

describe("betaFn", () => {
  it("B(1,1) = 1", () => expect(betaFn(1, 1)).toBeCloseTo(1, 8));
  it("B(2,3) = 1/12", () => expect(betaFn(2, 3)).toBeCloseTo(1 / 12, 8));
  it("B(0.5, 0.5) = pi", () => expect(betaFn(0.5, 0.5)).toBeCloseTo(Math.PI, 6));
});

describe("erf / erfc", () => {
  it("erf(0) = 0", () => expect(erf(0)).toBeCloseTo(0, 8));
  it("erf(Infinity) ~ 1", () => expect(erf(100)).toBeCloseTo(1, 10));
  it("erf(1) ~ 0.8427", () => expect(erf(1)).toBeCloseTo(0.8427, 3));
  it("erf(-x) = -erf(x)", () => expect(erf(-1)).toBeCloseTo(-erf(1), 8));
  it("erfc(x) = 1 - erf(x)", () => expect(erfc(1)).toBeCloseTo(1 - erf(1), 10));
});

describe("regularizedGammaP", () => {
  it("P(1, 1) = 1 - e^-1", () => {
    expect(regularizedGammaP(1, 1)).toBeCloseTo(1 - Math.exp(-1), 8);
  });
  it("P(2, 3) ~ 0.8009", () => {
    expect(regularizedGammaP(2, 3)).toBeCloseTo(0.8009, 3);
  });
  it("P(s, 0) = 0", () => {
    expect(regularizedGammaP(5, 0)).toBe(0);
  });
});

describe("regularizedBeta", () => {
  it("I_0(a,b) = 0", () => expect(regularizedBeta(0, 2, 3)).toBe(0));
  it("I_1(a,b) = 1", () => expect(regularizedBeta(1, 2, 3)).toBe(1));
  it("I_0.5(1,1) = 0.5", () => {
    expect(regularizedBeta(0.5, 1, 1)).toBeCloseTo(0.5, 8);
  });
  it("I_0.5(2,5) ~ 0.8906", () => {
    expect(regularizedBeta(0.5, 2, 5)).toBeCloseTo(0.8906, 3);
  });
});
