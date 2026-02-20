"""JSON vs TOON: API latency, token, and cost comparison across formats."""

import json
import time
from statistics import stdev

import dotenv
import tiktoken
from openai import OpenAI
from pydantic import BaseModel
from scipy.stats import ttest_ind

dotenv.load_dotenv()

MODEL   = "gpt-5.2"
RUNS    = 5
PRICING = {"gpt-5.2": {"input": 1.75, "output": 14.0}}  # $/1M tokens
SYSTEM  = "Answer the question using only the provided context. Be concise."
QUESTION = "Explain the fast inverse square root and cite the relevant doc ids."

client = OpenAI()


# --- Schema ---

class DocSchema(BaseModel):
    id: str
    title: str
    source: str
    span: str
    score: float
    summary: str


# --- Sample data ---

DOCS = [
    DocSchema(id="A1", title="Fast Inverse Square Root (1999)",
              source="https://en.wikipedia.org/wiki/Fast_inverse_square_root",
              span="sec:impl", score=0.92,
              summary="Quake III's 1/sqrt(x) bit hack: magic constant 0x5f3759df."),
    DocSchema(id="B7", title="Bit-level Floating Point Tricks",
              source="https://example.com/bit-tricks",
              span="p3", score=0.85,
              summary="Casting ints to floats, Newton-Raphson refinement."),
    DocSchema(id="C3", title="Numerical Methods in C (Press et al.)",
              source="https://example.com/numerical-methods",
              span="ch5", score=0.80,
              summary="Comprehensive guide to numerical algorithms including sqrt approximations."),
    DocSchema(id="D2", title="IEEE 754 Floating-Point Standard",
              source="https://en.wikipedia.org/wiki/IEEE_754",
              span="sec:binary32", score=0.78,
              summary="Binary32 format layout: sign, exponent, mantissa bit fields."),
    DocSchema(id="E9", title="Game Engine Architecture (Gregory)",
              source="https://example.com/game-engine-arch",
              span="ch6.3", score=0.74,
              summary="Performance tricks used in AAA game engines, including SIMD and bit hacks."),
    DocSchema(id="F5", title="Hacker's Delight (Warren)",
              source="https://example.com/hackers-delight",
              span="ch11", score=0.71,
              summary="Low-level bit manipulation tricks, integer approximations of float ops."),
    DocSchema(id="G4", title="Newton-Raphson Method — MathWorld",
              source="https://mathworld.wolfram.com/Newton-RaphsonMethod.html",
              span="eq:3", score=0.68,
              summary="Iterative root-finding method used as refinement step in fast inverse sqrt."),
    DocSchema(id="H6", title="Quake III Arena Source Code",
              source="https://github.com/id-Software/Quake-III-Arena",
              span="q_math.c:561", score=0.66,
              summary="Original C implementation of Q_rsqrt with the magic constant and one NR step."),
]

DOCS_DICT = [d.model_dump() for d in DOCS]


# --- Helpers ---

def count_tokens(text: str) -> int:
    try:
        enc = tiktoken.encoding_for_model(MODEL)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text or ""))

def estimate_cost(tok_in: int, tok_out: int) -> float:
    r = PRICING.get(MODEL, {"input": 0.0, "output": 0.0})
    return (tok_in / 1e6) * r["input"] + (tok_out / 1e6) * r["output"]

def call_api(context: str) -> dict:
    prompt = f"Context:\n{context}\n\nQuestion: {QUESTION}"
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": prompt},
        ],
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    tok_in  = resp.usage.prompt_tokens
    tok_out = resp.usage.completion_tokens
    return {
        "latency_ms": round(latency_ms, 2),
        "tok_in":     tok_in,
        "tok_out":    tok_out,
        "cost_usd":   round(estimate_cost(tok_in, tok_out), 6),
    }

def run_and_report(label: str, context: str, baseline_cost: float) -> list[dict]:
    ctx_bytes  = len(context.encode())
    ctx_tokens = count_tokens(context)
    print(f"\n{label} — bytes: {ctx_bytes}  context_tokens: {ctx_tokens}")

    runs = []
    for i in range(RUNS):
        r = call_api(context)
        runs.append(r)
        print(f"  run {i+1}: latency_ms={r['latency_ms']:>8}  "
              f"tok_in={r['tok_in']:>4}  tok_out={r['tok_out']:>4}  cost_usd={r['cost_usd']}")

    lats = [r["latency_ms"] for r in runs]
    costs = [r["cost_usd"] for r in runs]
    mean_lat  = round(sum(lats) / RUNS, 2)
    mean_cost = round(sum(costs) / RUNS, 6)
    print(f"  → mean/std/min/max latency (ms): "
          f"{mean_lat} / {round(stdev(lats), 2)} / {round(min(lats), 2)} / {round(max(lats), 2)}")
    print(f"  → mean cost_usd: {mean_cost}  ", end="")
    if label != "JSON":
        print(      f"cost_ratio vs JSON: {round(mean_cost / max(baseline_cost, 1e-9), 3)}", end="")
    print()
    return runs


# --- Context serializers ---

def json_context() -> str:
    return json.dumps(DOCS_DICT, separators=(",", ":"))

def toon_context() -> str:
    import toon
    return toon.encode(DOCS_DICT)

def toons_context() -> str:
    import toons
    return toons.dumps(DOCS_DICT, indent=2, delimiter=",")

# --- Main ---

def main():
    # JSON baseline
    json_ctx  = json_context()
    json_runs = run_and_report("JSON", json_ctx, baseline_cost=1.0)
    json_mean_lat  = sum(r["latency_ms"] for r in json_runs) / RUNS
    json_mean_cost = sum(r["cost_usd"]   for r in json_runs) / RUNS
    json_tok_in    = sum(r["tok_in"]     for r in json_runs) / RUNS
    json_tok_out   = sum(r["tok_out"]    for r in json_runs) / RUNS
    json_bytes     = len(json_ctx.encode())

    # TOON libraries
    toon_results = {}
    for label, ctx_fn in [("toon (toonify)", toon_context), ("toons", toons_context)]:
        try:
            ctx = ctx_fn()
            runs = run_and_report(label, ctx, baseline_cost=json_mean_cost)
            toon_results[label] = {"runs": runs, "bytes": len(ctx.encode())}
        except Exception as exc:
            print(f"\n{label}: error — {exc}")

    # --- Summary ---
    print("\n" + "=" * 80)
    print(f"{'SUMMARY':^80}")
    print("=" * 80)

    json_lats  = [r["latency_ms"] for r in json_runs]
    json_costs = [r["cost_usd"]   for r in json_runs]

    for label, data in toon_results.items():
        runs  = data["runs"]
        lats  = [r["latency_ms"] for r in runs]
        costs = [r["cost_usd"]   for r in runs]

        mean_lat  = sum(lats)  / RUNS
        mean_cost = sum(costs) / RUNS
        tok_in    = sum(r["tok_in"]  for r in runs) / RUNS
        tok_out   = sum(r["tok_out"] for r in runs) / RUNS

        # Welch's t-test (independent samples, unequal variances)
        t_lat,  p_lat  = ttest_ind(json_lats,  lats,  equal_var=False)
        t_cost, p_cost = ttest_ind(json_costs, costs, equal_var=False)

        def sig(p):
            if p < 0.001: return "*** (p<0.001)"
            if p < 0.01:  return "**  (p<0.01)"
            if p < 0.05:  return "*   (p<0.05)"
            return        "ns  (p≥0.05)"

        print(f"\n{label} vs JSON")
        print(f"  {'metric':<18}  {'JSON mean':>12}  {'TOON mean':>12}  {'ratio':>7}  {'t':>7}  {'p':>8}  significance")
        print(f"  {'-'*78}")
        print(f"  {'latency (ms)':<18}  {json_mean_lat:>12.2f}  {mean_lat:>12.2f}  "
              f"{mean_lat/max(json_mean_lat,1e-9):>7.3f}  {t_lat:>7.3f}  {p_lat:>8.4f}  {sig(p_lat)}")
        print(f"  {'cost_usd':<18}  {json_mean_cost:>12.6f}  {mean_cost:>12.6f}  "
              f"{mean_cost/max(json_mean_cost,1e-9):>7.3f}  {t_cost:>7.3f}  {p_cost:>8.4f}  {sig(p_cost)}")
        print(f"  {'tok_in':<18}  {json_tok_in:>12.1f}  {tok_in:>12.1f}  "
              f"{tok_in/max(json_tok_in,1e-9):>7.3f}")
        print(f"  {'tok_out':<18}  {json_tok_out:>12.1f}  {tok_out:>12.1f}  "
              f"{tok_out/max(json_tok_out,1e-9):>7.3f}")
        print(f"  {'context bytes':<18}  {json_bytes:>12}  {data['bytes']:>12}  "
              f"{data['bytes']/max(json_bytes,1e-9):>7.3f}")

    print("\n" + "=" * 80)
    print("  Welch's t-test (two-sided, unequal variances). *p<0.05  **p<0.01  ***p<0.001")
    print("=" * 80)

if __name__ == "__main__":
    main()
