# Predictive Autoscaling with Course Optimization Methods — slide outline

## 1. Problem and questions
- Reactive autoscaling reacts after load crosses a threshold, but instances take ~20 min to be useful.
- Predictive autoscaling forecasts load and provisions ahead (AWS, Azure and GCP all offer it).
- RQ1: which optimizer fits the forecast model best, and how does that depend on cond(H)?
- RQ2: what is the optimal instance allocation under a budget (KKT)?
- RQ3: does predictive beat reactive, for periodic vs bursty workloads?

## 2. Data
- Azure Functions 2019 trace: 14 days of per-minute invocations (Shahrad et al., ATC'20).
- 623 busy apps → 565 active all 14 days → 31 periodic, 17 bursty → top 5 of each.
- Visual: `results/part1_forecast.png` (P1 smooth daily cycle vs B1 isolated bursts).

## 3. Model and convexity (Day 4)
- Poisson GLM, 10 features (lags 2–4, 1 day, 2 days, time of day); lead time 20 min.
- f(w) = mean(e^{Xw} − y·Xw) + λ/2‖w‖²; Hessian XᵀDiag(μ)X/n + λI ⪰ 0 ⇒ convex.
- Key remark: the loss is **not globally L-smooth**, because curvature grows with e^{Xw}.
- Accuracy: periodic WAPE 2–5% vs naive 3–16%; bursty WAPE up to 194%.

## 4. RQ1: seven optimizers (Days 7–14)
- Visual: `results/part1_convergence.png`.
- Newton: 2–9 iterations, < 1.1 ms. NCG and BFGS: 8–146. GD: up to 5000+.
- Iterations of first-order methods track log cond(H) (r = 0.91–0.99).
- Pitfall: α = 1/λmax(H(w₀)) **diverged on B2, B5** (λmax(H*) is 3–5× larger); line-search methods never failed.
- Fixed β = 0.9 momentum is 10× slower than GD when κ = 3.

## 5. RQ2: budgeted allocation and KKT (Days 5, 14–15)
- min Σ E[(D_t − c n_t)⁺] s.t. pΣn_t = B, n_t ≥ n_min (convex, Slater holds).
- KKT: c(1 − F_t(c n_t)) = λp ⇒ c n_t = μ_t + σ_t z with the **same z for every interval**.
- Interpretation: equal marginal overload reduction per dollar; λ is the shadow price of the budget.
- Bisection on λ matches SLSQP (gap ≤ 1.3e-8) and is 10–300× faster; SLSQP flagged failure on 5/20 instances.
- Visual: `results/part2_allocation.png`; the optimal allocation gives 20–70× less expected overload than uniform.

## 6. RQ3: replay on held-out days
- Visual: `results/part3_frontiers.png`.
- Periodic: predictive saves **~20% cost at equal overload**, stable for K = 5–50.
- Bursty: no reliable edge. It wins only where reactive misses ~90% of requests, and costs 38–88% more after level shifts (B2, B5).
- Hybrid max(predictive, reactive u = 0.9) caps the bursty worst case at +4–11%.

## 7. Discussion and production design
- Failure modes: bursts shorter than the lead time, regime shifts, Gaussian tails.
- Production design: CloudWatch → scheduled Lambda forecaster → KKT optimizer → EC2 scheduled actions (MinSize), plus target tracking as a safety net. This mirrors AWS Predictive Scaling's "max of policies" rule.
- Limitations: assumed capacity c, Gaussian approximation, 10-min bins, 2019 data.

## 8. Conclusions
- Newton is the right tool for small convex GLMs; fixed steps need a curvature bound the Poisson loss lacks.
- KKT turns budgeted provisioning into one-dimensional root finding: same quantile everywhere.
- Predictive scaling pays off for periodic workloads (~20%) and needs a reactive safety net for bursty ones.
