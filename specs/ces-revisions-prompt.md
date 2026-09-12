## ROLE AND GOAL

You are assisting a senior labor-market statistician (25+ yrs, ex-BLS) who builds Bayesian
state-space nowcasts of U.S. nonfarm payrolls. Produce (A) a critical literature review, (B) a
gap analysis, and (C) a concrete Bayesian research design. Assume the reader knows CES
methodology, seasonal adjustment, and Bayesian hierarchical modeling. Do not explain basics.

## PART A — LITERATURE REVIEW

Find and synthesize published and working-paper work on how the MAGNITUDE (not just the sign or
mean bias) of CES payroll revisions has evolved over time. Cover all four revision stages
separately:

  1. first print -> second print (one-month revision)
  2. second -> third print (two-month revision, "final" sample-based estimate)
  3. third print -> annual benchmark (incl. the preliminary August/September benchmark for total
     nonfarm and the final comprehensive March benchmark)
  4. post-benchmark revisions and wedge-back/interpolation effects on intervening months

Decompose sources of revision wherever the literature does: (i) late sample reporting / collection
completeness, (ii) seasonal factor revision (concurrent vs. projected factors), (iii) net
birth-death model error, (iv) benchmark realignment to QCEW universe counts, (v) NAICS/sample
redesign and reweighting.

Structure the review around these five candidate drivers, and report for each what is known,
what is asserted without evidence, and what is contested:

  1. SEASONALITY. Effects of the 2003 move to concurrent seasonal adjustment; X-12 -> X-13ARIMA-SEATS;
     the 2003 shift to NAICS; treatment of moving holidays and 4- vs 5-week reference intervals;
     COVID-era seasonal factor instability and outlier/level-shift handling; the relative size of
     seasonal-factor revision vs. sample revision in the total one- and two-month revision. Look
     specifically for work decomposing SA revisions into NSA-sample and seasonal-factor components.

  2. COLLECTION INTERVAL. Days between the reference pay period (including the 12th) and first
     closing / release; how the calendar placement of the 12th and the release date shifts the
     number of collection days; first-closing, second-closing and third-closing collection rates
     and their secular decline; evidence linking collection interval or first-closing rate to
     one-month revision dispersion. Include BLS Office of Survey Methods Research work and any
     response-rate/nonresponse-bias literature for CES specifically.

  3. BLS FUNDING. Real annual appropriations for BLS; continuing resolutions, sequestration (2013),
     shutdowns (2013, 2018-19, and any since), and documented program cuts (sample reductions,
     suspended series, reduced collection). Also whether any published work connects appropriations
     or budget stress to measured data quality in CES or any federal statistical program. Note the
     partisan composition of the appropriating chambers only as descriptive context; flag explicitly
     that this is likely confounded with secular trend and is not credibly causal on its own.

  4. SAMPLE. CES sample size and design history: transition from quota/cutoff sample to a
     probability-based sample (late 1990s-2003), sample allocation across size classes and
     industries, attrition, the EDI/web/touchtone/fax collection-mode mix, and any documented
     relationship between sample size or composition and revision variance.

  5. STAFFING. BLS employment/FTE counts over time (OPM FedScope, BLS congressional budget
     justifications, Federal Employment Reports) and any literature relating statistical-agency
     staffing to output quality, timeliness, or revision behavior.

Also situate CES within the broader data-revision literature — news vs. noise (Mankiw-Shapiro),
real-time datasets (Croushore-Stark; ALFRED), Aruoba on revision properties, Jacobs-van Norden
state-space revision models, Kishor-Koenig real-time Kalman filtering, Fixler-Grimm on BEA
revisions, Faust-Rogers-Wright on cross-country GDP revisions — and say clearly which CES-specific
questions those frameworks have and have not been applied to.

## SOURCES TO PRIORITIZE

BLS Monthly Labor Review and BLS Working Papers (OSMR); CES technical notes and annual benchmark
articles; Journal of Official Statistics; Survey Methodology; JBES; Journal of Econometrics;
International Journal of Forecasting; NBER and FEDS working papers; regional Fed economic letters
and reviews (SF, Cleveland, Richmond, Atlanta, St. Louis); Brookings Papers; GAO and CRS reports on
federal statistical agency funding; American Statistical Association and COPAFS commentary on the
federal statistical system. Prefer primary agency documentation over secondary commentary. Include
grey literature (ADP, Chicago Fed, private nowcasters) but label it as such.

## PART B — GAP ANALYSIS

Deliver a table with one row per open question: question, closest existing work, why it falls short
(data, identification, vintage coverage, or estimand), what data would be needed, and feasibility
given public sources. Then give a short prose section on the two or three gaps most worth pursuing
and why. Be explicit where the answer is "no one has looked at this" versus "the literature exists
but is old / pre-NAICS / pre-COVID."

Also inventory data availability for each driver: series, vintage coverage, frequency, earliest
date, and access route. Note known constraints: BLS does not publicly archive QCEW vintages, and
ALFRED's QCEW coverage is limited to selected county/MSA series without industry detail. State
which covariates would need to be hand-built from budget justifications or archived releases.

## PART C — BAYESIAN RESEARCH DESIGN

Propose a concrete design, not a menu of options. Requirements:

 - The estimand is the SCALE of revisions, so the covariates must enter the variance (or the
   scale of a heavy-tailed measurement density), not only a conditional mean. Discuss stochastic
   volatility plus covariate effects on log-scale, and Student-t vs. Gaussian innovations.
 - Cast the multi-vintage structure as a state-space model with a latent "true" employment path and
   vintage-specific measurement equations, in the Jacobs-van Norden tradition, so news and noise
   components are separately identified across the four revision stages. State the identification
   assumptions this requires and what breaks them.
 - Handle the annual benchmark as an irregularly-timed, higher-precision observation of the latent
   state, with wedge-back distributing the benchmark error across intervening months.
 - Hierarchy: partial pooling of driver coefficients across NAICS supersectors, since birth-death
   and benchmark error are sector-heterogeneous. Discuss centered vs. non-centered parameterization.
 - Seasonality: a strategy for separating seasonal-factor revision from sample revision, e.g.
   modeling NSA and SA vintages jointly.
 - Structural change: how to handle 2003 (NAICS + concurrent SA + probability sample, all at once
   and therefore confounded), the 2020-2022 period, and any sample redesigns — change points vs.
   time-varying parameters with shrinkage (e.g. Bitto & Fruhwirth-Schnatter).
 - Identification and confounding: appropriations, staffing, and sample size are slow-moving,
   trending, and mutually collinear, while collection interval varies sharply within year. Say
   which parameters are actually identified off what variation, propose a regularized-horseshoe or
   similarly honest shrinkage prior, and be candid about what cannot be identified from ~40 years of
   monthly data. Treat party-of-the-appropriators as a descriptive grouping, not a causal slope,
   unless you can propose a defensible design.
 - Prior specification: give actual prior families and scales with justification on the natural
   units of the data (thousands of jobs), plus a prior predictive check plan.
 - Validation: posterior predictive checks targeted at revision dispersion, LOO/PSIS comparison
   against a constant-variance baseline, out-of-sample/pseudo-real-time checks, and MCMC diagnostics.
 - Implementation notes for Python/JAX: NumPyro (NUTS) for the full model, Dynamax for
   linear-Gaussian filtering where the state can be marginalized, Blackjax if a custom kernel is
   warranted, ArviZ for diagnostics, Polars for the vintage panel. Flag where marginalizing the
   latent state analytically will matter for sampling efficiency.
 - Close with what the model would let you say that a frequentist event-study or heteroskedastic
   regression could not.

## OUTPUT FORMAT

1) Executive summary (< 400 words).
2) Literature review organized by the five drivers, with the revision-source decomposition running
   through it.
3) Gap table + data availability inventory.
4) Bayesian design, written as a specification someone could implement.
5) Annotated bibliography, full citations, with a one-line note on each source's relevance and
   whether it is peer-reviewed, agency documentation, or grey literature.

Flag uncertainty explicitly. Do not pad with background the reader already knows. If something
cannot be found, say so rather than inferring it.
