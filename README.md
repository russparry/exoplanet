Goal (in one line)

Use open exoplanet catalogs + Kepler/TESS light curves to reproduce classic plots, detect/characterize transits, engineer physical features (radius, insolation, equilibrium temperature), and build simple predictive models for planet type / habitability — then present polished visualizations and an AI-assisted narrative and artwork.

Big picture workflow

Pick datasets (catalog + light curves)

Download data (APIs or Python libraries)

Explore & visualize basic relationships from the catalog

Light-curve analysis: detrend → search for transits → measure transit parameters

Feature engineering: compute physical quantities (insolation, equilibrium temp, radius if not given)

Modeling: build classifiers/regressors (e.g., rocky vs gaseous, habitability score)

Visualizations & storytelling: scientific plots + AI renders + plain-English explanations

Deliverables: report, code notebook, poster/slides, planetarium-ready images

1) Datasets — what to use (beginner friendly)

NASA Exoplanet Archive — catalog of confirmed planets and candidate parameters (period, radius, mass, insolation, discovery method).

TESS / Kepler light curves (from MAST) — time-series brightness data you’ll analyze for transits.

(Optional) Stellar parameters from Gaia or the Exoplanet Archive (host star radius, temperature, luminosity) — these are needed for habitability calculations.

Why both catalog + light curves?

Catalog gives labeled examples and physical parameters.

Light curves let you reproduce discovery steps (detect transits) and measure transit depth/shape yourself.

2) Tools & libraries you’ll use (Python)

astroquery — query NASA archives.

lightkurve — download and analyze Kepler/TESS light curves (great for beginners).

numpy, pandas — numeric + table work.

matplotlib or plotly — plots (plotly for interactive).

astropy (timeseries, constants) — astronomy utilities.

scikit-learn — model building (random forest, logistic regression, pipelines).

tensorflow/keras or pytorch — only if you want deep learning (optional).

seaborn — nicer statistical plots (optional).

exoplanet/batman — tools to model transit light curves precisely (optional, more advanced).

3) Data access (concrete beginner code)

Below are short code snippets showing how you’d fetch data. You can run these in a Jupyter notebook.

A. Query NASA Exoplanet Archive with astroquery

from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
import pandas as pd

# Get confirmed planets table (a subset of columns)
table = NasaExoplanetArchive.query_criteria(table="ps",
    select="pl_name, hostname, pl_orbper, pl_rade, pl_bmassj, st_rad, st_teff, st_lum",
    where="pl_rade IS NOT NULL")
df = table.to_pandas()
df.head()


B. Download a TESS light curve with lightkurve

import lightkurve as lk

# Search for light curves of a known target, e.g., TOI-700
search_result = lk.search_lightcurve('TOI 700', author='SPOC')
lc = search_result.download().remove_nans()
lc.plot()

4) Exploratory visualizations (what to make and why)

Make these early — they help you and your audience understand the data.

Radius vs orbital period scatter — shows populations (small/short-period vs large/long-period).

Mass-Radius plot — visual separation between rocky and gaseous planets.

Histogram of planet radii — look for the radius gap (~1.5–2 R⊕).

Habitable zone diagram — host star luminosity vs semi-major axis showing where liquid water could exist.

Light curve with transit marked — show raw flux, detrended flux, and phased/folded transit.

Example plotting code (radius vs period):

import matplotlib.pyplot as plt
plt.scatter(df['pl_orbper'], df['pl_rade'], s=5, alpha=0.6)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Orbital Period (days)')
plt.ylabel('Planet Radius (R_earth)')
plt.title('Radius vs Period')
plt.show()

5) Light-curve analysis — beginner level steps

Objective: find an obvious transit and measure its depth and period.

Inspect raw light curve (plot flux vs time).

Remove bad cadences (quality flags, NaNs).

Detrend long-term variability (use lightkurve's flatten() which fits and removes trends).

Search for periodic dips using Box Least Squares (BLS) — lightkurve has BoxLeastSquaresPeriodogram.

Fold the light curve on the detected period to stack transits and measure transit depth and duration.

Estimate planet radius from transit depth:

𝛿
=
(
𝑅
𝑝
𝑅
⋆
)
2
⇒
𝑅
𝑝
=
𝑅
⋆
𝛿
δ=(
R
⋆
	​

R
p
	​

	​

)
2
⇒R
p
	​

=R
⋆
	​

δ
	​


where 
𝛿
δ is fractional depth and 
𝑅
⋆
R
⋆
	​

 is stellar radius.

Quick lightkurve example:

from lightkurve import search_lightcurvefile
from astropy.timeseries import BoxLeastSquares

lc = lk.search_lightcurve('TOI 700', author='SPOC').download().remove_nans().flatten()
time = lc.time.value
flux = lc.flux.value / np.median(lc.flux.value) - 1  # relative flux

model = BoxLeastSquares(time, flux)
periods = np.linspace(0.5, 50, 10000)  # search range in days
bls = model.power(periods, 0.2)         # transit duration guess: 0.2 days
best = bls.period[np.argmax(bls.power)]
print("Best period:", best)

lc.fold(period=best).plot()

6) Feature engineering — what to compute and why

These are the features your model can use to classify or predict habitability.

Orbital period (days) — from catalog or BLS.

Transit depth (fraction) — from folded light curve → relates to Rp/R*.

Transit duration (hours) — geometry info.

Planet radius (R⊕) — from depth & stellar radius (or catalog).

Insolation flux (S/S⊕) — how much starlight the planet receives:

𝑆
=
𝐿
⋆
4
𝜋
𝑎
2
(often normalized to Earth)
S=
4πa
2
L
⋆
	​

	​

(often normalized to Earth)

If you have semi-major axis a or use approximate relation from period + stellar mass via Kepler’s 3rd law.

Equilibrium temperature (T_eq) — rough habitability proxy:

𝑇
𝑒
𝑞
=
𝑇
⋆
𝑅
⋆
2
𝑎
(
1
−
𝐴
)
1
/
4
T
eq
	​

=T
⋆
	​

2a
R
⋆
	​

	​

	​

(1−A)
1/4

(use albedo 
𝐴
A ≈ 0.3 as a simple assumption).

Stellar effective temperature and radius — required for many calculations.

SNR of detection, data quality flags — how trustworthy is the detection?

Explain in plain terms: these features try to capture how big the planet is, how close it is to its star, how bright the star is, and how reliable our measurements are. Those factors strongly influence whether the planet is rocky and if it could hold liquid water.

7) Modeling — simple, interpretable options

You don’t need deep learning to get impressive results. Start with classical models.

Labels you can predict:

Rocky vs Gaseous (binary) — label from catalog (e.g., radius threshold: <1.6 R⊕ often rocky).

Habitability class — e.g., “likely in habitable zone” vs “not”.

Recommended pipeline:

Split data into train/test.

Standardize numeric features.

Try Random Forest and Logistic Regression.

Evaluate with accuracy, precision/recall, ROC AUC.

Use feature importance to explain model.

Sample scikit-learn snippet:

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

X = df[features]
y = df['is_rocky']  # you create this from radius threshold
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))


Explainability: show feature importances (which features drove the model). Use SHAP or permutation importance for clearer explanations.

8) Generative AI — responsible uses

Generative AI can add narrative and visual polish — but be explicit where it’s used.

Good uses:

Generate high-quality planet visualizations (artistic impressions) from parameters (radius, temperature, star color). Use image generation to make posters.

Write plain-English summaries of findings for non-technical viewers (e.g., “This planet receives 0.8× Earth’s sunlight…”). Ask the AI to produce a few different tone options.

Draft figure captions and a presentation script.

Important: don’t use AI to fabricate scientific claims. If AI suggests a new detection or parameter, verify with your data and cite your methods. Label any AI-generated art/text clearly.

9) Visuals that impress (and what to show)

Interactive dashboard (Plotly Dash / Streamlit): allow viewers to filter by radius/period and see updated plots. Great for a planetarium display.

High-res poster: left side — data & plots; right side — AI-generated artist’s impression + plain English text boxes.

Animated folding of light curve (time → phase) showing the transit emergent as you fold.

Model interpretability figure: feature importance bar chart with short explanations.

10) Project deliverables (what to hand in)

Jupyter notebook(s) with code + outputs (clean and well commented).

PDF report (3–8 pages) describing background, methods, results, interpretation, and reflection as “personal expression.” Bold/annotate where you did the creative/AI parts as your instructor asked.

High-resolution images (PNG/TIFF) for planetarium or poster display.

Optional: a short video narration (2–4 minutes) of your findings with visuals.

11) Beginner pitfalls & how to avoid them

Pitfall: Trying to process the entire Gaia/Gaia-scale data.
Fix: work on a well-chosen subset (e.g., confirmed planets + a sample of candidates or a single TESS sector).

Pitfall: Trusting every catalog value blindly.
Fix: check for NaN values, unrealistic entries, and outliers; add data quality filters.

Pitfall: Letting AI write results without verification.
Fix: verify any numeric claim with your calculations and include code that reproduces numbers.

12) Example mini outline you can hand to your instructor (copy/paste)

Title: Exoplanet Characterization and Habitability Modeling with TESS and NASA Exoplanet Archive
Summary: I will query the NASA Exoplanet Archive for confirmed planets and download TESS/Kepler light curves for selected targets. I will reproduce standard exoplanet plots (radius vs period, mass–radius) and perform light-curve transit detection and characterization using Box Least Squares. From that I will compute planetary radii and insolation and build a simple classifier to separate rocky from gaseous planets and a habitability indicator. I will include AI-generated artistic impressions and plain-English explanations to make the project accessible to a general audience. Deliverables: code notebooks, report, posters/images, and a short narrated visualization suitable for planetarium display.
Turn-in date: December 10, 2025.

13) Ready-to-run starter checklist (what to do next, minimal)

Pick a concrete dataset target: TESS (one sector + a few TOIs) or Kepler (one quarter).

Install Python + libraries: pip install lightkurve astroquery numpy pandas matplotlib scikit-learn shap plotly.

Run the sample data download snippets above to make sure you can fetch data.

Make the first plot: radius vs period from the Exoplanet Archive.

Choose one target light curve and run the BLS example to detect a transit.

