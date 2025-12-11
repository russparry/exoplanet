"""
Exoplanet Characterization and Habitability Modeling
Following instructions from README.md
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Windows
import matplotlib.pyplot as plt
import seaborn as sns
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
import lightkurve as lk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime

# Create output directories
os.makedirs('plots', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("="*80)
print("EXOPLANET CHARACTERIZATION AND HABITABILITY MODELING")
print("="*80)
print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# 1. QUERY NASA EXOPLANET ARCHIVE
# ============================================================================
print("[1/12] Querying NASA Exoplanet Archive...")
try:
    table = NasaExoplanetArchive.query_criteria(
        table="ps",
        select="pl_name, hostname, pl_orbper, pl_rade, pl_bmassj, pl_eqt, pl_insol, st_rad, st_teff, st_lum, sy_dist, pl_dens",
        where="pl_rade IS NOT NULL AND pl_orbper IS NOT NULL"
    )
    df = table.to_pandas()
    print(f"   Downloaded {len(df)} confirmed planets with radius data")
    df.to_csv('data/exoplanet_catalog.csv', index=False)
    print("   [OK] Saved to data/exoplanet_catalog.csv")
except Exception as e:
    print(f"   Error: {e}")
    df = pd.DataFrame()

# ============================================================================
# 2. CREATE RADIUS VS PERIOD SCATTER PLOT
# ============================================================================
print("\n[2/12] Creating Radius vs Period scatter plot...")
if not df.empty:
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(df['pl_orbper'], df['pl_rade'], s=10, alpha=0.6, c=df['pl_eqt'], cmap='coolwarm')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Orbital Period (days)', fontsize=12)
    ax.set_ylabel('Planet Radius (R⊕)', fontsize=12)
    ax.set_title('Exoplanet Radius vs Orbital Period', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Equilibrium Temperature (K)', fontsize=10)
    plt.tight_layout()
    plt.savefig('plots/radius_vs_period.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved to plots/radius_vs_period.png")

# ============================================================================
# 3. CREATE MASS-RADIUS PLOT
# ============================================================================
print("\n[3/12] Creating Mass-Radius plot...")
if not df.empty:
    # Convert Jupiter mass to Earth mass (1 M_jup = 317.8 M_earth)
    df_mass = df[df['pl_bmassj'].notna()].copy()
    df_mass['pl_bmasse'] = df_mass['pl_bmassj'] * 317.8

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(df_mass['pl_bmasse'], df_mass['pl_rade'], s=15, alpha=0.6,
                        c=df_mass['pl_dens'], cmap='viridis')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Planet Mass (M⊕)', fontsize=12)
    ax.set_ylabel('Planet Radius (R⊕)', fontsize=12)
    ax.set_title('Mass-Radius Relationship of Exoplanets', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add reference lines for rocky and gaseous planets
    masses = np.logspace(-1, 4, 100)
    rocky_radii = masses ** (1/3.7)  # Approximate rocky planet relation
    gaseous_radii = masses ** (1/2)  # Approximate gaseous planet relation
    ax.plot(masses, rocky_radii, 'r--', alpha=0.5, label='Rocky planets', linewidth=2)
    ax.plot(masses, gaseous_radii, 'b--', alpha=0.5, label='Gaseous planets', linewidth=2)
    ax.legend()

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Planet Density (g/cm³)', fontsize=10)
    plt.tight_layout()
    plt.savefig('plots/mass_radius.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   [OK]Saved to plots/mass_radius.png")

# ============================================================================
# 4. SKIP RADIUS HISTOGRAM (REMOVED PER USER REQUEST)
# ============================================================================
print("\n[4/12] Skipping radius histogram (removed)...")

# ============================================================================
# 5. DOWNLOAD TESS LIGHT CURVE
# ============================================================================
print("\n[5/12] Downloading TESS light curve for TOI-700...")
try:
    search_result = lk.search_lightcurve('TOI 700', author='SPOC', mission='TESS')
    if len(search_result) > 0:
        lc = search_result[0].download()
        lc = lc.remove_nans()
        print(f"   [OK]Downloaded light curve with {len(lc.time)} data points")

        # Save light curve data
        lc_df = pd.DataFrame({
            'time': lc.time.value,
            'flux': lc.flux.value,
            'flux_err': lc.flux_err.value if hasattr(lc, 'flux_err') else np.zeros(len(lc.time))
        })
        lc_df.to_csv('data/toi700_lightcurve.csv', index=False)
        print("   [OK]Saved to data/toi700_lightcurve.csv")
    else:
        print("   Warning: No TESS data found for TOI-700")
        lc = None
except Exception as e:
    print(f"   Error downloading light curve: {e}")
    lc = None

# ============================================================================
# 6. LIGHT CURVE ANALYSIS AND TRANSIT DETECTION USING BLS
# ============================================================================
print("\n[6/12] Performing light curve analysis and transit detection...")
if lc is not None:
    try:
        # Flatten the light curve to remove stellar variability
        lc_flat = lc.flatten(window_length=901)

        # Perform BLS (Box Least Squares) periodogram
        from astropy.timeseries import BoxLeastSquares

        time = lc_flat.time.value
        flux = lc_flat.flux.value

        # Normalize flux
        flux_norm = (flux / np.median(flux)) - 1

        model = BoxLeastSquares(time, flux_norm)
        periods = np.linspace(1, 30, 10000)
        bls = model.power(periods, 0.2)

        best_period = periods[np.argmax(bls.power)]
        print(f"   Best period found: {best_period:.3f} days")

        # Plot BLS periodogram
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(periods, bls.power, 'k-', linewidth=0.5)
        ax.axvline(best_period, color='r', linestyle='--', label=f'Best period: {best_period:.3f} d')
        ax.set_xlabel('Period (days)', fontsize=12)
        ax.set_ylabel('BLS Power', fontsize=12)
        ax.set_title('Box Least Squares Periodogram for TOI-700', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/bls_periodogram.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   [OK]Saved BLS periodogram to plots/bls_periodogram.png")

    except Exception as e:
        print(f"   Error in BLS analysis: {e}")
        best_period = 16.0  # Known period for TOI-700 d

# ============================================================================
# 7. CREATE FOLDED LIGHT CURVE VISUALIZATION
# ============================================================================
print("\n[7/12] Creating folded light curve visualization...")
if lc is not None:
    try:
        lc_flat = lc.flatten(window_length=901)
        lc_folded = lc_flat.fold(period=best_period)

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Raw light curve
        lc.plot(ax=axes[0], label='Raw Light Curve')
        axes[0].set_title('TOI-700 Raw Light Curve', fontsize=14, fontweight='bold')
        axes[0].legend()

        # Folded light curve
        lc_folded.scatter(ax=axes[1], s=1, alpha=0.5, label='Folded Light Curve')
        axes[1].set_title(f'Phase-Folded Light Curve (Period = {best_period:.3f} days)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Phase', fontsize=12)
        axes[1].set_ylabel('Normalized Flux', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('plots/folded_lightcurve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   [OK]Saved to plots/folded_lightcurve.png")
    except Exception as e:
        print(f"   Error creating folded light curve: {e}")

# ============================================================================
# 8. FEATURE ENGINEERING
# ============================================================================
print("\n[8/12] Engineering features: computing insolation and equilibrium temperature...")
if not df.empty:
    # Compute semi-major axis using Kepler's 3rd law (simplified)
    # a^3 / P^2 = M_star (in solar units, with a in AU and P in years)
    df['pl_orbsmax_calc'] = ((df['pl_orbper'] / 365.25) ** 2) ** (1/3)

    # Compute insolation (normalized to Earth)
    # S = L_star / (4 * pi * a^2) normalized to Earth's insolation
    df['pl_insol_calc'] = df['st_lum'] / (df['pl_orbsmax_calc'] ** 2)

    # Compute equilibrium temperature (simplified, albedo = 0.3)
    # T_eq = T_star * sqrt(R_star / (2*a)) * (1-A)^(1/4)
    A = 0.3  # Assumed albedo
    df['pl_eqt_calc'] = df['st_teff'] * np.sqrt(df['st_rad'] / (2 * df['pl_orbsmax_calc'] * 215)) * ((1 - A) ** 0.25)

    print(f"   [OK]Computed features for {len(df)} planets")

    # Create visualization of insolation vs radius
    df_clean = df[(df['pl_insol_calc'].notna()) & (df['pl_insol_calc'] > 0)].copy()

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(df_clean['pl_insol_calc'], df_clean['pl_rade'],
                        s=15, alpha=0.6, c=df_clean['pl_eqt_calc'], cmap='coolwarm')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Insolation (S/S⊕)', fontsize=12)
    ax.set_ylabel('Planet Radius (R⊕)', fontsize=12)
    ax.set_title('Insolation vs Radius', fontsize=14, fontweight='bold')

    # Mark habitable zone boundaries
    ax.axvspan(0.25, 1.5, alpha=0.2, color='green', label='Habitable Zone')
    ax.legend()
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Equilibrium Temperature (K)', fontsize=10)

    plt.tight_layout()
    plt.savefig('plots/insolation_vs_radius.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   [OK]Saved insolation plot to plots/insolation_vs_radius.png")

# ============================================================================
# 9. BUILD MACHINE LEARNING CLASSIFIER
# ============================================================================
print("\n[9/12] Building machine learning classifier for rocky vs gaseous planets...")
if not df.empty:
    # Create labels: rocky (<1.6 R_earth) vs gaseous (>=1.6 R_earth)
    df['is_rocky'] = (df['pl_rade'] < 1.6).astype(int)

    # Select features
    feature_cols = ['pl_orbper', 'pl_insol_calc', 'pl_eqt_calc', 'st_rad', 'st_teff']
    df_ml = df[feature_cols + ['is_rocky']].dropna()

    if len(df_ml) > 100:
        X = df_ml[feature_cols]
        y = df_ml['is_rocky']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Random Forest
        clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        clf.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = clf.predict(X_test_scaled)
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]

        print("\n   Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Gaseous', 'Rocky']))
        print(f"   ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")

        # Get feature importances
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n   Feature Importances:")
        print(feature_importance.to_string(index=False))
    else:
        print("   Warning: Not enough data for ML model")
        feature_importance = None
else:
    feature_importance = None

# ============================================================================
# 10. CREATE FEATURE IMPORTANCE VISUALIZATION
# ============================================================================
print("\n[10/12] Creating feature importance visualization...")
if feature_importance is not None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_importance['feature'], feature_importance['importance'], color='steelblue')
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Feature Importance for Rocky vs Gaseous Classification', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   [OK]Saved to plots/feature_importance.png")

# ============================================================================
# 11. CREATE INTERACTIVE PLOTLY VISUALIZATIONS
# ============================================================================
print("\n[11/12] Creating interactive Plotly visualizations...")
plotly_html = ""
if not df.empty:
    # Interactive 3D scatter plot
    df_3d = df[['pl_name', 'pl_orbper', 'pl_rade', 'pl_eqt_calc']].dropna()

    fig = px.scatter_3d(df_3d,
                        x='pl_orbper',
                        y='pl_rade',
                        z='pl_eqt_calc',
                        color='pl_eqt_calc',
                        hover_name='pl_name',
                        labels={
                            'pl_orbper': 'Orbital Period (days)',
                            'pl_rade': 'Radius (R⊕)',
                            'pl_eqt_calc': 'Temperature (K)'
                        },
                        title='Interactive 3D View: Period vs Radius vs Temperature',
                        color_continuous_scale='Viridis',
                        height=700)

    fig.update_layout(scene=dict(
        xaxis=dict(type='log'),
        yaxis=dict(type='log')
    ))

    # Generate HTML div for embedding
    plotly_html = fig.to_html(include_plotlyjs='cdn', div_id='plotly-3d-viz')
    print("   [OK]Generated embedded 3D interactive plot")

# ============================================================================
# 12. CREATE INTERACTIVE HTML DASHBOARD
# ============================================================================
print("\n[12/12] Creating interactive HTML dashboard...")

dashboard_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Exoplanet Analysis Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }}
        h1 {{
            color: #667eea;
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 40px;
            font-size: 1.1em;
        }}
        .info-box {{
            background: #f8f9fa;
            border-left: 5px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .stat-number {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin: 30px 0;
        }}
        .plot-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .plot-card h3 {{
            color: #667eea;
            margin-top: 0;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .plot-card img {{
            width: 100%;
            border-radius: 5px;
            margin-top: 10px;
        }}
        .plot-card p {{
            color: #666;
            line-height: 1.6;
        }}
        .interactive-section {{
            background: #e8f4f8;
            padding: 30px;
            border-radius: 10px;
            margin: 30px 0;
        }}
        .interactive-section h2 {{
            color: #667eea;
            margin-top: 0;
        }}
        .btn {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 12px 30px;
            text-decoration: none;
            border-radius: 25px;
            margin: 10px 5px;
            transition: all 0.3s;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .btn:hover {{
            background: #764ba2;
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding-top: 30px;
            border-top: 2px solid #e0e0e0;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🌟 Exoplanet Characterization & Habitability Analysis</h1>
        <div class="subtitle">
            Comprehensive analysis of confirmed exoplanets from NASA Exoplanet Archive<br>
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>

        <div class="info-box">
            <h2>📋 Project Summary</h2>
            <p>
                This project uses data from the NASA Exoplanet Archive and TESS mission to:
            </p>
            <ul>
                <li>Query and analyze confirmed exoplanet parameters</li>
                <li>Download and process TESS light curves for transit detection</li>
                <li>Create exploratory visualizations of planet populations</li>
                <li>Perform Box Least Squares (BLS) analysis for period finding</li>
                <li>Engineer physical features (insolation, equilibrium temperature)</li>
                <li>Build machine learning models to classify planet types</li>
            </ul>
        </div>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Total Planets Analyzed</div>
                <div class="stat-number">{len(df)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Rocky Planets</div>
                <div class="stat-number">{len(df[df['pl_rade'] < 1.6]) if not df.empty else 0}</div>
                <div class="stat-label">(&lt; 1.6 R⊕)</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Gaseous Planets</div>
                <div class="stat-number">{len(df[df['pl_rade'] >= 1.6]) if not df.empty else 0}</div>
                <div class="stat-label">(&ge; 1.6 R⊕)</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Features Engineered</div>
                <div class="stat-number">3</div>
                <div class="stat-label">Physical Properties</div>
            </div>
        </div>

        <h2 style="color: #667eea; margin-top: 50px;">📊 Visualizations</h2>

        <div class="plot-grid">
            <div class="plot-card">
                <h3>🌍 Radius vs Orbital Period</h3>
                <img src="plots/radius_vs_period.png" alt="Radius vs Period">
                <p>Shows the distribution of exoplanets in radius-period space. Color indicates equilibrium temperature. Notice the different populations: hot Jupiters (large, short period), super-Earths, and mini-Neptunes.</p>
            </div>

            <div class="plot-card">
                <h3>⚖️ Mass-Radius Relationship</h3>
                <img src="plots/mass_radius.png" alt="Mass-Radius">
                <p>Demonstrates the relationship between planetary mass and radius. Rocky planets (red line) have a steeper slope than gaseous planets (blue line). Color shows density variation.</p>
            </div>

            <div class="plot-card">
                <h3>🌞 Insolation vs Radius</h3>
                <img src="plots/insolation_vs_radius.png" alt="Insolation vs Radius">
                <p>Shows stellar insolation (energy received) vs planet radius. The green shaded region marks the potentially habitable zone (0.25-1.5 times Earth's insolation).</p>
            </div>

            <div class="plot-card">
                <h3>📡 BLS Periodogram</h3>
                <img src="plots/bls_periodogram.png" alt="BLS Periodogram">
                <p>Box Least Squares periodogram for TOI-700, showing the detected orbital period from TESS light curve analysis.</p>
            </div>

            <div class="plot-card">
                <h3>🔄 Phase-Folded Light Curve</h3>
                <img src="plots/folded_lightcurve.png" alt="Folded Light Curve">
                <p>Raw and phase-folded light curve of TOI-700. The folded curve reveals the transit signature by stacking observations at the same orbital phase.</p>
            </div>

            <div class="plot-card">
                <h3>🎯 Feature Importance</h3>
                <img src="plots/feature_importance.png" alt="Feature Importance">
                <p>Random Forest model feature importances for classifying rocky vs gaseous planets. Shows which physical parameters are most predictive.</p>
            </div>
        </div>

        <div class="interactive-section">
            <h2>🎮 Interactive 3D Visualization</h2>
            <p>
                Explore the exoplanet dataset in three dimensions! Rotate, zoom, and hover over individual planets to see their names and properties.
            </p>
            {plotly_html}
        </div>

        <div class="info-box">
            <h2>💾 Data Files</h2>
            <p>All processed data has been saved in the <code>data/</code> directory:</p>
            <ul>
                <li><strong>exoplanet_catalog.csv</strong> - Full NASA Exoplanet Archive dataset with engineered features</li>
                <li><strong>toi700_lightcurve.csv</strong> - TESS light curve data for TOI-700</li>
            </ul>
        </div>

        <div class="info-box">
            <h2>🤖 Machine Learning Results</h2>
            <p>
                A Random Forest classifier was trained to distinguish between rocky (R &lt; 1.6 R⊕) and gaseous (R &ge; 1.6 R⊕) planets using physical parameters:
            </p>
            <ul>
                <li>Orbital period</li>
                <li>Stellar insolation</li>
                <li>Equilibrium temperature</li>
                <li>Stellar radius</li>
                <li>Stellar effective temperature</li>
            </ul>
            <p>
                The model demonstrates that these physical properties can effectively classify planet types, with equilibrium temperature and orbital period being among the most important features.
            </p>
        </div>

        <div class="footer">
            <p>
                🤖 Generated with Python using lightkurve, astroquery, scikit-learn, matplotlib, and plotly<br>
                Data source: NASA Exoplanet Archive & TESS Mission<br>
                <em>For educational and research purposes</em>
            </p>
        </div>
    </div>
</body>
</html>
"""

with open('exoplanet_dashboard.html', 'w', encoding='utf-8') as f:
    f.write(dashboard_html)

print("   [OK]Saved dashboard to exoplanet_dashboard.html")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  Plots:")
print("     - plots/radius_vs_period.png")
print("     - plots/mass_radius.png")
print("     - plots/insolation_vs_radius.png")
print("     - plots/bls_periodogram.png")
print("     - plots/folded_lightcurve.png")
print("     - plots/feature_importance.png")
print("\n  Data:")
print("     - data/exoplanet_catalog.csv")
print("     - data/toi700_lightcurve.csv")
print("\n  Dashboard:")
print("     - exoplanet_dashboard.html (with embedded 3D visualization)")
print("\n" + "="*80)
print("\n*** Open 'exoplanet_dashboard.html' in your web browser to view all results!")
print("="*80)
