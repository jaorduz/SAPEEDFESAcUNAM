
# Gettin SVG and PNG files
# python export_figures.py --data-dir data --professor demo_prof_03 --format svg
# python export_figures.py --data-dir data --professor demo_prof_03 --format svg --transparent --ratio 4:3
# python export_figures.py --data-dir data --professor demo_prof_04 --target-dim D1 --format svg --transparent --ratio 16:9

import argparse
import re
from pathlib import Path
import math 

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


UNAM_BLUE = "#003366"
UNAM_GOLD = "#C9A227"
ITEM_RE = re.compile(r"^D(\d+)Q(\d+)$")


def mean_ci(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)
    if n < 2:
        return (float(s.mean()) if n == 1 else np.nan, np.nan)
    mean = float(s.mean())
    ci = 1.96 * float(s.std(ddof=1)) / np.sqrt(n)
    return mean, ci


def load_professor_files(data_dir: Path):
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    all_professor_data = []
    professor_dimension_means = []
    reference_dim_structure = None
    reference_item_structure = None

    for file in files:
        prof_id = file.stem
        df = pd.read_csv(file)

        item_cols = []
        dim_item_map = {}

        for col in df.columns:
            m = ITEM_RE.match(col)
            if m:
                item_cols.append(col)
                dim = f"D{m.group(1)}"
                dim_item_map.setdefault(dim, []).append(col)

        if not item_cols:
            continue

        dimension_cols = sorted(dim_item_map.keys(), key=lambda x: int(x[1:]))

        for dim in dim_item_map:
            dim_item_map[dim] = sorted(
                dim_item_map[dim],
                key=lambda x: int(ITEM_RE.match(x).group(2))
            )

        current_item_structure = {
            dim: tuple(dim_item_map[dim]) for dim in dimension_cols
        }

        if reference_dim_structure is None:
            reference_dim_structure = dimension_cols
            reference_item_structure = current_item_structure
        else:
            if dimension_cols != reference_dim_structure:
                raise ValueError(f"Inconsistent dimension structure in {file.name}")
            if current_item_structure != reference_item_structure:
                raise ValueError(f"Inconsistent item structure in {file.name}")

        df[item_cols] = df[item_cols].apply(pd.to_numeric, errors="coerce")

        for dim in dimension_cols:
            df[dim] = df[list(current_item_structure[dim])].mean(axis=1)

        prof_means = df[dimension_cols].mean().to_dict()
        prof_means["ProfesorID"] = prof_id
        prof_means["N_estudiantes"] = len(df)

        professor_dimension_means.append(prof_means)

        df["ProfesorID"] = prof_id
        all_professor_data.append(df)

    if not all_professor_data:
        raise ValueError("No valid professor CSV files were loaded.")

    institutional_df = pd.concat(all_professor_data, ignore_index=True)
    professor_means_df = pd.DataFrame(professor_dimension_means)

    return institutional_df, professor_means_df, reference_dim_structure, reference_item_structure


def apply_publication_layout(fig, width, height, transparent=True):
    bg = "rgba(0,0,0,0)" if transparent else "white"

    fig.update_layout(
        width=width,
        height=height,
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(color="black", size=15),
        title_font=dict(color="black", size=15),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="black", size=12),
            bgcolor="rgba(0,0,0,0)"
        ),
        # xaxis=dict(
        #     visible=False,
        #     range=[6,-6],
        #     scaleanchor="y",
        #     scaleratio=1
        # ),
        # yaxis=dict(
        #     visible=False,
        #     range=[6,-6],
        # ),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    # Make axes readable for exported figures
    if "xaxis" in fig.layout:
        fig.update_xaxes(
            title_font=dict(color="black", size=15),
            tickfont=dict(color="black", size=15),
            showgrid=False
        )
    if "yaxis" in fig.layout:
        fig.update_yaxes(
            title_font=dict(color="black", size=15),
            tickfont=dict(color="black", size=15),
            gridcolor="lightgray"
        )

    return fig




def save_figure(fig, outpath: Path, scale: int = 3):
    outpath.parent.mkdir(parents=True, exist_ok=True)

    suffix = outpath.suffix.lower()

    if suffix == ".svg":
        fig.write_image(str(outpath), format="svg")
    else:
        fig.write_image(str(outpath), scale=scale)


def build_bar_figure(institutional_df, dimension_cols, selected_professor, width, height, transparent):
    df_selected = institutional_df[institutional_df["ProfesorID"] == selected_professor]

    prof_means = []
    inst_means = []
    inst_upper = []
    inst_lower = []

    for dim in dimension_cols:
        m_prof, _ = mean_ci(df_selected[dim])
        m_inst, ci_inst = mean_ci(institutional_df[dim])

        prof_means.append(m_prof)
        inst_means.append(m_inst)
        inst_upper.append(m_inst + ci_inst if not np.isnan(ci_inst) else np.nan)
        inst_lower.append(m_inst - ci_inst if not np.isnan(ci_inst) else np.nan)

    df_compare = pd.DataFrame({
        "Dimensión": dimension_cols,
        "Media Profesor": prof_means,
        "Media Institucional": inst_means,
        "Límite Inferior 95% Inst.": inst_lower,
        "Límite Superior 95% Inst.": inst_upper,
    })

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_compare["Dimensión"],
        y=df_compare["Media Profesor"],
        name=selected_professor,
        marker_color=UNAM_BLUE
    ))

    fig.add_trace(go.Bar(
        x=df_compare["Dimensión"],
        y=df_compare["Media Institucional"],
        name="Institutional Mean",
        marker_color=UNAM_GOLD
    ))

    fig.add_trace(go.Scatter(
        x=df_compare["Dimensión"],
        y=df_compare["Media Institucional"],
        mode="markers",
        error_y=dict(
            type="data",
            symmetric=False,
            array=df_compare["Límite Superior 95% Inst."] - df_compare["Media Institucional"],
            arrayminus=df_compare["Media Institucional"] - df_compare["Límite Inferior 95% Inst."],
            thickness=1.5,
            width=5
        ),
        marker=dict(color=UNAM_GOLD, size=5),
        showlegend=False
    ))

    fig.update_layout(
        barmode="group",
        title=f"{selected_professor} vs Institutional Mean",
        xaxis_title="Dimension",
        yaxis_title="Mean Score"
    )

    return apply_publication_layout(fig, width, height, transparent)

####### To fix the radar plot problem.


def build_radar_figure(institutional_df, dimension_cols, selected_professor, width, height, transparent):
    df_selected = institutional_df[institutional_df["ProfesorID"] == selected_professor]

    # -----------------------------
    # Compute means and institutional CI
    # -----------------------------
    prof_means = []
    inst_means = []
    inst_upper = []
    inst_lower = []

    for dim in dimension_cols:
        m_prof, _ = mean_ci(df_selected[dim])
        m_inst, ci_inst = mean_ci(institutional_df[dim])

        prof_means.append(m_prof)
        inst_means.append(m_inst)
        inst_upper.append(m_inst + ci_inst if not np.isnan(ci_inst) else np.nan)
        inst_lower.append(m_inst - ci_inst if not np.isnan(ci_inst) else np.nan)

    # -----------------------------
    # Geometry
    # -----------------------------
    n_dims = len(dimension_cols)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False)
    angles = np.append(angles, angles[0])  # close loop

    prof_vals = np.append(prof_means, prof_means[0])
    inst_vals = np.append(inst_means, inst_means[0])
    upper_vals = np.append(inst_upper, inst_upper[0])
    lower_vals = np.append(inst_lower, inst_lower[0])

    def pol2cart(r_vals, theta_vals):
        x = r_vals * np.cos(theta_vals)
        y = r_vals * np.sin(theta_vals)
        return x, y

    bg = "rgba(0,0,0,0)" if transparent else "white"

    fig = go.Figure()

    # -----------------------------
    # Draw concentric circles (Likert scale 1–5)
    # -----------------------------
    circle_theta = np.linspace(0, 2 * np.pi, 361)
    for r in [1, 2, 3, 4, 5]:
        x_c = r * np.cos(circle_theta)
        y_c = r * np.sin(circle_theta)

        fig.add_trace(go.Scatter(
            x=x_c,
            y=y_c,
            mode="lines",
            line=dict(color="black", width=1),
            hoverinfo="skip",
            showlegend=False
        ))

    # -----------------------------
    # Draw radial spokes
    # -----------------------------
    for ang, label in zip(angles[:-1], dimension_cols):
        x_end = 5 * np.cos(ang)
        y_end = 5 * np.sin(ang)

        fig.add_trace(go.Scatter(
            x=[0, x_end],
            y=[0, y_end],
            mode="lines",
            line=dict(color="black", width=1),
            hoverinfo="skip",
            showlegend=False
        ))

        # label position slightly outside the radius
        x_lab = 5.5 * np.cos(ang)
        y_lab = 5.5 * np.sin(ang)

        fig.add_trace(go.Scatter(
            x=[x_lab],
            y=[y_lab],
            mode="text",
            text=[label],
            textfont=dict(color="black", size=14),
            hoverinfo="skip",
            showlegend=False
        ))

    # -----------------------------
    # Radial tick labels
    # -----------------------------
    for r in [1, 2, 3, 4, 5]:
        fig.add_trace(go.Scatter(
            x=[0.15],
            y=[r],
            mode="text",
            text=[str(r)],
            textfont=dict(color="black", size=12),
            hoverinfo="skip",
            showlegend=False
        ))

    # -----------------------------
    # Institutional CI band
    # -----------------------------
    x_upper, y_upper = pol2cart(upper_vals, angles)
    x_lower, y_lower = pol2cart(lower_vals, angles)

    band_x = np.concatenate([x_upper, x_lower[::-1]])
    band_y = np.concatenate([y_upper, y_lower[::-1]])

    fig.add_trace(go.Scatter(
        x=band_x,
        y=band_y,
        fill="toself",
        fillcolor="rgba(201,162,39,0.15)",
        line=dict(color="rgba(0,0,0,0)", width=0),
        name="Institutional 95% CI"
    ))

    # -----------------------------
    # Institutional mean polygon
    # -----------------------------
    x_inst, y_inst = pol2cart(inst_vals, angles)
    fig.add_trace(go.Scatter(
        x=x_inst,
        y=y_inst,
        mode="lines",
        fill="toself",
        fillcolor="rgba(201,162,39,0.10)",
        line=dict(color=UNAM_GOLD, width=2.5),
        name="Institutional Mean"
    ))

    # -----------------------------
    # Professor polygon
    # -----------------------------
    x_prof, y_prof = pol2cart(prof_vals, angles)
    fig.add_trace(go.Scatter(
        x=x_prof,
        y=y_prof,
        mode="lines",
        fill="toself",
        fillcolor="rgba(0,51,102,0.12)",
        line=dict(color=UNAM_BLUE, width=3),
        name=selected_professor
    ))

    # -----------------------------
    # Layout
    # -----------------------------
    fig.update_layout(
        # title=f"Radar Profile: {selected_professor}",
        width=width,
        height=height,
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        font=dict(color="black", size=14),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="black", size=12),
            bgcolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(
            visible=False,
            range=[-6, 6],
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            visible=False,
            range=[-6, 6]
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig

#######
def build_corr_figure(institutional_df, dimension_cols, width, height, transparent):
    corr = institutional_df[dimension_cols].corr(method="pearson")

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Institutional Correlation Matrix"
    )

    fig.update_traces(textfont=dict(color="black", size=15))
    fig.update_layout(coloraxis_colorbar=dict(title="r"))

    return apply_publication_layout(fig, width, height, transparent)


def build_lambda_figure(institutional_df, dimension_cols, target_dim, width, height, transparent, k_folds=5):
    predictors = [d for d in dimension_cols if d != target_dim]
    df_model = institutional_df[predictors + [target_dim]].dropna()

    X = df_model[predictors]
    y = df_model[target_dim]

    lambda_grid = np.logspace(-4, 4, 50)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    mse_mean = []
    mse_std = []

    for lam in lambda_grid:
        fold_mse = []

        for tr, te in kf.split(X):
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=float(lam)))
            ])
            model.fit(X.iloc[tr], y.iloc[tr])
            pred = model.predict(X.iloc[te])
            fold_mse.append(mean_squared_error(y.iloc[te], pred))

        mse_mean.append(float(np.mean(fold_mse)))
        mse_std.append(float(np.std(fold_mse, ddof=1)) if len(fold_mse) > 1 else 0.0)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=lambda_grid,
        y=mse_mean,
        mode="lines+markers",
        name="Mean CV MSE",
        line=dict(color=UNAM_BLUE)
    ))

    fig.add_trace(go.Scatter(
        x=lambda_grid,
        y=np.array(mse_mean) + np.array(mse_std),
        mode="lines",
        name="+1 std",
        line=dict(color=UNAM_GOLD, dash="dot")
    ))

    fig.add_trace(go.Scatter(
        x=lambda_grid,
        y=np.array(mse_mean) - np.array(mse_std),
        mode="lines",
        name="-1 std",
        line=dict(color=UNAM_GOLD, dash="dot")
    ))

    fig.update_layout(
        title=f"Regularization Curve for {target_dim}",
        xaxis_type="log",
        xaxis_title="λ (log scale)",
        yaxis_title="CV-MSE"
    )

    return apply_publication_layout(fig, width, height, transparent)


def parse_ratio(ratio_str: str, base: int = 1600):
    w, h = ratio_str.split(":")
    w = float(w)
    h = float(h)
    factor = base / w
    return int(w * factor), int(h * factor)


def main():
    parser = argparse.ArgumentParser(description="Export publication-quality figures from SAPEED data.")
    parser.add_argument("--data-dir", type=str, required=True, help="Folder containing professor CSV files")
    parser.add_argument("--professor", type=str, required=True, help="ProfessorID to export comparison figures for")
    parser.add_argument("--target-dim", type=str, default=None, help="Target dimension for ridge lambda curve, e.g. D1")
    parser.add_argument("--ratio", type=str, default="4:3", help="Aspect ratio, e.g. 4:3, 16:9, 2:4")
    parser.add_argument("--scale", type=int, default=3, help="Export scale (higher = more resolution)")
    parser.add_argument("--transparent", action="store_true", help="Export with transparent background")
    parser.add_argument("--outdir", type=str, default="figures_export", help="Output directory")
    parser.add_argument("--format", type=str, default="png", choices=["png", "svg"], help="Output figure format")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    ext = args.format

    institutional_df, professor_means_df, dimension_cols, reference_item_structure = load_professor_files(data_dir)

    if args.professor not in professor_means_df["ProfesorID"].tolist():
        raise ValueError(f"Professor '{args.professor}' not found.")

    width, height = parse_ratio(args.ratio)

    # Export bar
    fig_bar = build_bar_figure(
        institutional_df, dimension_cols, args.professor, width, height, args.transparent
    )
    # save_figure(fig_bar, outdir / f"{args.professor}_bar.png", scale=args.scale)
    save_figure(fig_bar, outdir / f"{args.professor}_bar.{ext}", scale=args.scale)

    # Export radar
    fig_radar = build_radar_figure(
        institutional_df, dimension_cols, args.professor, width, height, args.transparent
    )
    # save_figure(fig_radar, outdir / f"{args.professor}_radar.png", scale=args.scale)    

    # fig_radar = go.Figure(fig_radar) # partiall solution.

    save_figure(fig_radar, outdir / f"{args.professor}_radar.{ext}", scale=args.scale)

    # Export correlation
    fig_corr = build_corr_figure(
        institutional_df, dimension_cols, width, height, args.transparent
    )
    # save_figure(fig_corr, outdir / "institutional_correlation.png", scale=args.scale)
    save_figure(fig_corr, outdir / f"institutional_correlation.{ext}", scale=args.scale)

    # Export lambda curve if requested
    if args.target_dim is not None:
        if args.target_dim not in dimension_cols:
            raise ValueError(f"Target dimension '{args.target_dim}' not found.")
        fig_lambda = build_lambda_figure(
            institutional_df, dimension_cols, args.target_dim, width, height, args.transparent
        )
        # save_figure(fig_lambda, outdir / f"ridge_lambda_{args.target_dim}.png", scale=args.scale)
        save_figure(fig_lambda, outdir / f"ridge_lambda_{args.target_dim}.{ext}", scale=args.scale)
    print(f"Figures exported to: {outdir.resolve()}")


if __name__ == "__main__":
    main()