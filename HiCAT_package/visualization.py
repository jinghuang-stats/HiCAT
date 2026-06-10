import matplotlib.pyplot as plt
import scanpy as sc


def cat_figure(input_adata, x_col, y_col, fig_title, fig_path, color_key, cat_color, size, invert_x=False, invert_y=True):
    input_adata.obs[color_key] = input_adata.obs[color_key].astype("category")
    fig = sc.pl.scatter(
        input_adata,
        alpha=1,
        x=x_col,
        y=y_col,
        color=color_key,
        palette=cat_color,
        show=False,
        size=size,
    )
    fig.set_aspect("equal", "box")
    if invert_y is True:
        fig.invert_yaxis()
    if invert_x is True:
        fig.invert_xaxis()
    fig.set_title(fig_title)
    fig.figure.savefig(fig_path, dpi=200, bbox_inches="tight")
    del input_adata.uns[color_key + "_colors"]
    plt.clf()
    plt.close()


def con_figure(input_adata, x_col, y_col, fig_title, fig_path, color_key, cnt_color, size, invert_x=False, invert_y=True):
    fig = sc.pl.scatter(
        input_adata,
        alpha=1,
        x=x_col,
        y=y_col,
        color=color_key,
        color_map=cnt_color,
        show=False,
        size=size,
    )
    fig.set_aspect("equal", "box")
    if invert_y is True:
        fig.invert_yaxis()
    if invert_x is True:
        fig.invert_xaxis()
    fig.set_title(fig_title)
    fig.figure.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.clf()
    plt.close()

