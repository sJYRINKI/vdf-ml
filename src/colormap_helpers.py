def expr_velocity(exprmaps, requestvariables=False):
    """
    Return bulk velocity for Analysator colormap plotting.

    Parameters
    ----------
    exprmaps : dict
        Variables requested from the VLSV file.
    requestvariables : bool, optional
        If true, return the required variable names.

    Returns
    -------
    list[str] or numpy.ndarray
        Required variable names, or the bulk velocity vector field.
    """

    if requestvariables is True:
        return ["rho", "rho_v"]

    rho = exprmaps["rho"][:, :]
    rhov = exprmaps["rho_v"][:, :, :]

    return rhov / rho[:, :, None]


def scatter_label_points(ax, metadata_rows):
    """
    Scatter saved dataset labels on an x-z colormap.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes where points are drawn.
    metadata_rows : pandas.DataFrame
        Metadata rows for one timestep.
    """

    styles = {
        "reconnection": {"color": "red", "marker": "x"},
        "other": {"color": "blue", "marker": "o"},
        "lobe": {"color": "gold", "marker": "2"},
    }

    plotted_classes = set()

    for _, row in metadata_rows.iterrows():
        class_name = row["class_name"]
        style = styles[class_name]
        label = class_name if class_name not in plotted_classes else None

        ax.scatter(
            row["x_re"],
            row["z_re"],
            label=label,
            s=50,
            **style,
        )

        plotted_classes.add(class_name)

    if plotted_classes:
        ax.legend()
