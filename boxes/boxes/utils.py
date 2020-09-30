import torch


def log1mexp(x):
    """
    Return log(1 - exp(-x)).

    For details, see
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    A = torch.log(-torch.expm1(-x))
    B = torch.log1p(-torch.exp(-x))
    Z = torch.empty_like(A)
    switch = x < 0.683
    Z[switch] = A[switch]
    Z[1 - switch] = B[1 - switch]
    return Z
