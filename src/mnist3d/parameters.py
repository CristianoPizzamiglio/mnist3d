from __future__ import annotations

import yaml
from pydantic import BaseModel, Field
from typing_extensions import Annotated


class Parameters(BaseModel):
    """
    Parameters.

    Attributes
    ----------
    pixel_intensity_threshold : int
    noise_standard_deviation : float

    """

    pixel_intensity_threshold: Annotated[int, Field(strict=True, gt=0, lt=255)]
    noise_standard_deviation: Annotated[float, Field(strict=True, gt=0.0, lt=1.0)]


def import_parameters() -> Parameters:
    """
    Import parameters.

    Returns
    -------
    Parameters

    """
    with open("params.yaml", "r") as file:
        parameter_to_value = yaml.safe_load(file.read())

    return Parameters(
        pixel_intensity_threshold=parameter_to_value["pixel_intensity_threshold"],
        noise_standard_deviation=parameter_to_value["noise_standard_deviation"],
    )
