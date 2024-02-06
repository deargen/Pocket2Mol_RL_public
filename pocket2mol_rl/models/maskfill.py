from .actor_model import ProbabilityMixIn


class MaskFillModelVN(ProbabilityMixIn):
    """Model for sampling and probability calculation.
    Base class is PretrainedMaskFillModel.
    Computing probability and sampling are implemented in ProbabilityMixIn and SamplingMixIn.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
