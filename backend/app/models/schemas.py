from pydantic import BaseModel
from typing import List, Optional

class SegmentFlag(BaseModel):
    start_time: float
    end_time: float
    reason: str

class AcousticComparison(BaseModel):
    voice_rt60: Optional[float] = None
    voice_drr: Optional[float] = None
    bg_rt60: Optional[float] = None
    bg_drr: Optional[float] = None

class PredictionResponse(BaseModel):
    filename: str
    is_fake: bool
    confidence: float
    rir_mismatch_score: Optional[float] = None
    breathing_score: Optional[float] = None
    fusion_enabled: bool = True
    flagged_segments: List[SegmentFlag] = []
    acoustic_comparison: Optional[AcousticComparison] = None
    inference_time_ms: Optional[float] = None
