# API Contract - POST /predict
Request: multipart/form-data, field "file" (audio file)
Response:
{
"filename": "string",
"is_fake": true,
"confidence": 0.93,
"rir_mismatch_score": 0.81,
"breathing_score": 0.44,
"fusion_enabled": true,
"flagged_segments": [
{"start_time": 1.2, "end_time": 2.4, "reason": "RIR mismatch"}
]
}
