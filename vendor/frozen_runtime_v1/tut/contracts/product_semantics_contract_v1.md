# Product Semantics Contract v1

`metadata` carries compact detection records; `roi` carries target chips; `roi_context` carries ROI plus contextual evidence; `full_frame` is B1's full-observation proxy; `defer` creates no payload. Staged aliases enqueue metadata first and a residual product later.

Metadata has a 0.002 MB fallback. ROI uses a formula clamped to 0.08--3.5 MB with 0.85 MB fallback. ROI+Context is clamped to 0.25--8 MB with 2.70 MB fallback. B1 uses fixed 32 MB raw / ratio 8 = 4 MB. None are real encoded measurements in this study. Quality and information factors are proxy values interpreted by the unchanged common evaluator.
