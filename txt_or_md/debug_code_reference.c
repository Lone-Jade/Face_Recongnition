/*
 * REFERENCE FILE — Debug printf code preserved for future testing.
 * This file is NOT compiled. It documents the debug output
 * that was added to ai_detection.c and main.c during the
 * face detection deployment debugging process.
 *
 * To re-enable debug output:
 *   1. Set #define AI_DBG 1 in ai_detection.c
 *   2. Restore printf lines in main.c (see below)
 *   3. Ensure fputc() redirect is active (provided by aiTestUtility.c)
 */

/* ================================================================
   SECTION 1: ai_detection.c — AI_DBG macro and debug printf blocks
   ================================================================ */

/* In ai_detection.c, line ~16-18: */
#define AI_DBG 1  /* Set to 1 to enable, 0 to disable */

/* The following printf blocks are guarded by #if AI_DBG ... #endif */

/* --- Postprocess header --- */
#if AI_DBG
  printf("
=== AI Postprocess ===
");
  printf("threshold=%.3f nms=%.3f minbox=%.1f max_dets=%d
",
         threshold, nms_threshold, min_box_size, max_dets);
  printf("Output tensor pointers:
");
  for (k = 0; k < 12; k++) {
    printf("  out[%d]=%p
", k, (void*)outputs[k]);
  }
#endif

/* --- Per-stride sampling --- */
#if AI_DBG
    printf("Stride %d (gs=%d): cls[0..4]=", (int)stride, gs);
    for (i = 0; i < 5 && i < gs*gs; i++)
      printf("%.4f ", cls[i]);
    printf(" obj[0..4]=");
    for (i = 0; i < 5 && i < gs*gs; i++)
      printf("%.4f ", obj[i]);
    printf("
");
    printf("  bbox[0,0] CHW: dx=%.4f dy=%.4f dw=%.4f dh=%.4f
",
           bbox[0*gs*gs+0], bbox[1*gs*gs+0],
           bbox[2*gs*gs+0], bbox[3*gs*gs+0]);
    printf("  bbox[0,0] IL:  dx=%.4f dy=%.4f dw=%.4f dh=%.4f
",
           bbox[0*4+0], bbox[0*4+1], bbox[0*4+2], bbox[0*4+3]);
#endif

/* --- Per-stride candidate stats --- */
#if AI_DBG
    printf("  Stride %d raw candidates: %d (threshold=%.3f)
",
           (int)stride, stride_candidates, threshold);
    printf("  Top-5 scores: %.4f %.4f %.4f %.4f %.4f
",
           top_scores[0], top_scores[1], top_scores[2],
           top_scores[3], top_scores[4]);
#endif

/* --- Best candidate bbox comparison (CHW vs IL) --- */
#if AI_DBG
    if (stride_candidates > 0 && top_scores[0] > 0.0f) {
      int best_loc = -1, best_i = -1, best_j = -1;
      float best_score = 0.0f;
      for (i = 0; i < gs; i++) {
        for (j = 0; j < gs; j++) {
          loc = i * gs + j;
          float s = cls[loc] * obj[loc];
          if (s > best_score) { best_score = s; best_loc = loc; best_i = i; best_j = j; }
        }
      }
      if (best_loc >= 0) {
        printf("  Best candidate grid=(%d,%d) loc=%d score=%.4f
",
               best_i, best_j, best_loc, best_score);
        printf("  bbox CHW[loc=%d]: dx=%.4f dy=%.4f dw=%.4f dh=%.4f
",
               best_loc,
               bbox[0*gs*gs+best_loc], bbox[1*gs*gs+best_loc],
               bbox[2*gs*gs+best_loc], bbox[3*gs*gs+best_loc]);
        printf("  bbox IL [loc=%d]: dx=%.4f dy=%.4f dw=%.4f dh=%.4f
",
               best_loc,
               bbox[best_loc*4+0], bbox[best_loc*4+1],
               bbox[best_loc*4+2], bbox[best_loc*4+3]);
        // ... CHW box decode + IL box decode printf ...
      }
    }
#endif

/* --- Total raw candidates --- */
#if AI_DBG
  printf("Total raw candidates (all strides): %d
", det_count);
#endif

/* --- Pre-NMS top-5 --- */
#if AI_DBG
    { /* scoped block for C90 compatibility */
      int show_n = det_count < 5 ? det_count : 5;
      int ta, tb, ti;
      for (ta = 0; ta < show_n - 1; ta++)
        for (tb = ta + 1; tb < show_n; tb++)
          if (dets[tb].score > dets[ta].score) {
            Detection t = dets[ta]; dets[ta] = dets[tb]; dets[tb] = t;
          }
      printf("Sorting %d raw detections...
", det_count);
      printf("Top-5 raw detections (pre-NMS):
");
      for (ti = 0; ti < show_n; ti++)
        printf("  #%d: score=%.4f box=(%.1f,%.1f,%.1f,%.1f) size=%.0fx%.0f
",
               ti, dets[ti].score,
               dets[ti].x1, dets[ti].y1, dets[ti].x2, dets[ti].y2,
               dets[ti].x2-dets[ti].x1, dets[ti].y2-dets[ti].y1);
    }
#endif

/* --- NMS stats --- */
    /* NMS loop: #if AI_DBG int nms_suppressed = 0; #endif */
#if AI_DBG
    printf("NMS: suppressed %d, remaining %d
", nms_suppressed, wi);
#endif

/* --- Final detections --- */
#if AI_DBG
  {
    int di;
    printf("=== Final detections: %d ===
", det_count);
    for (di = 0; di < det_count; di++) {
      printf("  #%d: score=%.4f box=(%.1f,%.1f)-(%.1f,%.1f) w=%.0f h=%.0f
",
             di, dets[di].score,
             dets[di].x1, dets[di].y1, dets[di].x2, dets[di].y2,
             dets[di].x2-dets[di].x1, dets[di].y2-dets[di].y1);
    }
  }
#endif

/* --- No-NMS fallback --- */
#if AI_DBG
  else {
    printf("Only %d raw detections (no sorting/NMS needed)
", det_count);
  }
#endif

/* ================================================================
   SECTION 2: main.c — preprocessing and inference debug printf
   ================================================================ */

/* In main.c, inside the main loop (if (pending_buffer < 0) block): */

      /* Step 2 debug — before preprocessing */
      printf("
--- Frame: %s ---
", use_builtin ? "builtin" : "USART");
      printf("src_img=%p ai_input=%p
", (void*)src_img, (void*)ai_input);
      printf("src pixels[0..4]=");
      {int _dbg; for(_dbg=0; _dbg<5; _dbg++) printf("0x%08lX ", (unsigned long)src_img[_dbg]);}
      printf("
");
      ai_preprocess(src_img, 320, 240, ai_input, 320, 320);
      printf("After preprocess - input tensor samples:
");
      printf("  B[0..4]: %.1f %.1f %.1f %.1f %.1f
",
             ai_input[0], ai_input[1], ai_input[2], ai_input[3], ai_input[4]);
      printf("  G[0..4]: %.1f %.1f %.1f %.1f %.1f
",
             ai_input[1*320*320+0], ai_input[1*320*320+1],
             ai_input[1*320*320+2], ai_input[1*320*320+3], ai_input[1*320*320+4]);
      printf("  R[0..4]: %.1f %.1f %.1f %.1f %.1f
",
             ai_input[2*320*320+0], ai_input[2*320*320+1],
             ai_input[2*320*320+2], ai_input[2*320*320+3], ai_input[2*320*320+4]);
      SCB_CleanDCache_by_Addr((uint32_t *)ai_input, (int32_t)(3 * 320 * 320 * sizeof(float)));

      /* Error print */
      printf("AI err: %d
", (int)ret);

      /* Final count print */
      printf("=== Final count: %d faces ===
", num_detections);

/* ================================================================
   SECTION 3: Important non-debug items to KEEP in main.c
   ================================================================ */

      /* Keep this! D-Cache clean before AI inference */
      SCB_CleanDCache_by_Addr((uint32_t *)ai_input, (int32_t)(3 * 320 * 320 * sizeof(float)));

/* End of debug reference */
