--- Frame: builtin ---
src_img=08058130 ai_input=d0800000
src pixels[0..4]=0xFF000000 0xFF000000 0xFF000000 0xFF000000 0xFF000000 
After preprocess - input tensor samples:
  B[0..4]: 0.0 0.0 0.0 0.0 0.0
  G[0..4]: 0.0 0.0 0.0 0.0 0.0
  R[0..4]: 0.0 0.0 0.0 0.0 0.0

=== AI Postprocess ===
threshold=0.700 nms=0.300 minbox=15.0 max_dets=50
Output tensor pointers:
  out[0]=2400976c
  out[1]=240007ac
  out[2]=2400cfac
  out[3]=24007e6c
  out[4]=2400016c
  out[5]=2400c96c
  out[6]=d080fa00
  out[7]=240020ac
  out[8]=2400d13c
  out[9]=d0800000
  out[10]=24003fec
  out[11]=2400d90c
Stride 8 (gs=40): cls[0..4]=0.4872 0.4897 0.4968 0.4976 0.4973  obj[0..4]=0.0000 0.0000 0.0000 0.0000 0.0000 
  bbox[0,0] CHW: dx=0.3111 dy=0.1471 dw=0.1790 dh=0.2290
  bbox[0,0] IL:  dx=0.3111 dy=0.2023 dw=0.2085 dh=0.5011
  Stride 8 raw candidates: 1 (threshold=0.700)
  Top-5 scores: 0.7646 0.0000 0.0000 0.0000 0.0000
  Best candidate grid=(11,7) loc=447 score=0.7646
  bbox CHW[loc=447]: dx=0.2941 dy=0.2404 dw=1.0621 dh=1.2416
  bbox IL [loc=447]: dx=0.7511 dy=0.0719 dw=0.9020 dh=1.3287
  -> CHW box: (47,76)-(70,104) 23x28
  -> IL  box: (52,73)-(72,104) 20x30
Stride 16 (gs=20): cls[0..4]=0.5592 0.5628 0.5611 0.5441 0.5289  obj[0..4]=0.0001 0.0000 0.0001 0.0001 0.0001 
  bbox[0,0] CHW: dx=0.4246 dy=0.4591 dw=0.3990 dh=0.5118
  bbox[0,0] IL:  dx=0.4246 dy=0.5002 dw=-0.0515 dh=0.2182
  Stride 16 raw candidates: 7 (threshold=0.700)
  Top-5 scores: 0.8805 0.8789 0.8768 0.8758 0.8739
  Best candidate grid=(6,12) loc=132 score=0.8805
  bbox CHW[loc=132]: dx=0.3116 dy=-0.5819 dw=0.3928 dh=0.3600
  bbox IL [loc=132]: dx=0.4496 dy=-0.3757 dw=1.4689 dh=1.8021
  -> CHW box: (185,75)-(209,98) 24x23
  -> IL  box: (164,41)-(234,138) 70x97
Stride 32 (gs=10): cls[0..4]=0.4866 0.4696 0.4654 0.4679 0.4918  obj[0..4]=0.0001 0.0001 0.0001 0.0003 0.0001 
  bbox[0,0] CHW: dx=0.3894 dy=0.7816 dw=0.7705 dh=0.4492
  bbox[0,0] IL:  dx=0.3894 dy=0.5661 dw=-0.4122 dh=-0.0393
  Stride 32 raw candidates: 0 (threshold=0.700)
  Top-5 scores: 0.0000 0.0000 0.0000 0.0000 0.0000
Total raw candidates (all strides): 8
Sorting 8 raw detections...
Top-5 raw detections (pre-NMS):
  #0: score=0.8789 box=(190.1,63.6,208.7,91.2) size=19x28
  #1: score=0.8768 box=(166.7,68.5,184.3,96.1) size=18x28
  #2: score=0.8729 box=(186.3,75.5,213.0,101.4) size=27x26
  #3: score=0.8614 box=(169.0,56.3,187.5,80.1) size=19x24
  #4: score=0.7646 box=(46.8,76.1,69.9,103.8) size=23x28
NMS: suppressed 3, remaining 5
=== Final detections: 5 ===
  #0: score=0.8805 box=(185.1,75.2)-(208.8,98.2) w=24 h=23
  #1: score=0.8768 box=(166.7,68.5)-(184.3,96.1) w=18 h=28
  #2: score=0.8758 box=(199.1,76.0)-(221.0,105.4) w=22 h=29
  #3: score=0.8614 box=(169.0,56.3)-(187.5,80.1) w=19 h=24
  #4: score=0.7646 box=(46.8,76.1)-(69.9,103.8) w=23 h=28
=== Final count: 5 faces ===

--- Frame: builtin ---
src_img=0800d130 ai_input=d0800000
src pixels[0..4]=0xFF000000 0xFF000000 0xFF000000 0xFF000000 0xFF000000 
After preprocess - input tensor samples:
  B[0..4]: 0.0 0.0 0.0 0.0 0.0
  G[0..4]: 0.0 0.0 0.0 0.0 0.0
  R[0..4]: 0.0 0.0 0.0 0.0 0.0

=== AI Postprocess ===
threshold=0.700 nms=0.300 minbox=15.0 max_dets=50
Output tensor pointers:
  out[0]=2400976c
  out[1]=240007ac
  out[2]=2400cfac
  out[3]=24007e6c
  out[4]=2400016c
  out[5]=2400c96c
  out[6]=d080fa00
  out[7]=240020ac
  out[8]=2400d13c
  out[9]=d0800000
  out[10]=24003fec
  out[11]=2400d90c
Stride 8 (gs=40): cls[0..4]=0.5756 0.5835 0.5952 0.5720 0.5172  obj[0..4]=0.0000 0.0000 0.0000 0.0000 0.0000 
  bbox[0,0] CHW: dx=0.5008 dy=0.4501 dw=0.4134 dh=0.3935
  bbox[0,0] IL:  dx=0.5008 dy=0.7053 dw=0.6100 dh=0.9269
  Stride 8 raw candidates: 0 (threshold=0.700)
  Top-5 scores: 0.0000 0.0000 0.0000 0.0000 0.0000
Stride 16 (gs=20): cls[0..4]=0.6299 0.6207 0.5949 0.5732 0.5660  obj[0..4]=0.0002 0.0000 0.0000 0.0000 0.0000 
  bbox[0,0] CHW: dx=0.7080 dy=0.5722 dw=0.4949 dh=0.5376
  bbox[0,0] IL:  dx=0.7080 dy=0.8980 dw=0.5351 dh=0.8584
  Stride 16 raw candidates: 8 (threshold=0.700)
  Top-5 scores: 0.8563 0.8521 0.8499 0.8476 0.8404
  Best candidate grid=(7,12) loc=152 score=0.8563
  bbox CHW[loc=152]: dx=0.2758 dy=0.1787 dw=0.3773 dh=0.4091
  bbox IL [loc=152]: dx=-0.6490 dy=0.4033 dw=1.5310 dh=2.1694
  -> CHW box: (185,103)-(208,127) 23x24
  -> IL  box: (145,48)-(219,188) 74x140
Stride 32 (gs=10): cls[0..4]=0.5638 0.5687 0.5522 0.5545 0.5833  obj[0..4]=0.0000 0.0000 0.0000 0.0000 0.0000 
  bbox[0,0] CHW: dx=0.6510 dy=0.5682 dw=0.9126 dh=0.3729
  bbox[0,0] IL:  dx=0.6510 dy=0.5452 dw=0.5559 dh=0.8735
  Stride 32 raw candidates: 0 (threshold=0.700)
  Top-5 scores: 0.0000 0.0000 0.0000 0.0000 0.0000
Total raw candidates (all strides): 8
Sorting 8 raw detections...
Top-5 raw detections (pre-NMS):
  #0: score=0.8521 box=(163.4,102.5,182.2,139.9) size=19x37
  #1: score=0.8499 box=(155.6,99.9,221.2,160.9) size=66x61
  #2: score=0.8476 box=(181.2,106.0,204.2,149.0) size=23x43
  #3: score=0.8257 box=(149.4,93.5,191.1,147.4) size=42x54
  #4: score=0.8245 box=(189.1,75.2,213.0,96.2) size=24x21
NMS: suppressed 4, remaining 4
=== Final detections: 4 ===
  #0: score=0.8563 box=(184.7,102.8)-(208.1,126.9) w=23 h=24
  #1: score=0.8521 box=(163.4,102.5)-(182.2,139.9) w=19 h=37
  #2: score=0.8404 box=(177.6,118.9)-(213.4,172.4) w=36 h=53
  #3: score=0.8245 box=(189.1,75.2)-(213.0,96.2) w=24 h=21
=== Final count: 4 faces ===
