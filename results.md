## Versions

- 206463: 2-class; AUFL; weight - 0.4; delta - 0.85; gamma - 0.1; Backbone - `resnet101`; lr = 5e-5(reduced_dataset + resize)
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃ Test metric ┃ DataLoader 0 ┃
  ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
  │ val_dice │ 0.9938957095146179 │
  │ val_f1 │ 0.5941480994224548 │
  │ val_jaccard │ 0.5769420862197876 │
  │ val_loss │ 0.1746467649936676 │
  │ val_precision │ 0.6195605397224426 │
  │ val_recall │ 0.5823692083358765 │
  └───────────────────────────┴───────────────────────────┘

- 206472: 2-class; AUFL; weight - 0.4; delta - 0.9; gamma - 0.1; Backbone - `resnet159`; lr = 5e-5

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Test metric ┃ DataLoader 0 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ val_dice │ 0.9901595711708069 │
│ val_f1 │ 0.58010333776474 │
│ val_jaccard │ 0.5630457401275635 │
│ val_loss │ 0.17069466412067413 │
│ val_precision │ 0.5810942649841309 │
│ val_recall │ 0.5807406902313232 │
└───────────────────────────┴───────────────────────────┘

- 206477: 2-class; AUFL; weight - 0.4; delta - 0.9; gamma - 0.3; Backbone - `resnet159`; lr = 1e-4, weight_decay = 1e-4

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Test metric ┃ DataLoader 0 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ val_dice │ 0.9827537536621094 │
│ val_f1 │ 0.5712105631828308 │
│ val_jaccard │ 0.5505161881446838 │
│ val_loss │ 0.17660804092884064 │
│ val_precision │ 0.563201904296875 │
│ val_recall │ 0.587618350982666 │
└───────────────────────────┴───────────────────────────┘

- 207514: 2-class; AUFL; weight - 0.3; delta - 0.25; gamma - 2.0; Backbone - `resnet159`; lr = 1e-5

- 207515: 2-class; Combined; Backbone - `resnet159`; lr = 1e-2, weight_decay = 1e-3;

- 211504: 2-class; Combined; Backbone - `resnet159`; lr = 1e-4, weight_decay = 1e-3; both datasets

- 213100: 2-class; Combined; Backbone - `resnet159`; lr = 5e-5, weight_decay = 1e-4; solar_dk - 1000 training

- 222167 (stopped but godlike maybe finetune to the other dataset now): 2-class; Combined; Backbone - `resnet159`; lr = 1e-5, weight_decay = 1e-4;

  1. Run with those configs for mega nl-dataset for 5 epochs: 0.881 jaccard

  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃ Test metric ┃ DataLoader 0 ┃
  ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
  │ accuracy │ 0.9820424318313599 │
  │ dice │ 0.9820424318313599 │
  │ f1_score │ 0.9820424318313599 │
  │ jaccard_index │ 0.8813039064407349 │
  │ precision │ 0.9820424318313599 │
  │ recall │ 0.9820424318313599 │
  │ testing_loss │ 0.6043753027915955 │
  └───────────────────────────┴───────────────────────────┘

  2. Finetune to more accurate solar_dk_dataset for 150 epochs (232481)

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Test metric ┃ DataLoader 0 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ accuracy │ 0.997550368309021 │
│ dice │ 0.997550368309021 │
│ f1_score │ 0.997550368309021 │
│ jaccard_index │ 0.7843235731124878 │
│ precision │ 0.997550368309021 │
│ recall │ 0.997550368309021 │
│ testing_loss │ 0.008377181366086006 │
└───────────────────────────┴───────────────────────────┘

- 240425 (same as before) LR-Decrease is 0.9 on 5 Jaccard Index:
  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
  ┃ Test metric ┃ DataLoader 0 ┃
  ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
  │ accuracy │ 0.997889518737793 │
  │ dice │ 0.997889518737793 │
  │ f1_score │ 0.997889518737793 │
  │ jaccard_index │ 0.8465917110443115 │
  │ precision │ 0.997889518737793 │
  │ recall │ 0.997889518737793 │
  │ testing_loss │ 0.011348080821335316 │
  └───────────────────────────┴───────────────────────────┘

- 251115 (Not finished but 0.93 IOU validation)

- 251386 FocalLoss / 251495 CombinedLoss
