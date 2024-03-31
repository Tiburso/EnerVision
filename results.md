## Versions

- 206347: 2-class; AUFL; weight - 0.5; delta - 0.85; gamma - 0.2; Backbone - `resnet101` (best)
- 206356: 2-class; AUFL; weight - 0.2; delta - 0.85; gamma - 0.7: Backbone - `resnet101`
<!-- - 206410: 2-class; AUFL; weight - 0.5; delta - 0.85; gamma - 0.7: Backbone - `resnet101` -->
- 206417: 2-class; AUFL; weight - 0.5; delta - 0.75; gamma - 0.2; Backbone - `resnet101`
- 206430: 2-class; AUFL; weight - 0.5; delta - 0.75; gamma - 0.2; Backbone - `resnet101`
- 206433: 2-class; AUFL; weight - 0.5; delta - 0.80; gamma - 0.2; Backbone - `resnet101`
- 206443: 2-class; AUFL; weight - 0.5; delta - 0.85; gamma - 0.4; Backbone - `resnet101`
- 206455: 2-class; AUFL; weight - 0.4; delta - 0.85; gamma - 0.1; Backbone - `resnet101` (2nd best)
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

- 206472: 2-class; AUFL; weight - 0.4; delta - 0.85; gamma - 0.1; Backbone - `resnet159`; lr = 5e-5
