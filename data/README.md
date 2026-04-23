# Dataset layout

Each dataset uses the R2Gen-style JSON:

```
data/<dataset>/
├── annotations.json
└── images/
    └── ...
```

`annotations.json`:

```json
{
  "train": [
    {"id": "CXR1000_1",
     "image_path": ["1000_IM-0001-4001.png", "1000_IM-0001-3001.png"],
     "report": "The heart is normal in size. The lungs are clear. ..."},
    ...
  ],
  "val":  [...],
  "test": [...]
}
```

The existing R2Gen annotations for IU X-Ray / MIMIC-CXR follow this
exact schema. The CASIA-CXR release needs the French reports to be
assembled into the same JSON; see the paper's appendix (Section A.2)
for the split sizes. For CASIA-CXR you can additionally put a
`category` field on each entry and set the `CasiaCxrDataset` filter to
`"Pneumonia"` (default).
