# 04 — Medical Vision-Language Models

## Overview

The previous three modules required training or fine-tuning a model on medical data. This
module asks a different question: how much can pre-trained **foundation models** do on
medical images *without any additional training*? The answer, it turns out, is quite a lot
— with important caveats.

This module covers three open-source multimodal large language models (MLLMs) that span
different capability/resource trade-offs: BiomedCLIP for zero-shot classification,
BLIP-2 for image captioning, and LLaVA-1.5 for open-ended visual question answering.
You will run all three on real medical image types, evaluate their outputs quantitatively,
and build a Gradio demo that ties them together.

---

## What you will learn

### The landscape of medical foundation models

There are three broad paradigms:

1. **Contrastive models (CLIP-style)** — learn a joint embedding space for images and
   text using a contrastive objective on image-caption pairs. At inference, embed candidate
   class labels as text and find the nearest one to the image embedding. No fine-tuning
   needed for zero-shot classification.

2. **Encoder-decoder captioning models (BLIP-2)** — a frozen vision encoder feeds into a
   lightweight trainable Q-Former, which in turn prompts a frozen LLM to generate text.
   Only the Q-Former (~188 M parameters) is trained; the ~9 B encoder + LLM are fixed.

3. **Aligned vision-language models (LLaVA)** — a vision encoder is linearly projected
   into the token embedding space of a full language model (LLaMA/Vicuna), and the whole
   system is fine-tuned on visual instruction data. This enables free-form conversation
   about images.

### BiomedCLIP: zero-shot classification

BiomedCLIP (Microsoft, 2023) is a CLIP model pre-trained on **15 million biomedical
image-text pairs** from PubMed Central — radiology reports, pathology paper figures,
microscopy captions, and more. Because it has seen diverse biomedical text, its text
encoder understands clinical terminology much better than general CLIP.

Zero-shot classification procedure:
```
Image ──► ImageEncoder (ViT-B/16) ──► image_embedding (512-d L2-normalised)
                                               │
                                        cosine_similarity
                                               │
Text prompts ──► TextEncoder ──► text_embeddings   ──► argmax ──► predicted class
```

The accuracy of zero-shot classification depends heavily on prompt engineering. The
notebook shows how rewording from "a photo of X" to "a clinical image of X showing Y"
substantially changes confidence scores.

### BLIP-2: medical image captioning

**BLIP-2** (Salesforce, 2023) introduces the **Querying Transformer (Q-Former)**, a
compact cross-attention module that extracts a fixed set of learned query vectors from
the image:

```
ViT-L (frozen, ~307 M params) ──► Q-Former (32 learned queries, cross-attention)
                                           │
                                     Projection layer
                                           │
                                   FlanT5-XL (frozen, ~3 B params) ──► Caption
```

Only the Q-Former and projection (~188 M total) are trained, which makes BLIP-2 far more
parameter-efficient than full fine-tuning. The model used here,
`Salesforce/blip2-flan-t5-xl`, occupies ~9 GB in fp16 — fits on T4 with careful management.

The notebook applies BLIP-2 to four image types: chest X-ray (CXR), brain MRI, retinal
fundus, and histopathology. You will see where the model produces plausible clinical
descriptions and where it generates anatomically confident but factually incorrect output —
an important safety calibration.

### LLaVA-1.5-7B: visual question answering

**LLaVA-1.5** (Liu et al., 2023) connects a CLIP ViT image encoder to a 7 B parameter
Vicuna language model via a two-layer MLP projection trained on visual instruction tuning
data:

```
CLIP ViT-L/336 ──► MLP projection ──► Vicuna-7B ──► Free-form answer
```

With **4-bit quantisation** (`bitsandbytes`), the 7 B model fits in ~5 GB of VRAM,
leaving room on T4 after the previous models are unloaded.

4-bit quantisation loads weights as int4 and performs computations in bfloat16, reducing
memory by ~4× at the cost of a small quality degradation. For inference-only use cases in
research (not deployment), this trade-off is almost always worthwhile.

VQA example queries covered in the notebook:
- "Is there any consolidation visible in this chest X-ray?"
- "What brain structure is highlighted in this MRI slice?"
- "Describe the tissue pattern visible in this histopathology image."

### Quantitative evaluation of generated text

Evaluating free-form generated text against a reference report requires metrics that go
beyond exact string matching:

| Metric | What it measures | Good when |
|--------|-----------------|-----------|
| **BERTScore** | Semantic similarity using pretrained BERT token embeddings | Paraphrases are acceptable (same meaning, different words) |
| **SacreBLEU** | Geometric mean of n-gram precision + brevity penalty | n-gram overlap matters (structured reports, formulaic text) |
| **ROUGE-L** | Longest common subsequence recall | Recall of key phrases is more important than precision |

For medical captioning, BERTScore is generally most informative because radiological
language allows many valid paraphrases ("no focal consolidation" ≡ "lungs are clear").

### VRAM management across multiple models

Running three large models in a single Colab session requires disciplined memory management.
After each model section:

```python
del model, processor          # Remove Python references
gc.collect()                  # Python garbage collector
torch.cuda.empty_cache()      # Release CUDA memory back to the allocator
```

Skipping any of these three steps will cause CUDA OOM when loading the next model. The
notebook enforces this pattern at the end of every model section.

### Building a Gradio demo

The notebook's final section wraps all three models in a **Gradio** interface that lets you
upload any medical image and query each model from a single web UI. This is the same
pattern used to prototype clinical decision-support tools before more robust deployment.

---

## Notebooks

| File | Description |
|------|-------------|
| `multimodal_medical_imaging_tutorial.ipynb` | GPU verification; downloading chest X-ray, brain MRI, retinal fundus, and histopathology sample images; BiomedCLIP zero-shot classification with prompt comparison; BLIP-2 captioning on all four image types; LLaVA-1.5 VQA with open-ended clinical queries; BERTScore + BLEU + ROUGE evaluation; interactive Gradio demo |

---

## Important caveats for clinical use

Foundation models trained on general biomedical literature are not validated clinical
devices. Key limitations to keep in mind:

- **Hallucination** — LLaVA and BLIP-2 will sometimes generate medically confident but
  factually incorrect statements. Always validate against ground-truth reports.
- **Distribution shift** — BiomedCLIP was trained on PubMed figures, which may differ from
  clinical acquisition protocols at your institution.
- **Prompt sensitivity** — small changes in prompt wording can dramatically change
  classification confidence. The notebook demonstrates this explicitly.
- **No regulatory approval** — none of these models carry FDA/CE clearance for diagnostic
  use. They are research tools.

---

## Setup

```bash
pip install "transformers>=4.37.0" "accelerate>=0.27.0" "bitsandbytes>=0.41.0" \
    open_clip_torch Pillow matplotlib seaborn requests datasets evaluate \
    bert-score sacrebleu einops "gradio>=4.20.0"
```

The notebook's install cell runs everything automatically on Colab.

---

## Tips for beginners

- Run BiomedCLIP before BLIP-2 and LLaVA. It requires only ~1 GB of VRAM and loads
  instantly, so it is a good warm-up before dealing with multi-GB model management.
- When BiomedCLIP gives low confidence on all classes, try reformulating the text prompt.
  "Fundus photograph showing drusen deposits" will score very differently from "retina."
- The Gradio demo cell is optional — skip it if you are only interested in the model
  evaluation results. It requires an active Colab session and opens a temporary public URL.
- If a model section returns CUDA OOM, restart the Colab runtime and run only that section
  — do not try to recover in-session after an OOM.
