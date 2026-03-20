# src/distillation/ — Knowledge Distillation Module

## Role
Transfer specific capabilities from a large model (Teacher) to a small model (Student).
Fully utilized from Phase 3, but the interface is designed in Phase 1.

## Core Files

### teacher.py
- Teacher model wrapper: inference + soft label generation
- Supports large model API calls (GPT-4, Claude, etc.)
- Output caching: prevent duplicate API calls for identical inputs

### student.py
- Student model architecture definition and training
- Simultaneous learning from Teacher's soft labels and hard labels (KD loss)
- Architecture search: configurable encoder/decoder depth, hidden dim, etc.

### pipeline.py
- Full Distillation pipeline orchestration
- Automates: data generation → Teacher inference → Student training → evaluation

## KD Loss Formula
```
L = alpha * KL(softmax(teacher_logits/T), softmax(student_logits/T)) * T^2
  + (1-alpha) * CrossEntropy(student_logits, hard_labels)
```
- alpha: teacher loss weight (default 0.7)
- T: temperature (default 4.0)
- These values must be configured via config YAML

## Rules
- Teacher model inference results (logits/translations) must be cached
- Student training must use validation set for early stopping
- Training logs saved as JSON Lines in `data/results/distillation/`
- Data preparation/preprocessing stages must be runnable without GPU
