# 483_LVFace
[ICCV 2025 Highlight] Official open-source repo for LVFace: Progressive Cluster Optimization for Large Vision Models in Face Recognition

<img width="2381" height="698" alt="image" src="https://github.com/user-attachments/assets/1d665e1e-8976-4f00-a8e6-87ef5fe1c617" />

## Face crop

```bash
python demo/demo_deimv2_onnx_wholebody34_with_edges.py \
-m deimv2_dinov3_x_wholebody34_680query_n_batch_640x640.onnx \
-ep cuda \
-i images \
--face_crop_output_dir images_out \
-dwk
```

## Batch calculation of Cosine similarity

```bash
python demo/demo_transface_onnx_cosine_similarity.py \
--model LVFace-T_Glint360K.onnx \
--images_dir images_out \
--landmarks_file images_out/ijb_landmarks.txt \
--output_markdown images_out/results_t.md \
--execution_provider cpu

python demo/demo_transface_onnx_cosine_similarity.py \
--model LVFace-S_Glint360K.onnx \
--images_dir images_out \
--landmarks_file images_out/ijb_landmarks.txt \
--output_markdown images_out/results_s.md \
--execution_provider cpu

python demo/demo_transface_onnx_cosine_similarity.py \
--model LVFace-B_Glint360K.onnx \
--images_dir images_out \
--landmarks_file images_out/ijb_landmarks.txt \
--output_markdown images_out/results_b.md \
--execution_provider cpu

python demo/demo_transface_onnx_cosine_similarity.py \
--model LVFace-L_Glint360K.onnx \
--images_dir images_out \
--landmarks_file images_out/ijb_landmarks.txt \
--output_markdown images_out/results_l.md \
--execution_provider cpu
```

## Convert Markdown results to a self-contained HTML file

```bash
python demo/demo_transface_onnx_cosine_similarity.py \
--input_markdown images_out/results_l.md \
--output_html images_out/results_l.html
```

## Citation
```bibtex
@inproceedings{you2025lvface,
  title={{LVFace}: Progressive Cluster Optimization for Large Vision Models in Face Recognition},
  author={You, Jinghan and Li, Shanglin and Sun, Yuanrui and Wei, Jiangchuan and Guo, Mingyu and Feng, Chao and Ran, Jiao},
  booktitle={ICCV},
  year={2025}
}
```
