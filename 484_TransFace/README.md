# 484_TransFace
[ICCV 2023] TransFace: Calibrating Transformer Training for Face Recognition from a Data-Centric Perspective

<img width="1541" height="889" alt="image" src="https://github.com/user-attachments/assets/0a7e3f75-7f07-467a-8375-47bd95330ff3" />

<img width="1748" height="1018" alt="image" src="https://github.com/user-attachments/assets/37718f1d-4066-4c4a-ae17-9ed6d9ed9054" />

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
--model glint360k_model_TransFace_S.onnx \
--images_dir images_out \
--landmarks_file images_out/ijb_landmarks.txt \
--output_markdown images_out/results_s.md \
--execution_provider cpu

python demo/demo_transface_onnx_cosine_similarity.py \
--model glint360k_model_TransFace_B.onnx \
--images_dir images_out \
--landmarks_file images_out/ijb_landmarks.txt \
--output_markdown images_out/results_b.md \
--execution_provider cpu

python demo/demo_transface_onnx_cosine_similarity.py \
--model glint360k_model_TransFace_L.onnx \
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
@inproceedings{dan2023transface,
  title={TransFace: Calibrating Transformer Training for Face Recognition from a Data-Centric Perspective},
  author={Dan, Jun and Liu, Yang and Xie, Haoyu and Deng, Jiankang and Xie, Haoran and Xie, Xuansong and Sun, Baigui},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={20642--20653},
  year={2023}
}
```
