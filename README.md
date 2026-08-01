# OcularPhenotypicAlignment
![Overview](overview_01.png)
## 📂 Dataset
| Dataset | Download |
|---------|----------|
| PolyU | [web](https://www4.comp.polyu.edu.hk/~csajaykr/polyuiris.htm) |
| Utiris | [dropbox](https://www.dropbox.com/scl/fi/z36qfhyzqum9n3slwta4t/UTIRIS_V.1.zip?rlkey=9w2hkphjuhwe4lruc0yicu0r8&dl=0) |
| CASIA | [web](http://www.cbsr.ia.ac.cn/english/NIR-VIS-2.0-Database.html) |
## 📂 Model Checkpoint
| Model | Dataset | Download |
|-------|---------|----------|
| Ours | PolyU | [Google Drive](https://drive.google.com/xxxx) |
| Ours | Utiris | [Google Drive](https://drive.google.com/yyyy) |
| Ours | CASIA | [Google Drive](https://drive.google.com/zzzz) |

## 🚀 Running Examples
## 🖥️ Environment
| Package | Version |
|---------|---------|
| Python | 3.10 |
| PyTorch | 2.1.2 |
| torchvision | 0.16.2 |
| diffusers | 0.27.2 |
| transformers | 4.38.2 |
| accelerate | 0.27.2 |
| xformers | 0.0.23 |
| numpy | 1.26.4 |
| Pillow | 10.2.0 |
| OpenCV | 4.9.0 |
| safetensors | 0.4.2 |

### Evaluation
Evaluate the pretrained model:

```bash
python casia.py \
    --weights checkpoints/yourpath.pth
```
```bash
python polyu.py \
    --weights checkpoints/yourpath.pth
```
```bash
python utiris.py \
    --weights checkpoints/yourpath.pth
```
