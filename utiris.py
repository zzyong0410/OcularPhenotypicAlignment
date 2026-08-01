from train_controlnetwithInstantID import StableDiffusionControlNetIPAdapterPipeline,UNet_IPAdapter,IPAdapter
from diffusers import AutoencoderKL,UNet2DConditionModel,ControlNetModel,DDPMScheduler,UniPCMultistepScheduler
from diffusers.utils import load_image,make_image_grid
from transformers import CLIPTextModel,CLIPVisionModelWithProjection,AutoTokenizer,CLIPImageProcessor
from ip_adapter.ip_adapter import AttnProcessor,IPAttnProcessor,ImageProjModel
from ip_adapter.ip_adapter_faceid import LoRAIPAttnProcessor,LoRAAttnProcessor,MLPProjModel
from safetensors.torch import load_model,save_model
from PIL import Image
import torch
import os
from tqdm import tqdm
sd_pretrained_model = 'runwayml/stable-diffusion-v1-5'
image_encoder_path = 'h94/IP-Adapter'

def insert_ip_2_controlnet(controlnet):
    attn_procs = {}
    controlnet_dict = controlnet.state_dict()
    for name in controlnet.attn_processors.keys():
        cross_attention_dim = None if name.endswith('attn1.processor') else controlnet.config.cross_attention_dim
        if name.startswith('mid_block'):
            hidden_size = controlnet.config.block_out_channels[-1]
        else:
            block_id = int(name[len("down_blocks.")])
            hidden_size = controlnet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {"to_k_ip.weight": controlnet_dict[layer_name + '.to_k.weight'],
                       "to_v_ip.weight": controlnet_dict[layer_name + '.to_v.weight']}
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    controlnet.set_attn_processor(attn_procs)
    return controlnet


def insert_loarip_2_unet(unet):
    lora_rank = 128
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = LoRAIPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
            attn_procs[name].load_state_dict(weights,strict=False)
    unet.set_attn_processor(attn_procs)
    return unet


vae = AutoencoderKL.from_pretrained(sd_pretrained_model, subfolder="vae")
text_encoder = CLIPTextModel.from_pretrained(sd_pretrained_model, subfolder="text_encoder")
image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path,subfolder='models/image_encoder')
tokenizer = AutoTokenizer.from_pretrained(sd_pretrained_model,subfolder='tokenizer',revision=None,use_fast = False)
unet = UNet2DConditionModel.from_pretrained(sd_pretrained_model, subfolder = 'unet')



controlnet = ControlNetModel.from_unet(unet,conditioning_channels=6)
image_proj_model = ImageProjModel(cross_attention_dim=controlnet.config.cross_attention_dim,
                                  clip_embeddings_dim=image_encoder.config.projection_dim, clip_extra_context_tokens=4)

unet = insert_loarip_2_unet(unet)
controlnet = insert_ip_2_controlnet(controlnet)

unet_image_proj_model = MLPProjModel(
    cross_attention_dim=unet.config.cross_attention_dim,
    id_embeddings_dim=768,  # iris_embeddings_dim
    num_tokens=4,
)
adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

unet_ipadapter = UNet_IPAdapter(unet,unet_image_proj_model,adapter_modules)
controlnet_instantid = IPAdapter(controlnet,image_proj_model)
device = torch.device('cuda:1')
vae.to(device,dtype=torch.float16)
text_encoder.to(device,dtype=torch.float16)
image_encoder.to(device,dtype=torch.float16)
unet_ipadapter.to(device,dtype=torch.float16)

unet_ipadapter.eval()
controlnet_instantid.to(device,dtype=torch.float16)
controlnet.eval()

load_model(controlnet_instantid,'/zhouzhiyong/ns219x/projects/diffusers/examples/controlnet/instantid_vis2nir_utiris_depth_efu4_epoch200/checkpoint-4000/model.safetensors')
load_model(unet_ipadapter,'/zhouzhiyong/ns219x/projects/diffusers/examples/controlnet/instantid_vis2nir_utiris_depth_efu4_epoch200/checkpoint-4000/model_1.safetensors')
feature_extractor = CLIPImageProcessor()
clip_image_processor = CLIPImageProcessor()

scheduler = DDPMScheduler.from_pretrained(sd_pretrained_model, subfolder='scheduler')
# scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder='scheduler')
scheduler = UniPCMultistepScheduler.from_config(scheduler.config)
pipeline = StableDiffusionControlNetIPAdapterPipeline(vae=vae, text_encoder=text_encoder, image_encoder=image_encoder,
                                                      tokenizer=tokenizer, unet=unet_ipadapter, controlnet=controlnet_instantid,
                                                      scheduler=scheduler, feature_extractor=feature_extractor,
                                                      clip_image_processor=clip_image_processor)
vit_image_embeddings_model = "/zhouzhiyong/ns219x/projects/diffusers/examples/controlnet/id_emb_vit/utiris_vit_output_lefu4_fu5_fu6/checkpoint-120"


generator = torch.Generator(device="cuda:1").manual_seed(30)
def test_controlnetforpolyuiris(test_vis_dir,test_nir_dir,test_vis_canny_dir,test_vis_mask_dir,save_resulted_nir_dir,save_grid_result_dir):

    vis_imgs = sorted(os.listdir(test_vis_dir))
    nir_imgs = []
    for nir_img in os.listdir(test_nir_dir):
        if 'Imag' in nir_img:
            nir_img=nir_img.replace('Imag','Img')
        nir_imgs.append(nir_img)


    nir_imgs = sorted(nir_imgs)
    vis_cannys = sorted(os.listdir(test_vis_canny_dir))
    vis_masks =  sorted(os.listdir(test_vis_mask_dir))
    for vis,nir,vis_canny,vis_mask in tqdm(zip(vis_imgs,nir_imgs,vis_cannys,vis_masks)):
        id_label = nir.split('_')[1]

        if "L" in nir:

            prompt = 'Iris image of the left eye of a person acquired by near-infrared light with identity label {}. The iris image is of best quality and extremely detailed'.format(
                id_label)
        else:
            prompt = 'Iris image of the right eye of a person acquired by near-infrared light with identity label {}. The iris image is of best quality and extremely detailed'.format(
                id_label)
        vis_img = load_image(os.path.join(test_vis_dir,vis))
        vis_img_show = vis_img.resize((512,512),resample=Image.LANCZOS)

        nir_img = load_image(os.path.join(test_nir_dir,nir))
        nir_img_show = nir_img.resize((512,512),resample=Image.LANCZOS)

        vis_canny = load_image(os.path.join(test_vis_canny_dir,vis_canny))
        vis_mask = load_image(os.path.join(test_vis_mask_dir,vis_mask))
        image_generated = \
        pipeline(vis_img, prompt, generator, num_images_per_prompt=1, device=torch.device('cuda:1'),
                num_inference_steps=20, canny_image=vis_canny, mask_image=vis_mask,
                vit_image_embeddings_model=vit_image_embeddings_model).images[0]

        grid = make_image_grid([nir_img_show, vis_img_show, image_generated], rows=1, cols=3)
        if not os.path.exists(save_grid_result_dir):
            os.makedirs(save_grid_result_dir)
        if not os.path.exists(save_resulted_nir_dir):
            os.makedirs(save_resulted_nir_dir)
        image_generated.save(save_resulted_nir_dir+'/{}.png'.format(os.path.splitext(vis)[0]))
        grid.save(save_grid_result_dir+'/grid_result_{}.png'.format(os.path.splitext(vis)[0]))


if __name__=='__main__':
    test_vis_dir ='/zhouzhiyong/ns219x/projects/diffusers/examples/controlnet/dataset/UTIRIS/train/VIS'
    test_nir_dir = '/zhouzhiyong/ns219x/projects/diffusers/examples/controlnet/dataset/UTIRIS/train/NIR'
    test_vis_canny_dir = '/zhouzhiyong/ns219x/projects/diffusers/examples/controlnet/dataset/UTIRIS/train/vis_depth'
    test_vis_mask_dir = '/zhouzhiyong/ns219x/projects/diffusers/examples/controlnet/dataset/UTIRIS/train/vis_mask'
    save_resulted_nir_dir = '/zhouzhiyong/ns219x/projects/diffusers/examples/controlnet/instantid_vis2nir_utiris_depth_efu4_epoch200/trainset_resulted_epoch4000/nir_generated'
    save_grid_result_dir = '/zhouzhiyong/ns219x/projects/diffusers/examples/controlnet/instantid_vis2nir_utiris_depth_efu4_epoch200/trainset_resulted_epoch4000/comparative_grid'
    test_controlnetforpolyuiris(test_vis_dir,test_nir_dir,test_vis_canny_dir,test_vis_mask_dir,save_resulted_nir_dir,save_grid_result_dir)
