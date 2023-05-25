import os
import gradio as gr
import torch
import numpy as np
import imageio
from PIL import Image
import uuid

from drag_gan import drag_gan, stylegan2
from stylegan2.inversion import inverse_image

device = 'cuda'


SIZE_TO_CLICK_SIZE = {
    1024: 8,
    512: 5,
    256: 2
}

CKPT_SIZE = {
    'stylegan2-ffhq-config-f.pt': 1024,
    'stylegan2-cat-config-f.pt': 256,
    'stylegan2-church-config-f.pt': 256,
    'stylegan2-horse-config-f.pt': 256,
    'ada/ffhq.pt': 1024,
    'ada/afhqcat.pt': 512,
    'ada/afhqdog.pt': 512,
    'ada/afhqwild.pt': 512,
    'ada/brecahad.pt': 512,
    'ada/metfaces.pt': 512,
}

DEFAULT_CKPT = 'stylegan2-ffhq-config-f.pt'


class grImage(gr.components.Image):
    is_template = True

    def preprocess(self, x):
        if x is None:
            return x
        if self.tool == "sketch" and self.source in ["upload", "webcam"]:
            decode_image = gr.processing_utils.decode_base64_to_image(x)
            width, height = decode_image.size
            mask = np.zeros((height, width, 4), dtype=np.uint8)
            mask[..., -1] = 255
            mask = self.postprocess(mask)
            x = {'image': x, 'mask': mask}
        return super().preprocess(x)


class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", tool="sketch", interactive=True, **kwargs)

    def preprocess(self, x):
        if x is None:
            return x
        if self.tool == "sketch" and self.source in ["upload", "webcam"] and type(x) != dict:
            decode_image = gr.processing_utils.decode_base64_to_image(x)
            width, height = decode_image.size
            mask = np.zeros((height, width, 4), dtype=np.uint8)
            mask[..., -1] = 255
            mask = self.postprocess(mask)
            x = {'image': x, 'mask': mask}
        return super().preprocess(x)


class ModelWrapper:
    def __init__(self, **kwargs):
        self.g_ema = stylegan2(**kwargs).to(device)


def to_image(tensor):
    tensor = tensor.squeeze(0).permute(1, 2, 0)
    arr = tensor.detach().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = arr * 255
    return arr.astype('uint8')


def add_points_to_image(image, points, size=5):
    h, w, = image.shape[:2]

    for x, y in points['target']:
        image[max(0, x - size):min(x + size, h - 1), max(0, y - size):min(y + size, w), :] = [255, 0, 0]
    for x, y in points['handle']:
        image[max(0, x - size):min(x + size, h - 1), max(0, y - size):min(y + size, w), :] = [0, 0, 255]

    return image


def on_click(image, target_point, points, size, evt: gr.SelectData):
    if target_point:
        points['target'].append([evt.index[1], evt.index[0]])
        image = add_points_to_image(image, points, size=SIZE_TO_CLICK_SIZE[size])
        return image, str(evt.index), not target_point
    points['handle'].append([evt.index[1], evt.index[0]])
    image = add_points_to_image(image, points, size=SIZE_TO_CLICK_SIZE[size])
    return image, str(evt.index), not target_point


def on_drag(model, points, max_iters, state, size, mask):
    if len(points['handle']) == 0:
        raise gr.Error('You must select at least one handle point and target point.')
    if len(points['handle']) != len(points['target']):
        raise gr.Error('You have uncompleted handle points, try to selct a target point or undo the handle point.')
    max_iters = int(max_iters)
    latent = state['latent']
    noise = state['noise']
    F = state['F']

    handle_points = [torch.tensor(p).float() for p in points['handle']]
    target_points = [torch.tensor(p).float() for p in points['target']]

    if mask.get('mask') is not None:
        mask = Image.fromarray(mask['mask']).convert('L')
        mask = np.array(mask) == 255

        mask = torch.from_numpy(mask).float().to(device)
        mask = mask.unsqueeze(0).unsqueeze(0)
    else:
        mask = None

    step = 0
    for sample2, latent, F, handle_points in drag_gan(model.g_ema, latent, noise, F,
                                                      handle_points, target_points, mask,
                                                      max_iters=max_iters):
        image = to_image(sample2)

        state['F'] = F
        state['latent'] = latent
        state['sample'] = sample2
        points['handle'] = [p.cpu().numpy().astype('int') for p in handle_points]
        add_points_to_image(image, points, size=SIZE_TO_CLICK_SIZE[size])

        state['history'].append(image)
        step += 1
        yield image, state, step


def on_reset(points, image, state):
    return {'target': [], 'handle': []}, to_image(state['sample'])


def on_undo(points, image, state, size):
    image = to_image(state['sample'])

    if len(points['target']) < len(points['handle']):
        points['handle'] = points['handle'][:-1]
    else:
        points['handle'] = points['handle'][:-1]
        points['target'] = points['target'][:-1]

    add_points_to_image(image, points, size=SIZE_TO_CLICK_SIZE[size])
    return points, image


def on_change_model(selected, model):
    size = CKPT_SIZE[selected]
    model = ModelWrapper(size=size, ckpt=selected)
    g_ema = model.g_ema
    sample_z = torch.randn([1, 512], device=device)
    latent, noise = g_ema.prepare([sample_z])
    sample, F = g_ema.generate(latent, noise)

    state = {
        'latent': latent,
        'noise': noise,
        'F': F,
        'sample': sample,
        'history': []
    }
    return model, state, to_image(sample), to_image(sample), size


def on_new_image(model):
    g_ema = model.g_ema
    sample_z = torch.randn([1, 512], device=device)
    latent, noise = g_ema.prepare([sample_z])
    sample, F = g_ema.generate(latent, noise)

    state = {
        'latent': latent,
        'noise': noise,
        'F': F,
        'sample': sample,
        'history': []
    }
    points = {'target': [], 'handle': []}
    target_point = False
    return to_image(sample), to_image(sample), state, points, target_point


def on_max_iter_change(max_iters):
    return gr.update(maximum=max_iters)


def on_save_files(image, state):
    os.makedirs('tmp', exist_ok=True)
    image_name = f'tmp/image_{uuid.uuid4()}.png'
    video_name = f'tmp/video_{uuid.uuid4()}.mp4'
    imageio.imsave(image_name, image)
    imageio.mimsave(video_name, state['history'])
    return [image_name, video_name]


def on_show_save():
    return gr.update(visible=True)


def on_image_change(model, image_size, image):
    image = Image.fromarray(image)
    result = inverse_image(
        model.g_ema,
        image,
        image_size=image_size
    )
    result['history'] = []
    image = to_image(result['sample'])
    points = {'target': [], 'handle': []}
    target_point = False
    return image, image, result, points, target_point


def on_mask_change(mask):
    return mask['image']


def main():
    torch.cuda.manual_seed(25)

    with gr.Blocks() as demo:
        wrapped_model = ModelWrapper(ckpt=DEFAULT_CKPT, size=CKPT_SIZE[DEFAULT_CKPT])
        model = gr.State(wrapped_model)
        sample_z = torch.randn([1, 512], device=device)
        latent, noise = wrapped_model.g_ema.prepare([sample_z])
        sample, F = wrapped_model.g_ema.generate(latent, noise)

        gr.Markdown(
            """
            # DragGAN
            
            Unofficial implementation of [Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold](https://vcai.mpi-inf.mpg.de/projects/DragGAN/)
            
            [Our Implementation](https://github.com/Zeqiang-Lai/DragGAN) | [Official Implementation](https://github.com/XingangPan/DragGAN) (Not released yet)

            ## Tutorial
            
            1. (Optional) Draw a mask indicate the movable region.
            2. Setup a least one pair of handle point and target point.
            3. Click "Drag it". 
            
            ## Hints
            
            - Handle points (Blue): the point you want to drag.
            - Target points (Red): the destination you want to drag towards to.
            
            ## Primary Support of Custom Image.
            
            - We now support dragging user uploaded image by GAN inversion.
            - **Please upload your image at `Setup Handle Points` pannel.** Upload it from `Draw a Mask` would cause errors for now.
            - Due to the limitation of GAN inversion, 
                - You might wait roughly 1 minute to see the GAN version of the uploaded image.
                - The shown image might be slightly difference from the uploaded one.
                - It could also fail to invert the uploaded image and generate very poor results.
                - Idealy, you should choose the closest model of the uploaded image. For example, choose `stylegan2-ffhq-config-f.pt` for human face. `stylegan2-cat-config-f.pt` for cat.
                
            > Please fire an issue if you have encounted any problem. Also don't forgot to give a star to the [Official Repo](https://github.com/XingangPan/DragGAN), [our project](https://github.com/Zeqiang-Lai/DragGAN) could not exist without it.
            """,
        )
        state = gr.State({
            'latent': latent,
            'noise': noise,
            'F': F,
            'sample': sample,
            'history': []
        })
        points = gr.State({'target': [], 'handle': []})
        size = gr.State(CKPT_SIZE[DEFAULT_CKPT])

        with gr.Row():
            with gr.Column(scale=0.3):
                with gr.Accordion("Model"):
                    model_dropdown = gr.Dropdown(choices=list(CKPT_SIZE.keys()), value=DEFAULT_CKPT,
                                                 label='StyleGAN2 model')
                    max_iters = gr.Slider(1, 500, 20, step=1, label='Max Iterations')
                    new_btn = gr.Button('New Image')
                with gr.Accordion('Drag'):
                    with gr.Row():
                        with gr.Column(min_width=100):
                            text = gr.Textbox(label='Selected Point', interactive=False)
                        with gr.Column(min_width=100):
                            target_point = gr.Checkbox(label='Target Point', interactive=False)
                    with gr.Row():
                        with gr.Column(min_width=100):
                            reset_btn = gr.Button('Reset All')
                        with gr.Column(min_width=100):
                            undo_btn = gr.Button('Undo Last')
                    with gr.Row():
                        btn = gr.Button('Drag it', variant='primary')

                with gr.Accordion('Save', visible=False) as save_panel:
                    files = gr.Files(value=[])

                progress = gr.Slider(value=0, maximum=20, label='Progress', interactive=False)

            with gr.Column():
                with gr.Tabs():
                    with gr.Tab('Draw a Mask', id='mask'):
                        mask = ImageMask(value=to_image(sample), label='Mask').style(height=768, width=768)
                    with gr.Tab('Setup Handle Points', id='input'):
                        image = grImage(to_image(sample)).style(height=768, width=768)

        image.select(on_click, [image, target_point, points, size], [image, text, target_point])
        image.upload(on_image_change, [model, size, image], [image, mask, state, points, target_point])
        mask.upload(on_mask_change, [mask], [image])
        btn.click(on_drag, inputs=[model, points, max_iters, state, size, mask], outputs=[image, state, progress]).then(
            on_show_save, outputs=save_panel).then(
            on_save_files, inputs=[image, state], outputs=[files]
        )
        reset_btn.click(on_reset, inputs=[points, image, state], outputs=[points, image])
        undo_btn.click(on_undo, inputs=[points, image, state, size], outputs=[points, image])
        model_dropdown.change(on_change_model, inputs=[model_dropdown, model], outputs=[model, state, image, mask, size])
        new_btn.click(on_new_image, inputs=[model], outputs=[image, mask, state, points, target_point])
        max_iters.change(on_max_iter_change, inputs=max_iters, outputs=progress)
    return demo


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('-p', '--port', default=None)
    parser.add_argument('--ip', default=None)
    args = parser.parse_args()
    device = args.device
    demo = main()
    demo.queue(concurrency_count=1, max_size=20).launch(share=args.share, server_name=args.ip, server_port=args.port)
