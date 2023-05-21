import gradio as gr
import torch

from drag_gan import drag_gan, stylegan2

device = 'cuda'


SIZE_TO_CLICK_SIZE = {
    1024: 5,
    256: 2
}

CKPT_SIZE = {
    'stylegan2-ffhq-config-f.pt': 1024,
    'stylegan2-cat-config-f.pt': 256,
    'stylegan2-church-config-f.pt': 256,
    'stylegan2-horse-config-f.pt': 256,
}


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


def on_drag(model, points, max_iters, state, size):
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
    mask = torch.zeros((1, 1, 1024, 1024)).to(device)
    mask[..., 720:820, 390:600] = 1
    for sample2, latent, F in drag_gan(model.g_ema, latent, noise, F,
                                       handle_points, target_points, mask,
                                       max_iters=max_iters):
        image = to_image(sample2)

        state['F'] = F
        state['latent'] = latent
        state['sample'] = sample2
        # yield points, image, state, gr.Tabs.update(selected='output')
        add_points_to_image(image, points, size=SIZE_TO_CLICK_SIZE[size])
        yield image, state


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


def on_save():
    pass


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
        'sample': sample
    }
    return model, state, to_image(sample), size


def on_new_image(model):
    g_ema = model.g_ema
    sample_z = torch.randn([1, 512], device=device)
    latent, noise = g_ema.prepare([sample_z])
    sample, F = g_ema.generate(latent, noise)

    state = {
        'latent': latent,
        'noise': noise,
        'F': F,
        'sample': sample
    }
    return to_image(sample), state


def main():
    torch.cuda.manual_seed(25)

    with gr.Blocks() as demo:
        wrapped_model = ModelWrapper()
        model = gr.State(wrapped_model)
        sample_z = torch.randn([1, 512], device=device)
        latent, noise = wrapped_model.g_ema.prepare([sample_z])
        sample, F = wrapped_model.g_ema.generate(latent, noise)

        gr.Markdown(
            """
            # DragGAN (Unofficial)
            
            Unofficial implementation of [Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold](https://github.com/XingangPan/DragGAN) 
            
            [https://github.com/Zeqiang-Lai/DragGAN](https://github.com/Zeqiang-Lai/DragGAN)
      
            """,
        )
        state = gr.State({
            'latent': latent,
            'noise': noise,
            'F': F,
            'sample': sample
        })
        points = gr.State({'target': [], 'handle': []})
        size = gr.State(1024)

        with gr.Row():
            with gr.Column(scale=0.3):
                with gr.Accordion("Model"):
                    model_dropdown = gr.Dropdown(choices=list(CKPT_SIZE.keys()), value='stylegan2-ffhq-config-f.pt',
                                                 label='StyleGAN2 model')
                    max_iters = gr.Slider(1, 100, 20, step=1, label='Max Iterations')
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
                        # copy_btn = gr.Button('Start from Output')
                        btn = gr.Button('Drag it', variant='primary')

                with gr.Accordion('Save'):
                    with gr.Row():
                        filename = gr.Textbox("draggan")
                    with gr.Row():
                        save_video_btn = gr.Button('Save Video')
                        save_cfg_btn = gr.Button('Save Config')
                        save_img_btn = gr.Button('Save Image', variant='primary')
            with gr.Column(scale=0.8):
                # with gr.Tabs() as tabs:
                with gr.Tab('Input', id='input'):
                    image = gr.Image(to_image(sample)).style(height=768, width=768)
                    # with gr.Tab('Ouput', id='output'):
                    #     output_image = gr.Image(to_image(sample)).style(height=768, width=768)
        image.select(on_click, [image, target_point, points, size], [image, text, target_point])
        btn.click(on_drag, inputs=[model, points, max_iters, state, size], outputs=[image, state])
        reset_btn.click(on_reset, inputs=[points, image, state], outputs=[points, image])
        undo_btn.click(on_undo, inputs=[points, image, state, size], outputs=[points, image])
        model_dropdown.change(on_change_model, inputs=[model_dropdown, model], outputs=[model, state, image, size])
        new_btn.click(on_new_image, inputs=[model], outputs=[image, state])
    return demo


if __name__ == '__main__':
    demo = main()
    demo = demo.queue(concurrency_count=1, max_size=20).launch(share=True)
