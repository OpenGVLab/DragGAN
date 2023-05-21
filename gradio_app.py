import gradio as gr
import torch
from drag_gan import stylegan2, drag_gan

device = 'cuda'
g_ema = stylegan2().to(device)


def to_image(tensor):
    tensor = tensor.squeeze(0).permute(1, 2, 0)
    arr = tensor.detach().cpu().numpy()
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    arr = arr * 255
    return arr.astype('uint8')


def on_click(image, target_point, points, evt: gr.SelectData):
    x = evt.index[1]
    y = evt.index[0]
    if target_point:
        image[x:x + 5, y:y + 5, :] = 255
        points['target'].append([evt.index[1], evt.index[0]])
        return image, str(evt.index)
    points['handle'].append([evt.index[1], evt.index[0]])
    image[x:x + 5, y:y + 5, :] = 0
    return image, str(evt.index)


def on_drag(points, max_iters, state):
    max_iters = int(max_iters)
    latent = state['latent']
    noise = state['noise']
    F = state['F']

    handle_points = [torch.tensor(p).float() for p in points['handle']]
    target_points = [torch.tensor(p).float() for p in points['target']]
    mask = torch.zeros((1, 1, 1024, 1024)).to(device)
    mask[..., 720:820, 390:600] = 1
    for sample2, latent, F in drag_gan(g_ema, latent, noise, F,
                                       handle_points, target_points, mask,
                                       max_iters=max_iters):
        points = {'target': [], 'handle': []}
        image = to_image(sample2)

        state['F'] = F
        state['latent'] = latent
        yield points, image, state


def main():
    torch.cuda.manual_seed(25)  
    sample_z = torch.randn([1, 512], device=device)
    latent, noise = g_ema.prepare([sample_z])
    sample, F = g_ema.generate(latent, noise)

    with gr.Blocks() as demo:
        state = gr.State({
            'latent': latent,
            'noise': noise,
            'F': F,
        })
        max_iters = gr.Slider(1, 100, 5, label='Max Iterations')
        image = gr.Image(to_image(sample)).style(height=512, width=512)
        text = gr.Textbox()
        btn = gr.Button('Drag it')
        points = gr.State({'target': [], 'handle': []})
        target_point = gr.Checkbox(label='Target Point')
        image.select(on_click, [image, target_point, points], [image, text])
        btn.click(on_drag, inputs=[points, max_iters, state], outputs=[points, image, state])

    demo.queue(concurrency_count=5, max_size=20).launch()


if __name__ == '__main__':
    main()
