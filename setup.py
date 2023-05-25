from setuptools import setup, find_packages

setup(
    name='draggan',
    packages=find_packages(),
    version='1.0.3',
    package_data={
        'draggan': ['stylegan2/op/fused_bias_act.cpp', 'stylegan2/op/upfirdn2d.cpp',
                    'stylegan2/op/fused_bias_act_kernel.cu', 'stylegan2/op/upfirdn2d_kernel.cu'], 
    },
    include_package_data=True,
    install_requires=[
        'gradio>=3.28',
        'tqdm',
        'torch>=1.8',
        'torchvision',
        'numpy',
        'ninja',
        'fire',
        'imageio',
        'imageio-ffmpeg',
        'scikit-image',
        'IPython',
        
    ]
)
