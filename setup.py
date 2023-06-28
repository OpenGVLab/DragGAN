from setuptools import setup, find_packages

setup(
    name='draggan',
    packages=find_packages(),
    version='1.1.6',
    package_data={
        'draggan': ['deprecated/stylegan2/op/fused_bias_act.cpp', 
                    'deprecated/stylegan2/op/upfirdn2d.cpp',
                    'deprecated/stylegan2/op/fused_bias_act_kernel.cu',
                    'deprecated/stylegan2/op/upfirdn2d_kernel.cu',
                    'stylegan2/torch_utils/ops/bias_act.cpp', 
                    'stylegan2/torch_utils/ops/upfirdn2d.cpp',
                    'stylegan2/torch_utils/ops/bias_act.cu',
                    'stylegan2/torch_utils/ops/upfirdn2d.cu',
                    'stylegan2/torch_utils/ops/bias_act.h', 
                    'stylegan2/torch_utils/ops/upfirdn2d.h', 
                    ], 
    },
    include_package_data=True,
    install_requires=[
        'gradio==3.34.0',
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
