from setuptools import setup, find_packages

setup(
    name='thermal_colorization',
    version='1.0.0',
    description='Thermal IR to RGB Colorization for ISRO Hackathon 2026',
    author='Your Team',
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'numpy>=1.19.0',
        'opencv-python>=4.5.0',
        'rasterio>=1.2.0',
        'albumentations>=1.0.0',
        'tensorboard>=2.5.0',
        'tqdm>=4.60.0',
        'pyyaml>=5.4.0',
    ],
    python_requires='>=3.7',
)