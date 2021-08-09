from setuptools import setup, find_packages

setup(
    name="progen-transformer",
    packages=find_packages(),
    version="0.0.35",
    license="MIT",
    description="Protein Generation (ProGen)",
    author="Phil Wang",
    author_email="",
    url="https://github.com/lucidrains/progen",
    keywords=[
        "artificial intelligence",
        "deep learning",
        "protein language model"
    ],
    install_requires=[
        "click",
        "click-option-group",
        "cloudpickle",
        "einops>=0.3",
        "dagster",
        "dagit",
        "dm-haiku",
        "google-cloud-storage",
        "humanize",
        "jax",
        "jaxlib",
        "Jinja2",
        "jmp",
        "omegaconf",
        "optax>=0.0.9",
        "python-dotenv",
        "pyfaidx",
        "tensorflow",
        "tqdm",
        "wandb"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
