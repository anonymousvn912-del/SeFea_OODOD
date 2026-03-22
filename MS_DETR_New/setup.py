from setuptools import setup, find_packages

setup(
    name="MS_DETR_New",
    version="0.1.0",
    description="A project to finetune CLIP for content-based image retrieval",
    packages=find_packages(exclude=["data", "exps*", "obj_spe_features*", "visualization*"]),
    python_requires=">=3.7",
)
