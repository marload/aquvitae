from setuptools import setup, find_packages

setup(
    name="aquvitae",
    version="0.1.5",
    description="The easiest Knowledge Distillation library for Light Weight DeepLearning",
    author="marload",
    author_email="rladhkstn8@gmail.com",
    url="https://github.com/aquvitae/aquvitae",
    download_url="https://github.com/aquvitae/aquvitae/archive/v0.1.5.tar.gz",
    license="MIT",
    install_requires=[],
    setup_requires=["tqdm>=4.46.1", "pytorch-ignite>=0.4.0"],
    packages=find_packages(exclude=[]),
    keywords=["tensorflow", "pytorch", "pytorch-ignite", "tqdm"],
    python_requires=">=3",
    package_data={},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.2",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
