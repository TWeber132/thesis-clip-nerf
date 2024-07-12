from setuptools import setup, find_packages

setup(
    name='manipulation_tasks',
    version='0.1.0',
    description='A Python package for modeling robotic manipulation tasks',
    url='https://www.w.hs-karlsruhe.de/gitlab/iras/research-projects/ki5grob/manipulation_tasks',
    author='Gergely Soti',
    author_email='gergely.soti@h-ka.de',
    license='BSD 2-clause',
    packages=find_packages(),
    install_requires=['numpy',
                      'scipy',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8 ',
    ],
)
