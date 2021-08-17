from setuptools import setup

setup(
    name='cc',
    version='0.1.0',
    py_modules=['cc'],
    install_requires=[
        'Click',
    ],
    entry_points={
        'console_scripts': [
            'cc = cc:remove_background',
        ],
    },
)
