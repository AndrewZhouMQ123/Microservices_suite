# Microservices
## password_generator
## countdown_timer
## hash_function
## simple web scraping
## following Let's Build GPT tutorial video by Andrej Karpathy
https://www.youtube.com/watch?v=kCc8FmEb1nY
Implement based on Attention Is All You Need Paper

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.0.0 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

There are two solutions for this error:

1. downgrade your numpy to 1.26.4
pip install numpy==1.26.4
or

pip install "numpy<2.0" 