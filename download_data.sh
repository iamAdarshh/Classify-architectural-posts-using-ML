#!/bin/bash

echo "Running download scripts..."
python download_scripts/architectural_posts.py
python download_scripts/programming_posts.py
echo "Download scripts completed."

echo "Installing Cython..."
pip install Cython
echo "Cython installed."

echo "Cloning gensim repository..."
git clone https://github.com/RaRe-Technologies/gensim.git
cd gensim

echo "Building the gensim package..."
python setup.py build_ext --inplace
python setup.py install
echo "Gensim package built and installed."

cd ..

echo "Installing dependencies from requirements.txt..."
pip install -r gensim/requirements.txt
echo "Dependencies installed."
