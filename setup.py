import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
  name = 'blackbeard2109',         
  version = '0.2.1',    
  author = 'The Bridge Data Science Team 2109',                   
  author_email = 'datascience2109thebridge@gmail.com',
  description = '''This library is designed for people who need to optimize time in an agile way with an ease of understanding and could 
  solve the main projects you may have in Data Science, starting with cleaning dataframe (including images), visualization and machine learning.''',   
  long_description=long_description,
  long_description_content_type="text/markdown",
  url = 'https://github.com/ds2109fulltime/BLACKBEARD',   
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  package_dir={"": "src"},
  packages=setuptools.find_packages(where="src"),
  python_requires=">=3.6",
)
