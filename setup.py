from distutils.core import setup
setup(
  name = 'blackbeard_ds2109',         
  packages = ['blackbeard_ds2109'],   
  version = '0.1',    
  license='MIT',        
  description = '''This library is designed for people who need to optimize time in an agile way with an ease of understanding and could 
  solve the main projects you may have in Data Science, starting with cleaning dataframe (including images), visualization and machine learning.''',   
  author = 'DS2109FULLTIME',                   
  author_email = 'datascience2109thebridge@gmail.com',     
  url = 'https://github.com/ds2109fulltime/Blackbeard',   
  download_url = 'https://github.com/ds2109fulltime/Blackbeard/dist/blackbeard_ds2109-0.1.tar.gz',    
  keywords = ['machinelearning', 'scrum', 'datascience'],  
  install_requires=[            
          'tensorflow','xgboost','imageio','utllib','matplotlib','seaborn','scipy','requests','sys','collections',
          'sklearn','pandas','numpy','cv2','plotly','folium','IPython','datetime','calendar','itertools','itertools'
      ],
  classifiers=[
    'Development Status :: 4 - Beta',      
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   
    'Programming Language :: Python :: 3.7',      
    'Programming Language :: Python :: 3.8',
  ],
)
