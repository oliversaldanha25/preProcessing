# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['path-to\multi_function_low_size.py'],
             pathex=['C:\\Users\\User\\Documents\\openslide-win64-20171122\\bin\\'],
             binaries=[],
             datas=[('C:\\Users\\User\\Documents\\openslide-win64-20171122\\bin\\images\\*','images')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=['pandas','slearn','tensorflow','absl-py','alabaster','absl-py','altgraph','argon2-cffi','astunparse','async-generator',
'atomicwrites','attrs','Babel','backcall','bleach','cachetools','certifi','cffi','chardet','colorama','confuse','contextvars','Cython','decorator',
'defusedxml','diff-match-patch','docutils','entrypoints','future','gast','google-auth','google-auth-oauthlib','google-pasta','grpcio','h5py','htmlmin',
'idna','ijson','ImageHash','ipykernel','ipython','ipython-genutils','ipywidgets','jedi','Jinja2','joblib','jsonschema','jupyter-client','jupyter-core',
'jupyterlab-pygments','Keras','Keras-Applications','Keras-Preprocessing','llvmlite','Markdown','MarkupSafe','missingno','mistune','nbclient','nbconvert',
'nbformat','nest-asyncio','networkx','notebook','numba','numpydoc','oauthlib','opt-einsum','packaging','pandas-profiling','pandocfilters','parso','pefile',
'phik','pickleshare','prometheus-client','protobuf','pyasn1','pyasn1-modules','pycparser','Pygments','pyinstaller','pyinstaller-hooks-contrib','PyNaCl',
'pyrsistent','python-dateutil','pytz','PyWavelets','pywin32','pywin32-ctypes','pywinpty','PyYAML','pyzbar','pyzmq','requests-oauthlib','pandas-profiling',
'pandocfilters','rsa','scikit-learn','sklearn','snowballstemmer','Sphinx','tensorboard','tensorboard-plugin-wit','tensorflow-cpu','tensorflow-estimator',
'tensorflow-gpu-estimator','tqdm','traitlets','urllib3','visions','wcwidth','webencodings','Werkzeug',
'wheel','widgetsnbextension','wrapt','zip'                                            ,
'sphinxcontrib-applehelp','sphinxcontrib-devhelp','sphinxcontrib-htmlhelp','sphinxcontrib-jsmath','sphinxcontrib-qthelp','sphinxcontrib-serializinghtml','pip'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='Multi_function_app_with_thumb_extaction_version_0.1',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True , icon='C:/Users/User/Documents/openslide-win64-20171122/bin/images/icon.ico')
