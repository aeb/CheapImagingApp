# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['./src/ngehtapp.py'],
             pathex=['../src'],
             binaries=[],
             datas=[],
             hiddenimports=['pkg_resources.py2_warn'],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=['_tkinter','Tkinter','enchant','twisted'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts, 
          [],
          exclude_binaries=True,
          name='ngehtapp',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True)
coll = COLLECT(exe,Tree('../src'),
               a.binaries,
               a.zipfiles,
               a.datas, 
               strip=False,
               upx=True,
               upx_exclude=[],
               name='ngehtapp')
app = BUNDLE(coll,
             name='ngehtapp.app',
             icon='./src/images/ngeht.ico',
             bundle_identifier=None,
             info_plist={'NSHighResolutionCapable': 'True'})
