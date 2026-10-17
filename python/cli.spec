# -*- mode: python ; coding: utf-8 -*-

import imagecodecs

hiddenimports = ["imagecodecs." + x for x in imagecodecs._extensions()]

a = Analysis(
    ['./ouroboros/cli.py'],
    pathex=[],
    binaries=[],
    datas=[('ouroboros/common/licenses/psutil.txt', 'licenses')],
    hiddenimports=hiddenimports + ['scipy._lib.messagestream', 'scipy', 'scipy.signal', 'scipy.signal.bsplines', 'scipy.special', 'scipy.special._ufuncs_cxx',
                            'scipy.linalg.cython_blas',
                            'scipy.linalg.cython_lapack',
                            'scipy.integrate',
                            'scipy.integrate.quadrature',
                            'scipy.integrate.odepack',
                            'scipy.integrate._odepack',
                            'scipy.integrate.quadpack',
                            'scipy.integrate._quadpack',
                            'scipy.integrate._ode',
                            'scipy.integrate.vode',
                            'scipy.special._special_ufuncs',
                            'scipy.integrate._dop', 'scipy._lib', 'scipy._build_utils','scipy.__config__', 'scipy._lib.array_api_compat.numpy.fft',
                            'scipy.integrate.lsoda', 'scipy.cluster', 'scipy.constants','scipy.fftpack','scipy.interpolate','scipy.io','scipy.linalg','scipy.misc','scipy.ndimage','scipy.odr','scipy.optimize','scipy.setup','scipy.sparse','scipy.spatial','scipy.special','scipy.stats','scipy.version'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ouroboros-cli',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
