"""Set up a Google Colab runtime for the TP-MLIP tutorials.

Each notebook runs this file with `%run /content/TP-MLIP/colab_setup.py` in its first cell.
It installs n2p2, CP2K and the Python packages, makes `import aml` work and defines `run_cp2k()` and `show_md()`.
Running it again is cheap: anything already installed is skipped.
"""

import glob
import importlib.metadata
import math
import os
import site
import subprocess
import sys
import time

TP_DIR = os.path.dirname(os.path.abspath(__file__))

# n2p2 comes prebuilt from conda-forge, in its own prefix so that Colab's Python is left untouched
MAMBA = '/content/micromamba/bin/micromamba'
NNP_PREFIX = '/content/n2p2'

CP2K_VERSION = '2024.1'
CP2K_EXE = f'/content/cp2k/cp2k-{CP2K_VERSION}-Linux-gnu-x86_64.ssmp'
CP2K_URL = f'https://github.com/cp2k/cp2k/releases/download/v{CP2K_VERSION}/{os.path.basename(CP2K_EXE)}'

# only these are exposed on PATH, the rest of the n2p2 prefix (its own python, etc.) stays hidden
BIN_DIR = '/content/bin'
NNP_TOOLS = ['nnp-scaling', 'nnp-train', 'nnp-dataset', 'nnp-convert', 'nnp-predict', 'mpirun']


def _sh(cmd):
    # Colab does not show the output of child processes, so print it ourselves if the command fails
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-3000:], result.stderr[-3000:], sep='\n')
        raise subprocess.CalledProcessError(result.returncode, cmd)


def _install_n2p2():
    if os.path.exists(f'{NNP_PREFIX}/bin/nnp-train'):
        return
    print('Installing n2p2...', flush=True)
    _sh('mkdir -p /content/micromamba && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest'
        ' | tar -xj -C /content/micromamba bin/micromamba')
    _sh(f'{MAMBA} create -q -y -p {NNP_PREFIX} -c conda-forge n2p2 > /dev/null')


def _install_cp2k():
    if os.path.exists(CP2K_EXE):
        return
    print(f'Installing CP2K {CP2K_VERSION}...', flush=True)
    _sh(f'mkdir -p {os.path.dirname(CP2K_EXE)} && curl -Ls -o {CP2K_EXE}.part {CP2K_URL}'
        f' && chmod +x {CP2K_EXE}.part && mv {CP2K_EXE}.part {CP2K_EXE}')


def _install_python_packages():
    # Trajectories are shown with py3Dmol (see show_md) rather than nglview: Colab creates two views of
    # every custom widget (googlecolab/colabtools#2853), so nglview leaves the first frame on top of the
    # animation. py3Dmol writes plain HTML/JavaScript and needs no widget manager.
    missing = []
    for package, version in [('mdtraj', None), ('ase', None), ('py3Dmol', '2.5.5')]:
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed is None or (version is not None and installed != version):
            missing.append(package if version is None else f'{package}=={version}')
    if missing:
        print(f'Installing {" ".join(missing)}...', flush=True)
        _sh(f'{sys.executable} -m pip install -q {" ".join(missing)}')


def _set_environment():
    os.makedirs(BIN_DIR, exist_ok=True)
    links = {tool: f'{NNP_PREFIX}/bin/{tool}' for tool in NNP_TOOLS}
    links['cp2k'] = CP2K_EXE
    for name, target in links.items():
        link = os.path.join(BIN_DIR, name)
        if not os.path.lexists(link):
            os.symlink(target, link)
    if BIN_DIR not in os.environ['PATH'].split(':'):
        os.environ['PATH'] = BIN_DIR + ':' + os.environ['PATH']

    # Colab runs everything as root on 2 CPUs: let OpenMPI run as root and start more ranks than cores
    os.environ['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
    os.environ['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'
    os.environ['OMPI_MCA_rmaps_base_oversubscribe'] = '1'
    os.environ['PRTE_MCA_rmaps_default_mapping_policy'] = ':oversubscribe'

    # n2p2 links the threaded OpenBLAS: with a whole committee training at once, every member would start
    # its own BLAS threads, which spin against each other on 2 CPUs (QbC training ~9x slower)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    # make `import aml` work, now and after a runtime restart. A plain sys.path.insert() is not enough:
    # %run puts this script's directory (= TP_DIR) on sys.path and removes it again when the script ends
    with open(os.path.join(site.getsitepackages()[0], 'tp_mlip.pth'), 'w') as f:
        f.write(TP_DIR + '\n')
    sys.path.append(TP_DIR)


def run_cp2k(fn_input, directory='.', fn_output='output', refresh=15):
    """Run a CP2K simulation in `directory` and wait for it to finish.

    The last line of the energy file (`*-1.ener`) is printed every `refresh` seconds to follow progress.
    If CP2K fails, the end of its output file is printed and an exception is raised.
    """

    directory = os.path.abspath(directory)
    for fn in glob.glob(os.path.join(directory, '*-1.ener')):
        os.remove(fn)

    print(f'Running CP2K on {fn_input} in {directory}')
    print('#     Step Nr.          Time[fs]        Kin.[a.u.]          Temp[K]            Pot.[a.u.]', flush=True)
    t_start = time.time()
    with open(os.path.join(directory, fn_output + '.log'), 'w') as log:
        process = subprocess.Popen(['cp2k', '-i', fn_input, '-o', fn_output], cwd=directory, stdout=log, stderr=log)
        while process.poll() is None:
            time.sleep(refresh)
            fns_ener = glob.glob(os.path.join(directory, '*-1.ener'))
            if fns_ener:
                with open(fns_ener[0]) as f:
                    lines = f.readlines()
                if len(lines) > 1:
                    print(lines[-1].rstrip()[:96], flush=True)

    if process.returncode != 0:
        for fn in [fn_output, fn_output + '.log']:
            if os.path.exists(os.path.join(directory, fn)):
                with open(os.path.join(directory, fn)) as f:
                    print(''.join(f.readlines()[-30:]))
        raise RuntimeError(f'CP2K failed (exit code {process.returncode}), see {directory}/{fn_output}')
    print(f'Done in {(time.time() - t_start) / 60:.1f} min.')


def show_md(system, max_frames=300, max_atom_frames=100000, width=640, height=420, interval=60):
    """Show a trajectory (mdtraj.Trajectory) or a single structure (ase.Atoms) in 3D, with py3Dmol.

    A trajectory plays in a loop, one frame every `interval` ms. Long trajectories are thinned to at
    most `max_frames` frames and `max_atom_frames` atoms x frames, to keep the notebook light.
    Bonds are guessed from distances in each frame. Coordinates are shown as they are, without
    wrapping them back into the periodic cell.
    """
    import py3Dmol

    if hasattr(system, 'xyz'):
        # mdtraj trajectory, positions in nm
        symbols = [atom.element.symbol for atom in system.topology.atoms]
        frames = 10 * system.xyz
    else:
        # ase Atoms, positions in Angstrom
        symbols = system.get_chemical_symbols()
        frames = system.get_positions()[None]
    n_frames, n_atoms = frames.shape[:2]

    stride = max(1, math.ceil(n_frames / max_frames), math.ceil(n_frames * n_atoms / max_atom_frames))
    frames = frames[::stride]
    if stride > 1:
        print(f'Showing {len(frames)} of {n_frames} frames (one in {stride}).')

    blocks = []
    for positions in frames:
        lines = [str(n_atoms), '']
        lines += [f'{s} {x:.3f} {y:.3f} {z:.3f}' for s, (x, y, z) in zip(symbols, positions)]
        blocks.append('\n'.join(lines))

    view = py3Dmol.view(width=width, height=height)
    view.addModelsAsFrames('\n'.join(blocks) + '\n', 'xyz')
    view.setStyle({'stick': {'radius': 0.12}, 'sphere': {'scale': 0.22}})
    if len(frames) > 1:
        view.animate({'loop': 'forward', 'interval': interval})
    view.zoomTo()
    return view


_install_n2p2()
_install_cp2k()
_install_python_packages()
_set_environment()
print('Setup complete.')
