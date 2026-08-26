# A6000/rootless Docker environment

This directory records two reproducible ways to restore the SimpleNet
environment on the Ubuntu 24.04.3 A6000 server. The host Ubuntu patch version
does not have to equal the container patch version; the host NVIDIA driver and
NVIDIA Container Toolkit are the compatibility boundary.

## Environment recovered from the workstation

- Image: `june6423/simplenet-rtx50:latest`
- Immutable repository digest:
  `sha256:9efed934c9ad7f4b66cacc1cffd33c1de6c63bf2b4ae7228e0e32051a98abb6b`
- Base family: NVIDIA PyTorch 25.01, Ubuntu 24.04, Python 3.12
- PyTorch: `2.6.0a0+ecf3bae40a.nv25.01`
- torchvision: `0.20.0a0`
- CUDA toolkit/runtime: 12.8, cuDNN 9.7
- Existing container shared memory: 30 GiB

The old image is about 27 GB unpacked because the NVIDIA base also contains
TensorRT, RAPIDS, Jupyter, and development tooling. Repository imports do not
show that 4ind requires those additional stacks; they come from the base image.

## Host/rootless prerequisite

Check the driver before pulling a large image:

```bash
nvidia-smi
docker info | sed -n '/Security Options/,+6p'
```

NVIDIA PyTorch 25.01 uses CUDA 12.8. NVIDIA documents driver 570 or newer as
the general requirement. Do not infer driver compatibility from Ubuntu
24.04.3 or from the A6000 model alone.

If `docker info` shows rootless mode but `docker run --gpus all` cannot see the
GPU, configure the NVIDIA runtime for the user daemon:

```bash
nvidia-ctk runtime configure \
  --runtime=docker \
  --config="$HOME/.config/docker/daemon.json"
systemctl --user restart docker
sudo nvidia-ctk config \
  --set nvidia-container-cli.no-cgroups \
  --in-place
```

The final command changes the host NVIDIA runtime configuration and therefore
requires administrator authority. It should only be used when the server
administrator confirms that the machine is intended to use the rootless
NVIDIA configuration.

## Path A: restore the exact experiment image

The Docker Hub manifest was confirmed to exist on 2026-08-26. Pulling by
digest preserves the image used by `anormal-rtx50`, even if the `latest` tag is
changed later.

```bash
docker pull \
  june6423/simplenet-rtx50@sha256:9efed934c9ad7f4b66cacc1cffd33c1de6c63bf2b4ae7228e0e32051a98abb6b
```

Verify both GPUs and the recovered software versions:

```bash
docker run --rm --gpus all \
  june6423/simplenet-rtx50@sha256:9efed934c9ad7f4b66cacc1cffd33c1de6c63bf2b4ae7228e0e32051a98abb6b \
  python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.get_device_name(0), torch.cuda.device_count())'
```

## Path B: rebuild from the recorded Dockerfile

The Dockerfile pins the linux/amd64 NVIDIA base digest and the Python package
versions observed in the working container. Use the small `docker` directory
as the build context so result folders and datasets are not sent to Docker:

```bash
docker build \
  --file docker/Dockerfile.a6000 \
  --tag simplenet-a6000:rebuild \
  docker
```

This path is reconstructable from source, but it is not byte-identical to the
published custom image: package indexes and Ubuntu package revisions can
change. The pinned base and Python versions reduce, but do not eliminate, that
difference.

## Run with explicit mounts

The runner mounts only the repository, dataset, result directory, and model
cache. The dataset is read-only. By default the container is ephemeral while
the result directory remains on the host.

Open a shell using the exact image:

```bash
./docker/run_a6000_rootless.sh \
  /server/path/to/4ind_dataset_202608 \
  /server/path/to/results_fourind_bestval
```

Run all six method/product jobs directly:

```bash
./docker/run_a6000_rootless.sh \
  /server/path/to/4ind_dataset_202608 \
  /server/path/to/results_fourind_bestval \
  ./run_fourind_bestval_multigpu.sh /data/4ind /results 0,1
```

Use the rebuilt image instead:

```bash
SIMPLENET_IMAGE=simplenet-a6000:rebuild \
  ./docker/run_a6000_rootless.sh \
  /server/path/to/4ind_dataset_202608 \
  /server/path/to/results_fourind_bestval
```

The command above assumes that two GPUs are visible inside the container. If
the server exposes a different count, pass the corresponding IDs to
`run_fourind_bestval_multigpu.sh`. Set `SIMPLENET_KEEP_CONTAINER=1` only when a
persistent named debugging container is needed; otherwise all experiment state
is already external to the container.
