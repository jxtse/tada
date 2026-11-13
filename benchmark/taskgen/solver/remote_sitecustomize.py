"""Use a configured digest-identical mirror for Harbor's kernel probe image."""
try:
    from local_config import get
    from harbor.environments.docker.docker import DockerEnvironment
    mirror = get('remote_mirror_image')
    if mirror:
        DockerEnvironment._EGRESS_CONTROL_KERNEL_PROBE_IMAGE = mirror
except ImportError:
    pass
